"""
Manga Text Detection Engine — Detection-only OCR for manga text masking.

Runs OCR once per image, caches all intermediate results,
and exposes the final mask for manual editing without re-running OCR.

Supports PP-OCRv5 detection-only backend via PaddleX.
English-only detection to avoid Japanese SFX false positives.
"""

import os
import sys
import threading
import time
import hashlib
import numpy as np
import cv2


def _dbg(msg):
    """Write debug message to file — guaranteed visible."""
    _dir = os.path.join(os.path.dirname(__file__), '..', 'ocr_debug')
    os.makedirs(_dir, exist_ok=True)
    _path = os.path.join(_dir, 'scoring_log.txt')
    with open(_path, 'a', encoding='utf-8') as f:
        f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")

# Fix for PaddleOCR 3.5 + modelscope crash
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'

# Suppress invisible matplotlib windows spawned by PaddleOCR/PaddleX
os.environ['MPLBACKEND'] = 'Agg'

# Suppress PaddleX/VisualDL GUI windows
os.environ['VISUALDL_SERVER_HOST'] = '127.0.0.1'
os.environ['FLAGS_use_mkldnn'] = '0'
os.environ.setdefault('PADDLE_DISABLE_VISUALDL', '1')
os.environ.setdefault('PADDLE_NO_CUDA_BLOCKING', '1')

# --- Monkey-patch subprocess.Popen to suppress console windows on Windows ---
# PaddlePaddle's cpp_extension calls `where ccache` via subprocess which spawns
# visible terminal windows. This patch adds CREATE_NO_WINDOW flag to all Popen
# calls made by PaddlePaddle internals.
if sys.platform == 'win32':
    import subprocess as _subprocess
    _orig_popen_init = _subprocess.Popen.__init__

    def _silent_popen_init(self, *args, **kwargs):
        if 'creationflags' not in kwargs:
            kwargs['creationflags'] = _subprocess.CREATE_NO_WINDOW
        return _orig_popen_init(self, *args, **kwargs)

    _subprocess.Popen.__init__ = _silent_popen_init

try:
    import matplotlib
    matplotlib.use('Agg')
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Detection Result Container
# ---------------------------------------------------------------------------
class DetectionResult:
    """Immutable snapshot of one OCR detection run."""
    __slots__ = (
        'image_path', 'image_hash', 'raw_boxes', 'bubble_mask',
        'sfx_mask', 'combined_mask', 'timestamp', 'image_shape',
    )

    def __init__(self, image_path, image_hash, raw_boxes, bubble_mask,
                 sfx_mask, combined_mask, image_shape):
        self.image_path = image_path
        self.image_hash = image_hash
        self.raw_boxes = raw_boxes
        self.bubble_mask = bubble_mask
        self.sfx_mask = sfx_mask
        self.combined_mask = combined_mask
        self.image_shape = image_shape
        self.timestamp = time.time()


# ---------------------------------------------------------------------------
# Main Detector Class
# ---------------------------------------------------------------------------
class MangaTextDetector:
    """
    Detection-only OCR for manga text masking.

    Usage::

        detector = MangaTextDetector(engine='paddle')
        result = detector.detect(grayscale_image, config)
        mask = result.combined_mask  # ready for luminance transfer
    """

    def __init__(self):
        self._paddle_reader = None
        self._cache = {}          # image_hash -> DetectionResult
        self._cache_max = 8       # keep last N results
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def detect(self, L_source, config=None):
        """
        Run YOLO-primary detection pipeline on a grayscale image.

        Pipeline:
          Step 0: YOLO bubble detection
          Step 1: Post-process (validate + filter FPs)
          Step 2: Contour refinement (bbox → actual shape)
          Step 3: (optional, 'full' mode) OCR text → SFX extraction
          Step 4: Combine + smooth

        Returns a DetectionResult with bubble_mask, sfx_mask, combined_mask.
        """
        current_path = getattr(config, 'current_image_path', None)
        if current_path and os.path.exists(current_path):
            image_hash = f"{current_path}_{os.path.getmtime(current_path)}"
        else:
            image_hash = self._fast_hash(L_source)

        log_cb = getattr(config, 'ocr_status_callback', None) if config else None
        _dbg(f"detect() called, path={current_path}, hash={image_hash}")

        if log_cb:
            log_cb(f"[OCR] detect() called for {current_path}")

        # Check cache
        cached = self._cache.get(image_hash)
        if cached is not None:
            if cached.image_shape == L_source.shape:
                _dbg("CACHE HIT — skipping scoring")
                if log_cb:
                    log_cb("[OCR] Using cached detection result")
                return cached
            else:
                _dbg(f"CACHE HIT (with resize) — resizing from {cached.image_shape} to {L_source.shape}")
                if log_cb:
                    log_cb(f"[OCR] CACHE HIT (with resize) — resizing from {cached.image_shape} to {L_source.shape}")
                resized_res = self._resize_detection_result(cached, L_source.shape, current_path, image_hash)
                self._cache_put(image_hash, resized_res)
                return resized_res

        _dbg("CACHE MISS — acquiring lock for pipeline execution")

        with self._lock:
            # Double-check cache inside lock
            cached = self._cache.get(image_hash)
            if cached is not None:
                if cached.image_shape == L_source.shape:
                    _dbg("CACHE HIT (double-checked) — skipping scoring")
                    return cached
                else:
                    _dbg(f"CACHE HIT (with resize, double-checked) — resizing from {cached.image_shape} to {L_source.shape}")
                    resized_res = self._resize_detection_result(cached, L_source.shape, current_path, image_hash)
                    self._cache_put(image_hash, resized_res)
                    return resized_res
            return self._detect_locked(L_source, config, image_hash, current_path, log_cb)

    def _detect_locked(self, L_source, config, image_hash, current_path, log_cb):
        _dbg("CACHE MISS — running full pipeline under lock")

        if log_cb:
            log_cb("[OCR] Cache miss — running full detection pipeline under lock")

        # Read config
        export_ocr = getattr(config, 'export_ocr_debug', False) if config else False
        yolo_raw = getattr(config, 'use_yolo_bubbles', 'full') if config else 'full'
        fp_strictness = getattr(config, 'fp_strictness', 0.5) if config else 0.5

        # Normalize yolo mode
        if yolo_raw is True or yolo_raw == 'full':
            run_ocr = True
        elif yolo_raw == 'yolo_only':
            run_ocr = False
        elif yolo_raw is False or yolo_raw == 'off':
            # No detection at all — return empty mask
            empty = np.zeros(L_source.shape[:2], dtype=np.uint8)
            result = DetectionResult(
                image_path=current_path, image_hash=image_hash,
                image_shape=L_source.shape, raw_boxes=[],
                bubble_mask=empty, sfx_mask=empty, combined_mask=empty,
            )
            self._cache[image_hash] = result
            return result
        else:
            run_ocr = True  # default to full

        h, w = L_source.shape[:2]
        img_area = h * w
        _profile = {}  # phase_name -> elapsed_seconds
        _t_pipeline = time.time()

        # ---- Step 0a: YOLO bubble detection ----
        if log_cb:
            log_cb("Detecting speech bubbles (YOLO)...")
        _t = time.time()
        yolo_bubbles, yolo_bubble_mask = self._detect_yolo_bubbles(
            L_source, config
        )
        _profile['0a_yolo_detect'] = time.time() - _t

        # ---- Step 0b: OCR text detection (BEFORE validation) ----
        text_mask = np.zeros((h, w), dtype=np.uint8)
        raw_boxes = []
        scored_boxes = []

        if run_ocr:
            if log_cb:
                log_cb("Detecting text regions (OCR)...")
            _t = time.time()
            raw_boxes, scored_boxes = self._detect_text(L_source, config)
            _profile['0b_ocr_detect'] = time.time() - _t

            # Build text mask from OCR boxes
            _t = time.time()
            for bbox in raw_boxes:
                pts = np.array(bbox, np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(text_mask, [pts], 255)
            _profile['0c_text_mask_build'] = time.time() - _t

        # ---- Step 1: Cross-validate bubbles using text_mask ----
        _t = time.time()
        if yolo_bubbles is not None:
            validated = self._postprocess_bubbles(
                yolo_bubbles, L_source, text_mask, fp_strictness, log_cb
            )
            if log_cb:
                removed = len(yolo_bubbles) - len(validated)
                log_cb(f"Validated {len(validated)} bubbles "
                       f"({removed} FPs removed)")
        else:
            validated = []
        _profile['1_bubble_validate'] = time.time() - _t

        # ---- Step 2: Contour refinement ----
        _t = time.time()
        if validated:
            if log_cb:
                log_cb("Refining bubble contours...")
            bubble_mask = self._refine_contours(L_source, validated)
        else:
            bubble_mask = np.zeros((h, w), dtype=np.uint8)
        _profile['2_contour_refine'] = time.time() - _t

        # ---- Step 3: SFX extraction ----
        sfx_mask = np.zeros((h, w), dtype=np.uint8)

        if run_ocr:
            # SFX extraction (floating text outside bubbles)
            if log_cb:
                log_cb("Processing SFX regions...")
            _t = time.time()
            sfx_mask = self._detect_floating_sfx(
                L_source, text_mask, bubble_mask, config, scored_boxes
            )
            _profile['3_sfx_extract'] = time.time() - _t

        # ---- Step 4: Combine + smooth ----
        _t = time.time()
        combined_mask = cv2.max(bubble_mask, sfx_mask)
        blur_ksize = self._kernel_scale_odd(9, w)
        combined_mask = cv2.GaussianBlur(
            combined_mask, (blur_ksize, blur_ksize), 0
        )
        _profile['4_combine_smooth'] = time.time() - _t

        # ---- Debug overlays ----
        if export_ocr:
            _t = time.time()
            self._save_debug_overlays(
                L_source, text_mask, bubble_mask, sfx_mask,
                combined_mask, config,
                yolo_bubble_mask=yolo_bubble_mask,
            )
            _profile['5_debug_overlays'] = time.time() - _t

        # ---- Cache ----
        result = DetectionResult(
            image_path=current_path,
            image_hash=image_hash,
            raw_boxes=raw_boxes,
            bubble_mask=bubble_mask,
            sfx_mask=sfx_mask,
            combined_mask=combined_mask,
            image_shape=L_source.shape,
        )
        self._cache_put(image_hash, result)

        total_elapsed = time.time() - _t_pipeline
        _profile['TOTAL'] = total_elapsed

        mode_str = "full" if run_ocr else "yolo_only"
        print(f"[OCR] {mode_str}: {len(validated)} bubbles, "
              f"{len(raw_boxes)} text boxes, "
              f"mask={np.count_nonzero(combined_mask)} px, "
              f"total={total_elapsed:.1f}s")

        # ---- Write profiling report ----
        self._write_profiling(current_path, _profile, h, w,
                              len(validated), len(raw_boxes))

        # Models stay warm for subsequent images (VRAM-aware lifecycle).
        # Call ensure_vram() or free_all_models() externally when VRAM is needed.

        return result

    def _write_profiling(self, image_path, profile, h, w,
                         n_bubbles, n_boxes):
        """Write profiling waterfall to ocr_debug/profiling.txt."""
        _dir = os.path.join(os.path.dirname(__file__), '..', 'ocr_debug')
        os.makedirs(_dir, exist_ok=True)
        _path = os.path.join(_dir, 'profiling.txt')

        total = profile.get('TOTAL', 0)

        lines = []

        # Session separator: insert if file is empty or last write > 60s ago
        try:
            if os.path.exists(_path):
                mtime = os.path.getmtime(_path)
                if time.time() - mtime > 60:
                    lines.append(f"\n{'─'*65}")
                    lines.append(f"  ── NEW SESSION ──  {time.strftime('%Y-%m-%d %H:%M:%S')}")
                    lines.append(f"{'─'*65}")
        except Exception:
            pass

        lines.append(f"\n{'='*65}")
        lines.append(f"  PROFILING: {os.path.basename(image_path or 'unknown')}")
        lines.append(f"  {time.strftime('%Y-%m-%d %H:%M:%S')}  |  {w}x{h}  |  "
                      f"{n_bubbles} bubbles  |  {n_boxes} text boxes")
        lines.append(f"{'='*65}")
        lines.append(f"  {'Phase':<30} {'Time':>8} {'%':>6}  Bar")
        lines.append(f"  {'-'*30} {'-'*8} {'-'*6}  {'-'*16}")

        for phase, elapsed in sorted(profile.items()):
            if phase == 'TOTAL':
                continue
            pct = (elapsed / total * 100) if total > 0 else 0
            bar = '█' * int(pct / 3) + '░' * max(0, 33 - int(pct / 3))
            lines.append(f"  {phase:<30} {elapsed:>7.2f}s {pct:>5.1f}%  {bar}")

        lines.append(f"  {'-'*30} {'-'*8} {'-'*6}")
        lines.append(f"  {'TOTAL':<30} {total:>7.2f}s {'100%':>6}")
        lines.append(f"{'='*65}\n")

        report = '\n'.join(lines)
        print(report)  # Also print to console

        with open(_path, 'a', encoding='utf-8') as f:
            f.write(report)

        # Truncate to last 2000 lines to prevent unbounded growth
        try:
            with open(_path, 'r', encoding='utf-8') as f:
                all_lines = f.readlines()
            if len(all_lines) > 2000:
                with open(_path, 'w', encoding='utf-8') as f:
                    f.writelines(all_lines[-2000:])
        except Exception:
            pass

    def invalidate_cache(self, image_path=None):
        """Clear cache for a specific image or all."""
        if image_path is None:
            self._cache.clear()
        else:
            to_remove = [
                k for k, v in self._cache.items()
                if v.image_path == image_path
            ]
            for k in to_remove:
                del self._cache[k]

    def free_readers(self):
        """Release OCR reader VRAM."""
        if self._paddle_reader is not None:
            self._paddle_reader = None
            import gc
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

    def _free_yolo(self):
        """Release YOLO model VRAM if loaded."""
        try:
            from Backend.bubble_detector import free_bubble_detector
            free_bubble_detector()
        except Exception:
            pass

    def ensure_vram(self, needed_mb=1500):
        """Free detection models only if VRAM is insufficient.

        Called before GPU-heavy operations (colorization, upscale) to
        ensure there is enough free VRAM. If sufficient VRAM exists,
        models stay warm for the next image.
        """
        try:
            import torch
            if not torch.cuda.is_available():
                return
            free_bytes, _ = torch.cuda.mem_get_info()
            free_mb = free_bytes // (1024 * 1024)
            if free_mb >= needed_mb:
                return  # Plenty of room — keep models warm
        except Exception:
            pass  # Can't check — free defensively
        self.free_readers()
        self._free_yolo()

    def free_all_models(self):
        """Nuclear option: free all detection models unconditionally."""
        self.free_readers()
        self._free_yolo()

    # ------------------------------------------------------------------
    # Text Detection Backends
    # ------------------------------------------------------------------
    def _detect_text(self, L_source, config):
        """
        Run PP-OCRv5 det-only model and return bounding-box polygons
        in original image coordinates.

        Scales down if needed, delegates to _detect_paddle, then filters.
        """
        max_ocr_dim = getattr(config, 'max_ocr_dimension', 3072) if config else 3072

        h, w = L_source.shape[:2]
        longest = max(h, w)

        # Dynamic scaling — only downscale, never upscale
        t_scale = time.time()
        scale_factor = 1.0
        ocr_source = L_source
        if longest > max_ocr_dim:
            scale_factor = max_ocr_dim / float(longest)
            new_w = int(w * scale_factor)
            new_h = int(h * scale_factor)
            ocr_source = cv2.resize(
                L_source, (new_w, new_h),
                interpolation=cv2.INTER_LANCZOS4
            )
        t_scale = time.time() - t_scale

        # Run detection backend
        t0 = time.time()
        detections = self._detect_paddle(ocr_source, scale_factor, config)
        t_infer = time.time() - t0

        # ---- False positive filtering ----
        t_fp = time.time()
        filtered = self._filter_false_positives(detections, L_source)
        t_fp = time.time() - t_fp

        filtered_boxes = [box for box, _ in filtered]
        print(f"[OCR] Raw: {len(detections)} → {len(filtered_boxes)} boxes "
              f"(scale={t_scale:.2f}s infer={t_infer:.2f}s "
              f"filter={t_fp:.2f}s)")

        return filtered_boxes, filtered  # (boxes_only, scored_tuples)

    def _detect_paddle(self, ocr_source, scale_factor, config):
        """PP-OCRv5 det-only backend.

        Pass 1: Original image (PP-OCRv5 handles most text natively).
        Pass 2: CLAHE fallback only if pass 1 found very few boxes.
        No inverted pass — PP-OCRv5 handles light-on-dark natively.
        """
        reader = self._get_paddle_reader(config)
        if reader is None:
            return []

        detections = []

        # Pass 1: Normal image
        ocr_bgr = cv2.cvtColor(ocr_source, cv2.COLOR_GRAY2BGR)
        detections.extend(self._paddle_extract_boxes(
            reader, ocr_bgr, scale_factor
        ))

        # Pass 2: CLAHE only if pass 1 found few boxes
        # With lower PaddleX thresholds, more raw boxes come through;
        # CLAHE helps catch faint text that even lower thresholds miss.
        if len(detections) < 5:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            ocr_enhanced = clahe.apply(ocr_source)
            enhanced_bgr = cv2.cvtColor(ocr_enhanced, cv2.COLOR_GRAY2BGR)
            detections.extend(self._paddle_extract_boxes(
                reader, enhanced_bgr, scale_factor
            ))

        return detections

    def _paddle_extract_boxes(self, reader, bgr_image, scale_factor):
        """
        Extract detection polygons from PP-OCRv5 det-only model.

        Returns (box, det_score) tuples.  det_score measures detection
        confidence ("is this text?") which is more useful for our SFX trust
        system than rec_score ("what does the text say?").
        """
        detections = []
        try:
            for res in reader.predict(bgr_image):
                polys = res.get('dt_polys', np.array([]))
                scores = res.get('dt_scores', [])
                if len(polys) == 0:
                    continue
                for idx, poly in enumerate(polys):
                    poly_arr = np.array(poly, dtype=np.float64)
                    if scale_factor != 1.0:
                        poly_arr = poly_arr / scale_factor
                    score = scores[idx] if idx < len(scores) else None
                    detections.append(
                        (poly_arr.astype(np.int32).tolist(), score)
                    )
        except Exception as e:
            print(f"[OCR] Detection failed: {e}")
            import traceback
            traceback.print_exc()
        return detections


    # ------------------------------------------------------------------
    # False Positive Filtering (4-layer)
    # ------------------------------------------------------------------
    def _filter_false_positives(self, detections, L_source):
        """
        Multi-layer filter to remove false positive detections.

        Parameters
        ----------
        detections : list of (box, det_score)
        L_source : grayscale image

        Returns
        -------
        list of (box, score) — filtered
        """
        if not detections:
            return []

        h, w = L_source.shape[:2]
        image_area = h * w

        # --- Layer 1: Detection Score Filtering ---
        scored = []
        for box, score in detections:
            if score is not None and score < 0.01:
                continue  # Definitely not text — extremely low det confidence
            scored.append((box, score))

        # --- Layer 2: Geometric Heuristics ---
        geometric = []
        for box, score in scored:
            pts = np.array(box, np.int32)
            x, y, bw, bh = cv2.boundingRect(pts)
            area = bw * bh

            # Too small (noise) or too large (panels/backgrounds)
            # Relaxed: let more through, FP system handles artefacts
            min_area = image_area * 0.00003
            max_area = image_area * 0.20
            if area < min_area or area > max_area:
                continue

            # Extreme aspect ratio (panel borders, lines)
            # Raised from 12→30: vertical manga text has extreme ratios
            aspect = max(bw, bh) / max(min(bw, bh), 1)
            if aspect > 30:
                continue

            geometric.append((box, score))

        # --- Layer 3: Luminance Interior Check ---
        luma_filtered = []
        for box, score in geometric:
            pts = np.array(box, np.int32)
            x, y, bw, bh = cv2.boundingRect(pts)
            # Clamp to image bounds
            x = max(0, x)
            y = max(0, y)
            x2 = min(w, x + bw)
            y2 = min(h, y + bh)
            if x2 <= x or y2 <= y:
                continue

            roi = L_source[y:y2, x:x2]
            mean_luma = float(roi.mean())
            std_luma = float(roi.std())

            # Any scored box passes — SFX system handles FP filtering
            if score is not None and score >= 0.01:
                luma_filtered.append((box, score))
                continue

            # Only scoreless boxes get luma check (very strict)
            # Reject obvious non-text: very low contrast interiors
            if std_luma < 5:
                continue  # Perfectly uniform = not text

            luma_filtered.append((box, score))

        # --- Layer 4: NMS Deduplication ---
        if len(luma_filtered) > 1:
            final = self._nms_boxes_scored(luma_filtered, iou_threshold=0.5)
        else:
            final = luma_filtered

        return final

    @staticmethod
    def _nms_boxes(boxes, iou_threshold=0.5):
        """Non-Maximum Suppression on polygon bounding boxes."""
        if not boxes:
            return []

        # Convert to axis-aligned bounding rects for IoU
        rects = []
        for box in boxes:
            pts = np.array(box, np.int32)
            x, y, bw, bh = cv2.boundingRect(pts)
            rects.append((x, y, x + bw, y + bh, bw * bh))

        # Sort by area (largest first)
        indices = sorted(range(len(rects)), key=lambda i: rects[i][4], reverse=True)
        keep = []
        suppressed = set()

        for i in indices:
            if i in suppressed:
                continue
            keep.append(i)
            x1_a, y1_a, x2_a, y2_a, area_a = rects[i]

            for j in indices:
                if j <= i or j in suppressed:
                    continue
                x1_b, y1_b, x2_b, y2_b, area_b = rects[j]

                # Intersection
                ix1 = max(x1_a, x1_b)
                iy1 = max(y1_a, y1_b)
                ix2 = min(x2_a, x2_b)
                iy2 = min(y2_a, y2_b)
                iw = max(0, ix2 - ix1)
                ih = max(0, iy2 - iy1)
                inter = iw * ih

                # Union
                union = area_a + area_b - inter
                if union > 0 and float(inter) / union > iou_threshold:
                    suppressed.add(j)

        return [boxes[i] for i in keep]

    @staticmethod
    def _nms_boxes_scored(scored_items, iou_threshold=0.5):
        """NMS on (box, score) tuples — preserves scores."""
        if not scored_items:
            return []

        boxes = [item[0] for item in scored_items]
        rects = []
        for box in boxes:
            pts = np.array(box, np.int32)
            x, y, bw, bh = cv2.boundingRect(pts)
            rects.append((x, y, x + bw, y + bh, bw * bh))

        indices = sorted(range(len(rects)), key=lambda i: rects[i][4], reverse=True)
        keep = []
        suppressed = set()

        for i in indices:
            if i in suppressed:
                continue
            keep.append(i)
            x1_a, y1_a, x2_a, y2_a, area_a = rects[i]

            for j in indices:
                if j <= i or j in suppressed:
                    continue
                x1_b, y1_b, x2_b, y2_b, area_b = rects[j]
                ix1 = max(x1_a, x1_b)
                iy1 = max(y1_a, y1_b)
                ix2 = min(x2_a, x2_b)
                iy2 = min(y2_a, y2_b)
                iw = max(0, ix2 - ix1)
                ih = max(0, iy2 - iy1)
                inter = iw * ih
                union = area_a + area_b - inter
                if union > 0 and float(inter) / union > iou_threshold:
                    suppressed.add(j)

        return [scored_items[i] for i in keep]

    # ------------------------------------------------------------------
    # Reader Initialization
    # ------------------------------------------------------------------
    def _get_paddle_reader(self, config=None):
        """Lazy-initialize PP-OCRv5 detection-only model (no recognition).

        We only need text LOCATION (bounding polygons), not the actual text.
        Using PaddleX's detection model directly is ~100x faster than
        PaddleOCR's full det+rec pipeline.
        """
        if self._paddle_reader is not None:
            return self._paddle_reader

        try:
            import torch
            has_gpu = torch.cuda.is_available()
        except ImportError:
            has_gpu = False

        # Model tier: 'server' (accurate) or 'mobile' (faster, less accurate)
        model_tier = getattr(config, 'ocr_model_tier', 'server') if config else 'server'
        if model_tier not in ('server', 'mobile'):
            model_tier = 'server'

        det_model = f'PP-OCRv5_{model_tier}_det'

        try:
            from paddlex import create_predictor

            # ---- GPU Acceleration Flags ----
            if has_gpu:
                import os
                os.environ['NVIDIA_TF32_OVERRIDE'] = '1'
                try:
                    import paddle
                    paddle.set_flags({
                        'FLAGS_cudnn_exhaustive_search': False,
                        'FLAGS_conv_workspace_size_limit': 4096,
                    })
                    print("[OCR] GPU accel: TF32 enabled (cuDNN benchmark disabled for speed)")
                except Exception:
                    pass

            device = 'gpu:0' if has_gpu else 'cpu'
            print(f"[OCR] Initializing {det_model} (det-only) on {device}...")
            t_init = time.time()
            self._paddle_reader = create_predictor(
                model_name=det_model,
                device=device,
            )
            t_init = time.time() - t_init

            # ---- Detection sensitivity tuning for manga ----
            # PaddleX defaults are conservative (box_thresh=0.6, thresh=0.3,
            # resize_long=960). Lowering these catches faint/low-contrast
            # text; our FP filter system handles any extra false positives.
            self._paddle_reader.thresh = 0.15         # default 0.3
            self._paddle_reader.box_thresh = 0.3      # default 0.6 — #1 miss cause
            self._paddle_reader.unclip_ratio = 2.5    # default 1.5, boosted to 2.5 to fix early bounding cutoffs
            # Make our max_ocr_dimension slider the SOLE resize controller.
            # PaddleX default resize_long=960 silently overrides our slider.
            # Set to match slider max so PaddleX never resizes again unnecessarily.
            max_dim = getattr(config, 'max_ocr_dimension', 3072) if config else 3072
            self._paddle_reader.limit_side_len = max_dim
            self._paddle_reader.limit_type = 'max'

            print(f"[OCR] {det_model} loaded on {device} in {t_init:.1f}s "
                  f"(det-only, no rec, thresh=0.15, box_thresh=0.3, unclip_ratio=2.5, "
                  f"limit={self._paddle_reader.limit_side_len})")

        except Exception as e:
            import traceback
            print(f"[OCR] Failed to load det model: {e}")
            traceback.print_exc()
            self._paddle_reader = None
        return self._paddle_reader


    # ------------------------------------------------------------------
    # YOLO Bubble Detection
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Bubble Validation & FP Removal
    # ------------------------------------------------------------------
    def _score_bubble(self, bubble, L_source, ocr_text_mask,
                      fp_strictness=0.5, log_cb=None):
        """
        Score a single YOLO bubble using weighted multi-factor analysis.

        Instead of binary pass/fail checks, computes a composite score
        from 5 factors.  The dominant signal is OCR text overlap (35%),
        which rescues valid dark bubbles and eliminates bright non-text FPs.

        fp_strictness: 0.0 = permissive, 1.0 = aggressive FP removal.
        Returns (score: float, is_valid: bool).
        """
        x1, y1, x2, y2 = bubble['bbox']
        conf = bubble.get('confidence', 0.5)
        h, w = L_source.shape[:2]
        img_area = h * w

        bw = x2 - x1
        bh = y2 - y1
        if bw <= 0 or bh <= 0:
            return 0.0, False

        area = bw * bh

        # ---- Hard rejects (geometry, not scoreable) ----
        if area < 200:  # Tiny noise
            return 0.0, False
        if area > img_area * 0.30:  # Panel-sized, not a bubble
            return 0.0, False
        ratio = max(bw, bh) / max(min(bw, bh), 1)
        if ratio > 6.0:  # Extreme aspect ratio
            return 0.0, False

        # Clamp ROI to image bounds
        cx1 = max(0, x1)
        cy1 = max(0, y1)
        cx2 = min(w, x2)
        cy2 = min(h, y2)
        roi = L_source[cy1:cy2, cx1:cx2]
        if roi.size == 0:
            return 0.0, False

        score = 0.0
        has_text_data = (ocr_text_mask is not None and ocr_text_mask.any())

        # ------ Hard reject: very low YOLO confidence ------
        if conf < 0.02:
            return 0.0, False

        # ------ Factor 1 (25%): Tiered YOLO confidence ------
        if conf >= 0.90:
            score += 0.25 * 1.0
        elif conf >= 0.70:
            score += 0.25 * (conf / 0.90)
        elif conf >= 0.30:
            score += 0.25 * (conf / 0.90) * 0.6
        else:
            score += 0.25 * (conf / 0.90) * 0.3

        # ------ Factor 2 (25%): OCR text overlap ------
        text_overlap = 0.0
        if has_text_data:
            text_roi = ocr_text_mask[cy1:cy2, cx1:cx2]
            text_pixels = int(np.count_nonzero(text_roi))
            text_overlap = text_pixels / max(area, 1)
            score += 0.25 * min(text_overlap / 0.03, 1.0)

        # ------ Factor 3 (10%): Brightness ------
        mean_brightness = float(roi.mean())
        brightness_score = float(np.clip(
            (mean_brightness - 100.0) / 100.0, 0.0, 1.0
        ))
        score += 0.10 * brightness_score

        # ------ Factor 4 (10%): Edge density ------
        edges = cv2.Canny(roi, 50, 150)
        edge_ratio = float(edges.mean()) / 255.0
        edge_score = 1.0 - min(edge_ratio / 0.25, 1.0)
        score += 0.10 * edge_score

        # ------ Factor 5 (10%): White-fill ratio ------
        white_pixels = int(np.count_nonzero(roi > 200))
        white_ratio = white_pixels / max(roi.size, 1)
        score += 0.10 * min(white_ratio / 0.3, 1.0)

        # ------ Factor 6 (10%): Interior uniformity ------
        # Bubbles = uniform white interior (low stdev)
        # Faces/detail = complex patterns (high stdev)
        roi_std = float(roi.std())
        # Bubbles typically have stdev < 40; faces/detail > 60
        uniformity = float(np.clip(1.0 - (roi_std - 20.0) / 60.0, 0.0, 1.0))
        score += 0.10 * uniformity

        # ------ Factor 7 (10%): Shape regularity ------
        # Bubbles are roughly convex. Irregular shapes = likely FP.
        if 'mask' in bubble and bubble['mask'] is not None:
            seg = bubble['mask']
            seg_coords = cv2.findNonZero(seg)
            if seg_coords is not None and len(seg_coords) > 5:
                hull = cv2.convexHull(seg_coords)
                hull_area = cv2.contourArea(hull)
                seg_area_px = np.count_nonzero(seg)
                solidity = seg_area_px / max(hull_area, 1)
                # Bubbles are convex (solidity > 0.85)
                # Complex shapes (solidity < 0.6) are likely FP
                shape_score = float(np.clip(
                    (solidity - 0.5) / 0.4, 0.0, 1.0
                ))
            else:
                shape_score = 0.5
        else:
            shape_score = 0.5  # No mask, neutral
        score += 0.10 * shape_score

        # ------ Adaptive threshold ------
        threshold = 0.30 + fp_strictness * 0.15

        # No OCR text data → raise threshold
        if not has_text_data:
            threshold += 0.10

        # Compound FP penalty: no text + high detail = NOT a bubble
        # regardless of YOLO confidence
        if text_overlap < 0.005:
            # No text inside — need higher bar
            if has_text_data:
                threshold += 0.08

            # High interior detail = complex art, not uniform white bubble
            if roi_std > 60:
                threshold += 0.15  # Heavy penalty
            elif roi_std > 45:
                threshold += 0.08  # Moderate penalty

            # Low confidence compounds with no text
            if conf < 0.70:
                threshold += 0.10

        # Debug scoring detail
        x1, y1, x2, y2 = bubble['bbox']
        print(f"    [DETAIL] bbox=({x1},{y1},{x2},{y2}) conf={conf:.2f} "
              f"score={score:.3f} thr={threshold:.3f} "
              f"text={text_overlap:.4f} bright={mean_brightness:.0f} "
              f"std={roi_std:.1f} shape={shape_score:.2f} "
              f"{'PASS' if score >= threshold else 'FAIL'}")
        if log_cb:
            log_cb(f"    [SCORE] conf={conf:.2f} s={score:.3f}/t={threshold:.3f} "
                   f"std={roi_std:.0f} shape={shape_score:.2f} "
                   f"{'PASS' if score >= threshold else 'FAIL'}")
        _dbg(f"SCORE bbox=({x1},{y1},{x2},{y2}) conf={conf:.2f} "
             f"score={score:.3f} thr={threshold:.3f} "
             f"text={text_overlap:.4f} bright={mean_brightness:.0f} "
             f"std={roi_std:.1f} shape={shape_score:.2f} "
             f"{'PASS' if score >= threshold else 'FAIL'}")

        return score, score >= threshold

    def _postprocess_bubbles(self, yolo_bubbles, L_source,
                             ocr_text_mask=None, fp_strictness=0.5,
                             log_cb=None):
        """
        Filter YOLO bubble detections using multi-factor scoring.

        Returns a list of validated bubbles sorted by score (highest first).
        """
        validated = []
        rejected = []
        for bubble in yolo_bubbles:
            score, is_valid = self._score_bubble(
                bubble, L_source, ocr_text_mask, fp_strictness, log_cb
            )
            conf = bubble.get('confidence', 0)
            x1, y1, x2, y2 = bubble['bbox']

            if is_valid:
                bubble['_score'] = score
                validated.append(bubble)
            else:
                rejected.append((conf, score))

        if log_cb:
            log_cb(f"[OCR] Scoring: {len(validated)} accepted, "
                   f"{len(rejected)} rejected / {len(yolo_bubbles)} total")

        # Sort by composite score (highest first)
        validated.sort(key=lambda b: b.get('_score', 0), reverse=True)
        return validated

    # ------------------------------------------------------------------
    # YOLO Bubble Detection
    # ------------------------------------------------------------------
    def _detect_yolo_bubbles(self, L_source, config):
        """
        Run YOLO speech bubble detector.

        Supports segmentation model (pixel masks) and detection model
        (bounding boxes only).  When both are available and config
        requests 'both', merges results from seg + det models.

        Returns (bubbles_list, raw_mask) or (None, None).
        """
        try:
            from Backend.bubble_detector import get_bubble_detector
            detector = get_bubble_detector()

            t0 = time.time()
            # Lowering YOLO conf to 0.15 safely catches faint bubbles. 
            # OCR postprocess validates and removes any false positives.
            bubbles = detector.detect(L_source, conf=0.15, imgsz=1024)
            elapsed = time.time() - t0

            seg_status = 'seg' if detector.has_segmentation else 'det'
            has_masks = any('mask' in b for b in (bubbles or []))

            if bubbles:
                raw_mask = detector.create_bubble_mask(
                    L_source.shape, bubbles
                )
                print(f"[YOLO] Detected {len(bubbles)} bubbles in "
                      f"{elapsed:.1f}s (model={seg_status}, "
                      f"masks={'yes' if has_masks else 'no'})")
                # YOLO model stays warm (VRAM-aware lifecycle)
                return bubbles, raw_mask
            else:
                print(f"[YOLO] No bubbles detected ({elapsed:.1f}s, "
                      f"model={seg_status})")
                return None, None

        except ImportError as e:
            print(f"[YOLO] Not available: {e}")
            return None, None
        except Exception as e:
            print(f"[YOLO] Detection failed: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def _refine_contours(self, L_source, validated_bubbles):
        """
        Refine validated YOLO detections into actual bubble contours.

        When YOLO segmentation masks are available (seg model), uses
        the pixel-perfect masks directly — no threshold guessing needed.

        When only bounding boxes are available (det model), falls back
        to Otsu-based threshold refinement within each bbox ROI.

        Returns a refined bubble mask (uint8, same size as L_source).
        """
        h, w = L_source.shape[:2]
        refined_mask = np.zeros((h, w), dtype=np.uint8)
        ks = lambda base: self._kernel_scale(base, w)
        used_seg = 0
        used_threshold = 0

        for bubble in validated_bubbles:
            # ---- Path A: Pixel-perfect contours guided by YOLO ----
            # YOLO bbox+15% = WHERE to search (ensures full bubble coverage)
            # Path B's threshold+contour = HOW to find edges (pixel-perfect)
            # Seg mask = WHICH contour is the right bubble (overlap test)
            if 'mask' in bubble and bubble['mask'] is not None:
                seg_mask = bubble['mask']
                if seg_mask.shape[:2] != (h, w):
                    seg_mask = cv2.resize(seg_mask, (w, h),
                                          interpolation=cv2.INTER_LINEAR)
                    seg_mask = (seg_mask > 127).astype(np.uint8) * 255

                # Use bbox expanded by 15% to ensure full bubble coverage
                x1, y1, x2, y2 = bubble['bbox']
                bw = x2 - x1
                bh = y2 - y1
                expand_x = max(ks(10), int(bw * 0.15))
                expand_y = max(ks(10), int(bh * 0.15))
                rx1 = max(0, x1 - expand_x)
                ry1 = max(0, y1 - expand_y)
                rx2 = min(w, x2 + expand_x)
                ry2 = min(h, y2 + expand_y)

                if rx2 - rx1 < 10 or ry2 - ry1 < 10:
                    refined_mask = cv2.max(refined_mask, seg_mask)
                    used_seg += 1
                    continue

                roi = L_source[ry1:ry2, rx1:rx2]
                seg_roi = seg_mask[ry1:ry2, rx1:rx2]

                # Same proven threshold approach as Path B
                _, roi_binary = cv2.threshold(
                    roi, 230, 255, cv2.THRESH_BINARY
                )

                white_ratio = np.count_nonzero(roi_binary) / max(roi_binary.size, 1)
                if white_ratio < 0.15:
                    otsu_thresh, roi_binary = cv2.threshold(
                        roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                    )
                    if otsu_thresh < 150:
                        roi_binary = cv2.adaptiveThreshold(
                            roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                            cv2.THRESH_BINARY, 21, 5
                        )

                # Morphological open to clean noise
                morph_k = max(3, ks(3))
                kernel = np.ones((morph_k, morph_k), np.uint8)
                roi_binary = cv2.morphologyEx(
                    roi_binary, cv2.MORPH_OPEN, kernel
                )

                # Find contours — pixel-perfect edges
                contours, _ = cv2.findContours(
                    roi_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                if not contours:
                    # No contours — fall back to raw seg mask
                    refined_mask = cv2.max(refined_mask, seg_mask)
                    used_seg += 1
                    continue

                # Pick the contour with the MOST overlap with seg mask
                # (seg mask tells us which bright region is the bubble)
                best_contour = None
                best_overlap = 0
                roi_area = roi.shape[0] * roi.shape[1]

                for c in contours:
                    c_area = cv2.contourArea(c)
                    if c_area < roi_area * 0.03:
                        continue  # Skip tiny noise

                    # Draw this contour and check overlap with seg mask
                    c_mask = np.zeros_like(seg_roi)
                    cv2.drawContours(c_mask, [c], -1, 255, -1)
                    overlap = np.count_nonzero(
                        cv2.bitwise_and(c_mask, seg_roi)
                    )

                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_contour = c

                if best_contour is not None:
                    offset_contour = best_contour.copy()
                    offset_contour[:, :, 0] += rx1
                    offset_contour[:, :, 1] += ry1
                    cv2.drawContours(
                        refined_mask, [offset_contour], -1, 255, -1
                    )
                else:
                    # No matching contour — fall back to seg mask
                    refined_mask = cv2.max(refined_mask, seg_mask)

                used_seg += 1
                continue

            # ---- Path B: Threshold-based refinement (det model) ----
            x1, y1, x2, y2 = bubble['bbox']
            pad = ks(10)
            rx1 = max(0, x1 - pad)
            ry1 = max(0, y1 - pad)
            rx2 = min(w, x2 + pad)
            ry2 = min(h, y2 + pad)

            if rx2 - rx1 < 10 or ry2 - ry1 < 10:
                cv2.rectangle(refined_mask, (x1, y1), (x2, y2), 255, -1)
                used_threshold += 1
                continue

            roi = L_source[ry1:ry2, rx1:rx2]

            # Use fixed 230 threshold (proven stable for white bubbles)
            # with Otsu fallback for gray bubbles
            _, roi_binary = cv2.threshold(roi, 230, 255, cv2.THRESH_BINARY)

            # Check if the fixed threshold found enough white area
            white_ratio = np.count_nonzero(roi_binary) / max(roi_binary.size, 1)
            if white_ratio < 0.15:
                # Not enough white — try Otsu for gray bubbles
                otsu_thresh, roi_binary = cv2.threshold(
                    roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
                if otsu_thresh < 150:
                    roi_binary = cv2.adaptiveThreshold(
                        roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv2.THRESH_BINARY, 21, 5
                    )

            # Morphological open to clean noise
            morph_k = max(3, ks(3))
            kernel = np.ones((morph_k, morph_k), np.uint8)
            roi_binary = cv2.morphologyEx(roi_binary, cv2.MORPH_OPEN, kernel)

            # Find contours in ROI
            contours, _ = cv2.findContours(
                roi_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            if not contours:
                cv2.rectangle(refined_mask, (x1, y1), (x2, y2), 255, -1)
                used_threshold += 1
                continue

            # Pick the largest contour
            best_contour = None
            best_area = 0
            roi_area = (rx2 - rx1) * (ry2 - ry1)

            for c in contours:
                area = cv2.contourArea(c)
                if area < roi_area * 0.05:
                    continue
                if area > best_area:
                    best_area = area
                    best_contour = c

            if best_contour is not None:
                offset_contour = best_contour.copy()
                offset_contour[:, :, 0] += rx1
                offset_contour[:, :, 1] += ry1
                cv2.drawContours(
                    refined_mask, [offset_contour], -1, 255, -1
                )
            else:
                cv2.rectangle(refined_mask, (x1, y1), (x2, y2), 255, -1)
            used_threshold += 1

        # Dilate slightly to cover anti-aliased edges
        dilate_k = ks(3)
        refined_mask = cv2.dilate(
            refined_mask,
            np.ones((dilate_k, dilate_k), np.uint8),
            iterations=1
        )
        print(f"[OCR] Contour refinement: {used_seg} seg masks, "
              f"{used_threshold} threshold-based")
        return refined_mask

    # ------------------------------------------------------------------
    # (geometry-based _detect_bubbles RETIRED in Phase 3A)
    # YOLO is now the sole bubble detector — see _detect_yolo_bubbles()
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Stroke Width Consistency Check (text vs art discriminator)
    # ------------------------------------------------------------------
    def _check_stroke_consistency(self, binary_roi):
        """
        Check if strokes have consistent width (text) vs varying (hatching/art).

        Uses distance transform to measure stroke width at each pixel.
        Text has consistent stroke width (low CV), art/hatching has varying widths.

        Returns (is_consistent: bool, cv_value: float).
        """
        if binary_roi is None or not np.any(binary_roi):
            return True, 0.0

        # Distance transform: value at each white pixel = distance to nearest black
        dist = cv2.distanceTransform(binary_roi, cv2.DIST_L2, 5)

        # Use morphological skeleton (thinning) to get medial axis
        # cv2.ximgproc.thinning is ideal but may not be available
        try:
            skeleton = cv2.ximgproc.thinning(binary_roi)
        except (AttributeError, cv2.error):
            # Fallback: erode until thin, or just use local maxima
            # Simple approach: sample from the distance transform directly
            # where distance > 0 (all foreground pixels)
            widths = dist[dist > 0]
            if len(widths) < 5:
                return True, 0.0
            cv_val = float(np.std(widths)) / max(float(np.mean(widths)), 0.1)
            return cv_val < 1.5, cv_val

        # Sample stroke widths along the skeleton
        widths = dist[skeleton > 0]
        if len(widths) < 5:
            return True, 0.0  # Too few samples to judge

        cv_val = float(np.std(widths)) / max(float(np.mean(widths)), 0.1)
        # Text: CV typically < 0.8-1.2 (consistent strokes)
        # Hatching: CV typically > 1.5 (crossing lines create varying widths)
        return cv_val < 1.2, cv_val

    # ------------------------------------------------------------------
    # Floating SFX Detection
    # ------------------------------------------------------------------
    def _detect_floating_sfx(self, L_source, text_mask, bubble_mask, config,
                             scored_boxes=None):
        """
        Extract floating SFX text (text outside speech bubbles).

        Pipeline:
        1. Per-box: use individual OCR boxes (with scores) for validation
        2. Stroke extraction: adaptive threshold → connected components
        3. Stroke-only mask: no convex hull, just actual ink + tiny dilation
        4. Confidence-based opacity: high-score → 80%, low-score → fades
        """
        if not np.any(text_mask):
            return np.zeros_like(L_source)

        ks = lambda base: self._kernel_scale(base, L_source.shape[1])
        log_cb = getattr(config, 'ocr_status_callback', None)

        h, w = L_source.shape[:2]



        min_stroke_area = ks(10)

        # Per-box opacity SFX mask (float accumulator)
        sfx_float = np.zeros((h, w), dtype=np.float32)

        # Use scored OCR boxes if available, otherwise fall back to text_mask contours
        if scored_boxes:
            box_list = []
            for box, score in scored_boxes:
                pts = np.array(box, np.int32)
                bx, by, bw, bh = cv2.boundingRect(pts)
                box_list.append((bx, by, bw, bh, pts, score))
        else:
            # Fallback: extract boxes from text_mask contours (no scores)
            contours, _ = cv2.findContours(
                text_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            box_list = []
            for cnt in contours:
                bx, by, bw, bh = cv2.boundingRect(cnt)
                box_list.append((bx, by, bw, bh, cnt, None))

        for bx, by, bw, bh, pts, score in box_list:
            # Clamp to image bounds
            x1 = max(0, bx)
            y1 = max(0, by)
            x2 = min(w, bx + bw)
            y2 = min(h, by + bh)
            if x2 - x1 < 5 or y2 - y1 < 5:
                continue

            bx, by, bw, bh = x1, y1, x2 - x1, y2 - y1
            box_area = bw * bh
            if box_area > (h * w) * 0.30:  # Panel-sized, highly likely false positive
                continue

            # -- Check: is this inside or near a bubble? Skip if so --
            # Dilate bubble mask to catch text near bubble edges (dialogue
            # that extends slightly beyond the bubble contour).
            if not hasattr(self, '_sfx_dilated_bubble_mask') or \
               self._sfx_dilated_bubble_src is not bubble_mask:
                dil_k = max(15, self._kernel_scale(15, w))
                self._sfx_dilated_bubble_mask = cv2.dilate(
                    bubble_mask,
                    np.ones((dil_k, dil_k), np.uint8),
                    iterations=1,
                )
                self._sfx_dilated_bubble_src = bubble_mask
            dilated_bm = self._sfx_dilated_bubble_mask
            bubble_overlap = np.count_nonzero(
                dilated_bm[by:by+bh, bx:bx+bw] > 127
            ) / max(box_area, 1)
            if bubble_overlap > 0.3:
                continue  # Inside or near a bubble, not floating SFX

            roi = L_source[by:by+bh, bx:bx+bw]
            roi_std = float(roi.std())

            # ============================================================
            # TIERED TRUST SYSTEM
            # ============================================================
            s = score if score is not None else 0.0

            if s >= 0.7:
                # ------ HIGH TRUST: OCR confirmed text ------
                # Use Otsu to cleanly extract text pixels from any background
                _, otsu_mask = cv2.threshold(
                    roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
                )
                # Also try inverse Otsu for white-on-dark text
                _, otsu_inv = cv2.threshold(
                    roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
                # Pick the one with fewer pixels (text = minority)
                otsu_px = np.count_nonzero(otsu_mask)
                inv_px = np.count_nonzero(otsu_inv)
                if otsu_px < inv_px and otsu_px > 0:
                    text_pixels = otsu_mask
                elif inv_px > 0:
                    text_pixels = otsu_inv
                else:
                    continue

                # Sanity check: text shouldn't cover > 70% of box
                text_density = np.count_nonzero(text_pixels) / max(box_area, 1)
                if text_density > 0.70:
                    _dbg(f"SFX REJECT high_trust_overdense density={text_density:.2f} "
                         f"box=({bx},{by},{bw},{bh})")
                    continue
                if text_density < 0.01:
                    continue

                # Connected component cleanup (remove tiny noise)
                num_labels, labels, cc_stats, _ = cv2.connectedComponentsWithStats(
                    text_pixels, connectivity=8
                )
                clean_roi = np.zeros_like(roi)
                for i in range(1, num_labels):
                    if cc_stats[i, cv2.CC_STAT_AREA] >= min_stroke_area:
                        clean_roi[labels == i] = 255

                if not np.any(clean_roi):
                    continue

                opacity = 0.80
                _dbg(f"SFX ACCEPT [HIGH] box=({bx},{by},{bw},{bh}) score={s:.3f} "
                     f"otsu_density={text_density:.2f} std={roi_std:.1f} "
                     f"opacity={opacity:.2f}")

            elif s >= 0.3:
                # ------ MEDIUM TRUST: needs validation ------
                # Localized adaptive thresholding on padded ROI to avoid boundary artifacts
                pad = 21
                x1_pad = max(0, bx - pad)
                y1_pad = max(0, by - pad)
                x2_pad = min(w, bx + bw + pad)
                y2_pad = min(h, by + bh + pad)

                padded_roi = L_source[y1_pad:y2_pad, x1_pad:x2_pad]

                # Perform adaptive threshold on padded ROI
                local_thresh_roi = cv2.adaptiveThreshold(
                    padded_roi, 255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV, 21, 10
                )
                local_thresh_white_roi = cv2.adaptiveThreshold(
                    padded_roi, 255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, 21, -10
                )

                # Extract unpadded ROI coordinates relative to padded ROI
                rx1 = bx - x1_pad
                ry1 = by - y1_pad

                roi_dark_strokes = local_thresh_roi[ry1:ry1+bh, rx1:rx1+bw]
                roi_white_strokes = local_thresh_white_roi[ry1:ry1+bh, rx1:rx1+bw]

                # Create local box mask (unpadded)
                roi_box_mask = np.zeros((bh, bw), dtype=np.uint8)
                if pts.ndim >= 2:
                    # Shift points to local ROI coordinates
                    pts_local = pts.reshape(-1, 2) - [bx, by]
                    cv2.fillPoly(roi_box_mask, [pts_local.astype(np.int32).reshape(-1, 1, 2)], 255)
                else:
                    roi_box_mask.fill(255)

                # Mask the local strokes
                roi_dark_strokes = cv2.bitwise_and(roi_dark_strokes, roi_box_mask)
                roi_white_strokes = cv2.bitwise_and(roi_white_strokes, roi_box_mask)

                dark_px = np.count_nonzero(roi_dark_strokes)
                white_px = np.count_nonzero(roi_white_strokes)
                roi_strokes = roi_dark_strokes if dark_px < white_px else roi_white_strokes

                # Noise floor + density cap
                stroke_px = np.count_nonzero(roi_strokes)
                density = stroke_px / max(box_area, 1)
                if density < 0.01 or density > 0.70:
                    _dbg(f"SFX REJECT [MED] density={density:.2f} "
                         f"box=({bx},{by},{bw},{bh})")
                    continue

                if roi_std < 20:
                    _dbg(f"SFX REJECT [MED] low_contrast std={roi_std:.1f} "
                         f"box=({bx},{by},{bw},{bh})")
                    continue

                # Connected component cleanup
                num_labels, labels, cc_stats, _ = cv2.connectedComponentsWithStats(
                    roi_strokes, connectivity=8
                )
                clean_roi = np.zeros_like(roi_strokes)
                for i in range(1, num_labels):
                    if cc_stats[i, cv2.CC_STAT_AREA] >= min_stroke_area:
                        clean_roi[labels == i] = 255

                if not np.any(clean_roi):
                    continue

                # Hatching / dense art rejection
                comp_areas = [cc_stats[i, cv2.CC_STAT_AREA]
                              for i in range(1, num_labels)
                              if cc_stats[i, cv2.CC_STAT_AREA] >= min_stroke_area]
                if len(comp_areas) >= 3:
                    cv_ratio = float(np.std(comp_areas)) / max(float(np.mean(comp_areas)), 1)
                    if density > 0.30 and cv_ratio < 0.4:
                        _dbg(f"SFX REJECT [MED] hatching density={density:.2f} "
                             f"cv={cv_ratio:.2f} box=({bx},{by},{bw},{bh})")
                        continue
                    if density > 0.40 and len(comp_areas) > 6:
                        _dbg(f"SFX REJECT [MED] dense_art density={density:.2f} "
                             f"n={len(comp_areas)} box=({bx},{by},{bw},{bh})")
                        continue

                # Stroke width consistency check — text has uniform widths
                is_text_stroke, stroke_cv = self._check_stroke_consistency(clean_roi)
                if not is_text_stroke:
                    _dbg(f"SFX REJECT [MED] inconsistent_strokes cv={stroke_cv:.2f} "
                         f"box=({bx},{by},{bw},{bh})")
                    continue

                # Opacity: exponential decay
                ratio = s / 0.9
                opacity = 0.70 * (ratio ** 2)
                _dbg(f"SFX ACCEPT [MED] box=({bx},{by},{bw},{bh}) score={s:.3f} "
                     f"density={density:.2f} std={roi_std:.1f} stroke_cv={stroke_cv:.2f} "
                     f"opacity={opacity:.2f}")

            else:
                # ------ LOW TRUST: safety net only ------
                # Minimal validation — just Otsu + very low opacity
                _, otsu_mask = cv2.threshold(
                    roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
                )
                _, otsu_inv = cv2.threshold(
                    roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                )
                otsu_px = np.count_nonzero(otsu_mask)
                inv_px = np.count_nonzero(otsu_inv)
                if otsu_px < inv_px and otsu_px > 0:
                    text_pixels = otsu_mask
                elif inv_px > 0:
                    text_pixels = otsu_inv
                else:
                    continue

                text_density = np.count_nonzero(text_pixels) / max(box_area, 1)
                if text_density > 0.60 or text_density < 0.01:
                    continue

                clean_roi = text_pixels
                opacity = 0.08  # Nearly invisible
                _dbg(f"SFX ACCEPT [LOW] box=({bx},{by},{bw},{bh}) score={s:.3f} "
                     f"density={text_density:.2f} opacity={opacity:.2f}")

            # ---- Apply stroke mask with opacity ----
            tiny_k = max(2, ks(2))
            strokes_dilated = cv2.dilate(
                clean_roi, np.ones((tiny_k, tiny_k), np.uint8), iterations=1
            )

            # Accumulate into float mask (max blend per pixel)
            roi_float = sfx_float[by:by+bh, bx:bx+bw]
            stroke_opacity = strokes_dilated.astype(np.float32) / 255.0 * opacity
            sfx_float[by:by+bh, bx:bx+bw] = np.maximum(roi_float, stroke_opacity)

        if not np.any(sfx_float > 0):
            return np.zeros_like(L_source)

        if log_cb:
            log_cb("Processing SFX regions...")

        # ---- Inpainting cloud (INVISIBLE — for background inpaint only) ----
        floating_binary = (sfx_float > 0).astype(np.uint8) * 255
        soft_k = ks(15)
        inpaint_cloud = cv2.dilate(
            floating_binary,
            np.ones((soft_k, soft_k), np.uint8),
            iterations=2
        )
        self._last_sfx_inpaint_cloud = inpaint_cloud

        # ---- Exponential decay feather (blur fix for colorizer bleed) ----
        feather_r = int(getattr(config, 'sfx_feather_radius', 0) if config else 0)
        if feather_r > 0:
            stroke_binary = (sfx_float > 0).astype(np.uint8)
            # Distance from each non-stroke pixel to nearest stroke pixel
            dist = cv2.distanceTransform(
                1 - stroke_binary, cv2.DIST_L2, 5
            ).astype(np.float32)

            # Decay constant: opacity ≈ 5% at d = feather_r
            k = np.log(20.0) / float(feather_r)

            # Propagate max opacity outward (dilated opacity source)
            ksize = feather_r * 2 + 1
            dilated_opacity = cv2.dilate(
                sfx_float,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize)),
            )

            # Apply exponential decay in the feather zone
            feather_zone = (dist > 0) & (dist <= feather_r)
            decay = np.exp(-k * dist)
            feather_values = (dilated_opacity * decay).astype(np.float32)

            sfx_float[feather_zone] = np.maximum(
                sfx_float[feather_zone],
                feather_values[feather_zone],
            )

        # ---- Convert float opacity to uint8 mask (0-255 range) ----
        sfx_mask = np.clip(sfx_float * 255.0, 0, 255).astype(np.uint8)

        return sfx_mask

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _kernel_scale(base_val, image_width):
        return max(base_val, int(base_val * (image_width / 1796.0)))

    @staticmethod
    def _kernel_scale_odd(base_val, image_width):
        val = MangaTextDetector._kernel_scale(base_val, image_width)
        return val if val % 2 != 0 else val + 1

    @staticmethod
    def _fast_hash(image):
        """Quick hash of image content for cache keying."""
        # Sample sparse pixels for speed instead of hashing entire image
        h, w = image.shape[:2]
        sample = image[::max(1, h // 32), ::max(1, w // 32)]
        return hashlib.md5(sample.tobytes()).hexdigest()

    def _cache_put(self, key, result):
        self._cache[key] = result
        # Evict oldest if over capacity
        while len(self._cache) > self._cache_max:
            oldest_key = min(
                self._cache, key=lambda k: self._cache[k].timestamp
            )
            del self._cache[oldest_key]

    def _resize_detection_result(self, cached, new_shape, current_path, image_hash):
        """Resize cached DetectionResult masks and scale bounding boxes to match new_shape."""
        new_h, new_w = new_shape[:2]
        old_h, old_w = cached.image_shape[:2]
        
        # Avoid division by zero
        scale_x = new_w / max(1, old_w)
        scale_y = new_h / max(1, old_h)
        
        bubble_mask = cv2.resize(cached.bubble_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        sfx_mask = cv2.resize(cached.sfx_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        combined_mask = cv2.resize(cached.combined_mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        
        scaled_boxes = []
        for bbox in cached.raw_boxes:
            scaled_bbox = []
            for pt in bbox:
                scaled_bbox.append([int(pt[0] * scale_x), int(pt[1] * scale_y)])
            scaled_boxes.append(scaled_bbox)
            
        return DetectionResult(
            image_path=current_path or cached.image_path,
            image_hash=image_hash,
            raw_boxes=scaled_boxes,
            bubble_mask=bubble_mask,
            sfx_mask=sfx_mask,
            combined_mask=combined_mask,
            image_shape=new_shape,
        )

    def _save_debug_overlays(self, L_source, text_mask, bubble_mask,
                             sfx_mask, combined_mask, config,
                             yolo_bubble_mask=None):
        """Write debug overlay images when export_ocr_debug is enabled."""
        ocr_debug_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "ocr_debug"
        )
        os.makedirs(ocr_debug_dir, exist_ok=True)

        current_path = getattr(config, 'current_image_path', None)
        basename = "debug"
        if current_path:
            basename = os.path.splitext(os.path.basename(current_path))[0]

        def overlay(mask, name, color_ch=2):
            vis = cv2.cvtColor(L_source, cv2.COLOR_GRAY2BGR)
            colored = np.zeros_like(vis)
            # Ensure mask is 2D
            m = mask.squeeze() if mask.ndim > 2 else mask
            colored[:, :, color_ch] = m
            out = cv2.addWeighted(vis, 0.5, colored, 0.5, 0)
            cv2.imwrite(
                os.path.join(ocr_debug_dir, f"{basename}_{name}.jpg"), out
            )

        overlay(text_mask, "01_raw_ocr_bboxes")
        overlay(bubble_mask, "02_speech_bubbles")
        overlay(sfx_mask, "03_floating_sfx")
        overlay(combined_mask, "04_final_combined")

        # YOLO bubbles in green channel for comparison
        if yolo_bubble_mask is not None:
            overlay(yolo_bubble_mask, "05_yolo_bubbles", color_ch=1)

        # Scoring detail overlay — annotate each YOLO bbox with score
        if yolo_bubble_mask is not None:
            score_vis = cv2.cvtColor(L_source, cv2.COLOR_GRAY2BGR)
            # Re-run scoring to get per-bubble scores for visualization
            yolo_raw = getattr(config, 'use_yolo_bubbles', 'full')
            if yolo_raw not in (False, 'off'):
                try:
                    from Backend.bubble_detector import get_bubble_detector
                    det = get_bubble_detector()
                    raw_bubs = det.detect(L_source, conf=0.15, imgsz=1024)
                    fp_s = getattr(config, 'fp_strictness', 0.5)
                    for b in (raw_bubs or []):
                        sc, valid = self._score_bubble(
                            b, L_source, text_mask, fp_s
                        )
                        bx1, by1, bx2, by2 = b['bbox']
                        color = (0, 255, 0) if valid else (0, 0, 255)
                        cv2.rectangle(score_vis, (bx1, by1),
                                      (bx2, by2), color, 2)
                        label = f"{sc:.2f}"
                        cv2.putText(
                            score_vis, label, (bx1, by1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            color, 1, cv2.LINE_AA
                        )
                except Exception:
                    pass
            cv2.imwrite(
                os.path.join(ocr_debug_dir,
                             f"{basename}_06_scoring_detail.jpg"),
                score_vis
            )

        # Log — count text_mask nonzero regions as proxy for box count
        num_boxes = np.count_nonzero(text_mask) // 100  # rough estimate
        with open(os.path.join(ocr_debug_dir, "log.txt"), "a") as f:
            f.write(
                f"[{time.strftime('%H:%M:%S')}] "
                f"path={current_path}, text_px={np.count_nonzero(text_mask)}, "
                f"bubbles={np.count_nonzero(bubble_mask)}, "
                f"yolo={'yes' if yolo_bubble_mask is not None else 'no'}\n"
            )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_detector = None


def get_detector():
    """Get or create the global MangaTextDetector singleton."""
    global _detector
    if _detector is None:
        _detector = MangaTextDetector()
    return _detector


def free_detector():
    """Release the global detector and its VRAM."""
    global _detector
    if _detector is not None:
        _detector.free_all_models()
        _detector = None
