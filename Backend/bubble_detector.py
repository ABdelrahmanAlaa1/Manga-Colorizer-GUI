"""
YOLO-based Speech Bubble Detector for manga pages.

Primary model: kitsumed/yolov8m_seg-speech-bubble (segmentation)
  - Outputs pixel-perfect bubble masks via result.masks
  - Contours follow actual bubble shape (not just rectangles)

Fallback model: ogkalu/comic-speech-bubble-detector-yolov8m (detection only)
  - Outputs bounding boxes only
  - Used if seg model download fails

Classes: 'speech_bubble' (seg model has single class)
"""

import os
import numpy as np
import cv2

_detector = None


class BubbleDetector:
    """YOLO speech bubble detector with segmentation support."""

    # Primary: segmentation model (pixel-perfect masks)
    SEG_HF_REPO = "kitsumed/yolov8m_seg-speech-bubble"
    SEG_MODEL_FILENAME = "model.pt"
    SEG_LOCAL_NAME = "yolov8m-seg-speech-bubble.pt"

    # Fallback: detection-only model (boxes only)
    DET_HF_REPO = "ogkalu/comic-speech-bubble-detector-yolov8m"
    DET_MODEL_FILENAME = "comic-speech-bubble-detector.pt"

    def __init__(self, model_path=None, device=None):
        self._model = None
        self._model_path = model_path
        self._device = device
        self._is_seg_model = False  # True if using segmentation model

    def _ensure_model(self):
        """Lazy-load YOLO model, preferring seg model, falling back to det."""
        if self._model is not None:
            return

        from ultralytics import YOLO

        models_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "models"
        )
        os.makedirs(models_dir, exist_ok=True)

        # Determine device
        device = self._device
        if device is None:
            try:
                import torch
                device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            except ImportError:
                device = 'cpu'

        # If user provided explicit path, use it
        if self._model_path and os.path.isfile(self._model_path):
            print(f"[YOLO] Loading custom model from {self._model_path}...")
            self._model = YOLO(self._model_path)
            self._model.to(device)
            # Detect if it's a seg model by checking model type
            self._is_seg_model = hasattr(self._model.model, 'model') and \
                any('Segment' in str(type(m)) for m in self._model.model.model.modules())
            print(f"[YOLO] Custom model loaded (seg={self._is_seg_model}).")
            return

        # Try segmentation model first
        seg_path = os.path.join(models_dir, self.SEG_LOCAL_NAME)
        if self._try_load_model(seg_path, self.SEG_HF_REPO,
                                self.SEG_MODEL_FILENAME, device,
                                is_seg=True):
            return

        # Fall back to detection-only model
        det_path = os.path.join(models_dir, self.DET_MODEL_FILENAME)
        if self._try_load_model(det_path, self.DET_HF_REPO,
                                self.DET_MODEL_FILENAME, device,
                                is_seg=False):
            return

        raise RuntimeError("[YOLO] Could not load any bubble detection model")

    def _try_load_model(self, local_path, hf_repo, hf_filename, device,
                        is_seg=False):
        """Try to load a model from local path or download from HF."""
        from ultralytics import YOLO

        model_type = "segmentation" if is_seg else "detection"

        # Download if not cached
        if not os.path.isfile(local_path):
            print(f"[YOLO] Downloading {model_type} model from {hf_repo}...")
            try:
                from huggingface_hub import hf_hub_download
                downloaded = hf_hub_download(
                    repo_id=hf_repo,
                    filename=hf_filename,
                    local_dir=os.path.dirname(local_path),
                )
                # Rename if needed (seg model downloads as model.pt)
                if os.path.basename(downloaded) != os.path.basename(local_path):
                    final_path = local_path
                    if os.path.isfile(downloaded) and not os.path.isfile(final_path):
                        os.rename(downloaded, final_path)
                    local_path = final_path
                else:
                    local_path = downloaded
                print(f"[YOLO] Downloaded to {local_path}")
            except Exception as e:
                print(f"[YOLO] HF download failed for {model_type}: {e}")
                return False

        if not os.path.isfile(local_path):
            return False

        try:
            print(f"[YOLO] Loading {model_type} model on {device}...")
            self._model = YOLO(local_path)
            self._model.to(device)
            self._is_seg_model = is_seg
            print(f"[YOLO] {model_type.capitalize()} model ready "
                  f"(seg_masks={'yes' if is_seg else 'no'}).")
            return True
        except Exception as e:
            print(f"[YOLO] Failed to load {model_type} model: {e}")
            self._model = None
            return False

    @property
    def has_segmentation(self):
        """Whether the loaded model provides per-pixel masks."""
        return self._is_seg_model

    def detect(self, image, conf=0.3, imgsz=1024):
        """
        Detect speech bubbles in an image.

        Parameters
        ----------
        image : np.ndarray
            BGR or grayscale image.
        conf : float
            Minimum confidence threshold.
        imgsz : int
            Inference image size.

        Returns
        -------
        list of dict
            Each dict has: 'bbox' [x1,y1,x2,y2], 'class' str,
            'confidence' float, 'mask' np.ndarray (if seg model).
        """
        self._ensure_model()

        # YOLO expects BGR 3-channel input
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        # Ensure contiguous array for YOLO
        image = np.ascontiguousarray(image)

        results = self._model.predict(
            image, imgsz=imgsz, conf=conf, verbose=False
        )

        bubbles = []
        if results and len(results) > 0:
            result = results[0]
            h_img, w_img = image.shape[:2]

            for idx, box in enumerate(result.boxes):
                cls_id = int(box.cls)
                cls_name = result.names.get(cls_id, f"class_{cls_id}")
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                bubble = {
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'class': cls_name,
                    'confidence': float(box.conf),
                }

                # Extract per-pixel mask if segmentation model
                if (self._is_seg_model and result.masks is not None
                        and idx < len(result.masks)):
                    try:
                        # Use .xy which gives polygon coordinates already
                        # in original image space (accounts for letterbox
                        # padding transform — avoids shift on non-square images)
                        xy_poly = result.masks.xy[idx]
                        if len(xy_poly) >= 3:
                            mask_np = np.zeros((h_img, w_img), dtype=np.uint8)
                            pts = xy_poly.astype(np.int32).reshape(-1, 1, 2)
                            cv2.fillPoly(mask_np, [pts], 255)
                            bubble['mask'] = mask_np
                    except Exception as e:
                        print(f"[YOLO] Mask extraction failed for bubble {idx}: {e}")

                bubbles.append(bubble)

        return bubbles

    def create_bubble_mask(self, image_shape, bubbles):
        """
        Create a binary mask from detected bubbles.

        Uses per-pixel segmentation masks when available (seg model),
        falls back to filled rectangles (det model).

        Parameters
        ----------
        image_shape : tuple
            (H, W) of the target image.
        bubbles : list of dict
            Output from detect().

        Returns
        -------
        np.ndarray
            Binary mask (255 inside bubbles, 0 outside).
        """
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        for b in bubbles:
            if 'mask' in b and b['mask'] is not None:
                # Use pixel-perfect segmentation mask
                seg_mask = b['mask']
                if seg_mask.shape[:2] == (h, w):
                    mask = cv2.max(mask, seg_mask)
                else:
                    # Resize if shape mismatch
                    resized = cv2.resize(seg_mask, (w, h),
                                        interpolation=cv2.INTER_LINEAR)
                    mask = cv2.max(mask, (resized > 127).astype(np.uint8) * 255)
            else:
                # Fallback: filled rectangle
                x1, y1, x2, y2 = b['bbox']
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

        return mask

    def free(self):
        """Release YOLO model VRAM."""
        if self._model is not None:
            self._model = None
            import gc
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass


def get_bubble_detector():
    """Get or create the global BubbleDetector singleton."""
    global _detector
    if _detector is None:
        _detector = BubbleDetector()
    return _detector


def free_bubble_detector():
    """Release the global detector."""
    global _detector
    if _detector is not None:
        _detector.free()
        _detector = None
