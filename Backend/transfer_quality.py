"""
Studio-quality color transfer using LAB space and Lineart Multiplication.

OCR-guided bubble/SFX detection is delegated to Backend.ocr_engine.
This module focuses purely on luminance transfer and chroma compositing.
"""

import PIL.Image
import PIL.ImageFilter
import numpy as np


F32 = np.float32
ZERO_F32 = F32(0.0)
ONE_F32 = F32(1.0)
INV_255_F32 = F32(1.0 / 255.0)
MAX_U8_F32 = F32(255.0)
CHROMA_CENTER_F32 = F32(127.5)


# ---------------------------------------------------------------------------
# Compatibility shims — callers that used the old module-level OCR functions
# are transparently redirected to the new ocr_engine module.
# ---------------------------------------------------------------------------
def get_ocr_reader(config=None):
    """Legacy shim — returns the OCR engine's reader for backward compat."""
    from Backend.ocr_engine import get_detector
    return get_detector()


def free_ocr_reader():
    """Legacy shim — frees OCR reader VRAM."""
    from Backend.ocr_engine import free_detector
    free_detector()
    return True


# ---------------------------------------------------------------------------
# Debug mask infrastructure (unchanged from original)
# ---------------------------------------------------------------------------
DEBUG_MASK_KEYS = ("edge", "ink", "screentone")
DEBUG_MASK_UI_VALUES = (
    "None",
    "Edge",
    "Line Ink",
    "Screentone",
)
DEBUG_MASK_COLOR_MAP = {
    "edge": (255, 96, 32),
    "ink": (64, 196, 255),
    "screentone": (244, 210, 44),
}


def debug_mask_label_to_key(label):
    normalized = str(label or "").strip().lower().replace("_", " ")
    normalized = " ".join(normalized.split())

    aliases = {
        "none": "none",
        "edge": "edge",
        "ink": "ink",
        "line ink": "ink",
        "line/ink": "ink",
        "line-ink": "ink",
        "screentone": "screentone",
        "screen tone": "screentone",
    }
    return aliases.get(normalized, "none")


def _compute_transfer_masks_from_luma(y):
    y = y.astype(np.float32, copy=False)
    sensitivity = F32(0.5)

    grad_y, grad_x = np.gradient(y)
    grad_mag = np.sqrt(grad_x * grad_x + grad_y * grad_y)
    grad_scale = F32(np.percentile(grad_mag, 95)) + F32(1e-6)
    grad_norm = (grad_mag / grad_scale).astype(np.float32, copy=False)
    edge_mask = np.clip((grad_norm - F32(0.15)) / F32(0.85), ZERO_F32, ONE_F32).astype(np.float32, copy=False)

    y_mean = (
        y
        + np.roll(y, 1, axis=0)
        + np.roll(y, -1, axis=0)
        + np.roll(y, 1, axis=1)
        + np.roll(y, -1, axis=1)
    ) / F32(5.0)
    hf = np.abs(y - y_mean).astype(np.float32, copy=False)
    tone_mask = (
        np.clip((hf - F32(0.01)) / F32(0.05), ZERO_F32, ONE_F32).astype(np.float32, copy=False)
        * (ONE_F32 - np.clip(edge_mask * F32(1.5), ZERO_F32, ONE_F32).astype(np.float32, copy=False))
    ).astype(np.float32, copy=False)

    y_u8 = np.clip(y * MAX_U8_F32, ZERO_F32, MAX_U8_F32).astype(np.uint8)
    local_mean = np.array(
        PIL.Image.fromarray(y_u8).filter(PIL.ImageFilter.BoxBlur(radius=2.5)),
        dtype=np.float32,
    ) * INV_255_F32

    stroke_delta = np.clip(local_mean - y, ZERO_F32, ONE_F32).astype(np.float32, copy=False)
    stroke_floor = F32(0.018) - F32(0.010) * sensitivity
    stroke_range = np.maximum(F32(0.035), F32(0.095) - F32(0.040) * sensitivity)
    stroke_gate = np.clip((stroke_delta - stroke_floor) / stroke_range, ZERO_F32, ONE_F32).astype(np.float32, copy=False)

    dark_binary_u8 = (y < (F32(0.52) + F32(0.10) * sensitivity)).astype(np.uint8) * 255
    dark_density = np.array(
        PIL.Image.fromarray(dark_binary_u8).filter(PIL.ImageFilter.BoxBlur(radius=3.0)),
        dtype=np.float32,
    ) * INV_255_F32

    sparse_dark_gate = np.clip((dark_density - F32(0.015)) / F32(0.10), ZERO_F32, ONE_F32).astype(np.float32, copy=False)
    dense_dark_gate = np.clip((F32(0.62) - dark_density) / F32(0.46), ZERO_F32, ONE_F32).astype(np.float32, copy=False)
    density_gate = (sparse_dark_gate * dense_dark_gate).astype(np.float32, copy=False)

    line_edge_gate = np.clip((edge_mask - F32(0.08)) / F32(0.92), ZERO_F32, ONE_F32).astype(np.float32, copy=False)
    line_dark_gate = np.clip((F32(0.40) - y) / F32(0.26), ZERO_F32, ONE_F32).astype(np.float32, copy=False)
    ink_mask = (line_edge_gate * np.maximum(line_dark_gate, stroke_gate * F32(0.90))).astype(np.float32, copy=False)
    ink_mask *= density_gate
    ink_mask *= ONE_F32 - np.clip(tone_mask * F32(0.95), ZERO_F32, ONE_F32).astype(np.float32, copy=False)

    ink_mask_u8 = np.clip(ink_mask * MAX_U8_F32, ZERO_F32, MAX_U8_F32).astype(np.uint8)
    ink_mask = np.array(
        PIL.Image.fromarray(ink_mask_u8).filter(PIL.ImageFilter.GaussianBlur(radius=0.35)),
        dtype=np.float32,
    ) * INV_255_F32

    return {
        "edge": np.clip(edge_mask, ZERO_F32, ONE_F32).astype(np.float32, copy=False),
        "ink": np.clip(ink_mask, ZERO_F32, ONE_F32).astype(np.float32, copy=False),
        "screentone": np.clip(tone_mask, ZERO_F32, ONE_F32).astype(np.float32, copy=False),
    }


# ---------------------------------------------------------------------------
# Main luminance transfer function
# ---------------------------------------------------------------------------
def transfer_luminance_from_source(source_rgb, colorized_rgb, config=None):
    """
    Studio-quality color transfer using LAB space and Lineart Multiplication.
    
    This method guarantees 100% preservation of painted shading and colors from the 
    colorized source, while stamping perfectly sharp, slightly color-blended lineart on top.
    
    OCR-guided bubble/SFX detection is handled by Backend.ocr_engine.MangaTextDetector.
    """
    if source_rgb.shape[2] != 3 or colorized_rgb.shape[2] != 3:
        return colorized_rgb

    import cv2

    chroma_resize_mode = getattr(config, "chroma_resize_mode", "LANCZOS") if config else "LANCZOS"
    resize_map = {
        "LANCZOS": cv2.INTER_LANCZOS4,
        "BICUBIC": cv2.INTER_CUBIC,
        "BILINEAR": cv2.INTER_LINEAR,
    }
    interp = resize_map.get(chroma_resize_mode, cv2.INTER_LANCZOS4)

    # Resize colorized to match source dimensions if needed
    if source_rgb.shape[:2] != colorized_rgb.shape[:2]:
        colorized_rgb = cv2.resize(colorized_rgb, (source_rgb.shape[1], source_rgb.shape[0]), interpolation=interp)

    # Convert to LAB color space
    source_lab = cv2.cvtColor(source_rgb, cv2.COLOR_RGB2LAB)
    colorized_lab = cv2.cvtColor(colorized_rgb, cv2.COLOR_RGB2LAB)

    L_source = source_lab[..., 0]
    L_comfy = colorized_lab[..., 0]
    a_comfy = colorized_lab[..., 1]
    b_comfy = colorized_lab[..., 2]

    # -----------------------------------------------------------------
    # User mask inpainting (manual edits from preview_tools)
    # -----------------------------------------------------------------
    user_mask = getattr(config, 'user_mask', None) if config else None
    if user_mask is not None:
        if user_mask.ndim == 3:
            user_mask = cv2.cvtColor(user_mask, cv2.COLOR_RGB2GRAY)
        # Resize mask if dimensions don't match colorized image
        if user_mask.shape[:2] != colorized_rgb.shape[:2]:
            user_mask = cv2.resize(
                user_mask,
                (colorized_rgb.shape[1], colorized_rgb.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
        _, binary_mask = cv2.threshold(user_mask, 127, 255, cv2.THRESH_BINARY)
        
        if np.any(binary_mask):
            print(f"[Transfer] User mask: {np.count_nonzero(binary_mask)} px "
                  f"(merged into bubble_mask for clean B&W restoration)")

    # -----------------------------------------------------------------
    # OCR-Guided Bubble/SFX Detection (delegated to ocr_engine)
    # -----------------------------------------------------------------
    log_cb = getattr(config, 'ocr_status_callback', None)

    try:
        from Backend.ocr_engine import get_detector
        detector = get_detector()
        result = detector.detect(L_source, config)
        bubble_mask = result.combined_mask
    except Exception as e:
        import traceback
        print(f"[OCR] Detection pipeline failed, using empty mask: {e}")
        traceback.print_exc()
        bubble_mask = np.zeros_like(L_source)

    # -----------------------------------------------------------------
    # SFX Inpainting (remove color bleed behind floating text)
    # -----------------------------------------------------------------
    if np.any(bubble_mask):
        # The ocr_engine already handles SFX extraction and feathering.
        # We just need to inpaint behind the SFX regions.
        sfx_only = np.zeros_like(L_source)
        try:
            sfx_only = result.sfx_mask
        except Exception:
            pass

        if np.any(sfx_only):
            _, sfx_binary = cv2.threshold(sfx_only, 127, 255, cv2.THRESH_BINARY)
            if np.any(sfx_binary):
                if log_cb:
                    log_cb("Inpainting SFX background...")
                L_comfy = cv2.inpaint(L_comfy, sfx_binary, 5, cv2.INPAINT_TELEA)
                a_comfy = cv2.inpaint(a_comfy, sfx_binary, 5, cv2.INPAINT_TELEA)
                b_comfy = cv2.inpaint(b_comfy, sfx_binary, 5, cv2.INPAINT_TELEA)

    if log_cb:
        log_cb("Applying final polish...")

    # Merge user mask edits into bubble_mask for clean B&W restoration
    # user_mask = OCR base mask + user edits (additions AND deletions)
    # So it REPLACES bubble_mask entirely — user subtractions take effect
    if user_mask is not None and np.any(binary_mask):
        if binary_mask.shape[:2] != bubble_mask.shape[:2]:
            binary_mask = cv2.resize(
                binary_mask,
                (bubble_mask.shape[1], bubble_mask.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
        bubble_mask = binary_mask  # Replace, not merge — allows FP deletion

    bubble_alpha = bubble_mask.astype(np.float32) / 255.0

    # -----------------------------------------------------------------
    # Luminance transfer (unchanged from original)
    # -----------------------------------------------------------------
    def kernel_scale(base_val):
        return max(base_val, int(base_val * (L_source.shape[1] / 1796.0)))

    def kernel_scale_odd(base_val):
        val = kernel_scale(base_val)
        return val if val % 2 != 0 else val + 1

    # Clean the ComfyUI image to remove blurry/misaligned lines
    comfy_clean_ksize = kernel_scale(5)
    kernel = np.ones((comfy_clean_ksize, comfy_clean_ksize), np.uint8)
    L_comfy_clean = cv2.dilate(L_comfy, kernel, iterations=1)

    # Smooth chroma to prevent white halos around lineart
    chroma_blur_ksize = kernel_scale_odd(5)
    a_comfy_clean = cv2.GaussianBlur(a_comfy, (chroma_blur_ksize, chroma_blur_ksize), 0)
    b_comfy_clean = cv2.GaussianBlur(b_comfy, (chroma_blur_ksize, chroma_blur_ksize), 0)

    # Normalize source lineart as multiply mask
    L_source_f = L_source.astype(np.float32)
    L_source_norm = np.clip(L_source_f / 230.0, 0.0, 1.0)

    # Multiply blend
    L_final_f = L_comfy_clean.astype(np.float32) * L_source_norm

    # Restore bubbles to original luminance
    L_final_f = L_final_f * (1.0 - bubble_alpha) + L_source_f * bubble_alpha
    L_final = np.clip(L_final_f, 0, 255).astype(np.uint8)

    # Fade out chroma inside bubbles
    a_comfy_clean_f = a_comfy_clean.astype(np.float32) * (1.0 - bubble_alpha) + 128.0 * bubble_alpha
    b_comfy_clean_f = b_comfy_clean.astype(np.float32) * (1.0 - bubble_alpha) + 128.0 * bubble_alpha
    a_comfy_clean = np.clip(a_comfy_clean_f, 0, 255).astype(np.uint8)
    b_comfy_clean = np.clip(b_comfy_clean_f, 0, 255).astype(np.uint8)

    # Recombine LAB channels
    merged_lab = np.stack([L_final, a_comfy_clean, b_comfy_clean], axis=-1)

    # Convert back to RGB
    result_rgb = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2RGB)
    return result_rgb


# ---------------------------------------------------------------------------
# Public debug utilities (unchanged)
# ---------------------------------------------------------------------------
def compute_transfer_debug_masks(source_rgb):
    """Return normalized debug masks used by color-transfer heuristics."""
    if source_rgb.ndim != 3 or source_rgb.shape[2] != 3:
        empty = np.zeros(source_rgb.shape[:2], dtype=np.float32)
        return {
            "edge": empty,
            "ink": empty,
            "screentone": empty,
        }

    source = source_rgb.astype(np.float32, copy=False) * INV_255_F32
    y = (
        F32(0.299) * source[..., 0]
        + F32(0.587) * source[..., 1]
        + F32(0.114) * source[..., 2]
    ).astype(np.float32, copy=False)
    return _compute_transfer_masks_from_luma(y)


def apply_debug_mask_visual(base_rgb, mask, mode="overlay", color=(255, 90, 50), opacity=0.55):
    """Visualize a mask either on top of image or on blank background."""
    if mask is None:
        return base_rgb

    mask = np.clip(mask.astype(np.float32, copy=False), ZERO_F32, ONE_F32)
    color_arr = np.array(color, dtype=np.float32).reshape(1, 1, 3)

    if mode == "mask_only":
        out = color_arr * mask[..., None]
    else:
        base = base_rgb.astype(np.float32, copy=False)
        blend = F32(np.clip(opacity, 0.0, 1.0)) * mask[..., None]
        out = base * (1.0 - blend) + color_arr * blend

    return np.clip(out, ZERO_F32, MAX_U8_F32).astype(np.uint8)


def mask_to_u8(mask):
    if mask is None:
        return None
    return np.clip(mask.astype(np.float32, copy=False) * MAX_U8_F32, ZERO_F32, MAX_U8_F32).astype(np.uint8)
