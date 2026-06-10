import json
import os
import time

import PIL.Image
import numpy as np

from transfer_quality import (
    DEBUG_MASK_COLOR_MAP,
    DEBUG_MASK_KEYS,
    apply_debug_mask_visual,
    compute_transfer_debug_masks,
    mask_to_u8,
)

MASK_EXPORT_MODE = "full_strength_unweighted"
MASK_EXPORT_NOTE = "Mask overlays/exports always use detector masks at full intensity; suppression strength sliders do not weight exported masks."


def export_current_mask_bundle(
    target_dir,
    image_name,
    source_np,
    working_np,
    preview_colorized_pil,
    active_debug_mask,
    active_debug_mask_label,
    active_debug_mode,
    active_debug_opacity,
    source_image_path,
):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    bundle_dir = os.path.join(target_dir, f"{image_name}_masks_{timestamp}")
    os.makedirs(bundle_dir, exist_ok=True)

    raw_masks = compute_transfer_debug_masks(working_np)

    PIL.Image.fromarray(source_np).save(os.path.join(bundle_dir, "original_input.png"))
    PIL.Image.fromarray(working_np).save(os.path.join(bundle_dir, "working_input_after_denoise.png"))
    preview_colorized_pil.save(os.path.join(bundle_dir, "preview_colorized.png"))

    for mask_name in DEBUG_MASK_KEYS:
        mask_data = raw_masks.get(mask_name)
        if mask_data is None:
            continue

        full_strength_mask = np.clip(mask_data, 0.0, 1.0)

        raw_u8 = mask_to_u8(full_strength_mask)
        if raw_u8 is not None:
            PIL.Image.fromarray(raw_u8, mode="L").save(os.path.join(bundle_dir, f"mask_{mask_name}_raw.png"))

        overlay = apply_debug_mask_visual(
            working_np,
            full_strength_mask,
            mode="overlay",
            color=DEBUG_MASK_COLOR_MAP[mask_name],
            opacity=0.65,
        )
        mask_only = apply_debug_mask_visual(
            np.zeros_like(working_np),
            full_strength_mask,
            mode="mask_only",
            color=DEBUG_MASK_COLOR_MAP[mask_name],
            opacity=1.0,
        )
        PIL.Image.fromarray(overlay).save(os.path.join(bundle_dir, f"mask_{mask_name}_overlay.png"))
        PIL.Image.fromarray(mask_only).save(os.path.join(bundle_dir, f"mask_{mask_name}_only.png"))

    metadata = {
        "source_image": source_image_path,
        "exported_at": timestamp,
        "active_debug_mask": active_debug_mask,
        "active_debug_mask_label": active_debug_mask_label,
        "active_debug_mode": active_debug_mode,
        "active_debug_opacity": active_debug_opacity,
        "mask_export_mode": MASK_EXPORT_MODE,
        "mask_export_note": MASK_EXPORT_NOTE,
    }
    with open(os.path.join(bundle_dir, "metadata.json"), "w", encoding="utf-8") as meta_file:
        json.dump(metadata, meta_file, indent=4)

    return bundle_dir
