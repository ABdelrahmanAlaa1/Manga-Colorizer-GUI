import os
import sys
import warnings
# Suppress Triton's warning about not finding CUDA on startup (harmless check)
warnings.filterwarnings("ignore", category=UserWarning, module="triton")
import argparse
import copy
import time
import tkinter as tk
import numpy as np
import PIL.Image
import torch

# When launched via a GUI (pythonw), sys.stdout/stderr can be None.
# torch.compile/logging will try to write to them and crash if they are None.
if sys.stdout is None:
    sys.stdout = open(os.devnull, 'w')
if sys.stderr is None:
    sys.stderr = open(os.devnull, 'w')

# --- Constants ---
MAX_INPUT_WIDTH = 1792
MIN_INPUT_WIDTH = 256
DEFAULT_INPUT_WIDTH = 800
PREVIEW_SOURCE_CACHE_SIZE = 12
PREVIEW_DENOISE_CACHE_SIZE = 24
PREVIEW_RERUN_DEBOUNCE_MS = 280
MAX_LOG_BUFFER_SIZE = 50000
APP_CLOSE_WAIT_TIMEOUT_S = 12.0
PIPELINE_CPU_MPS_INSTANCE_CAPS = {'denoise': 4, 'colorize': 2, 'upscale': 1}
PIPELINE_PROFILE_CUSTOM = 'Custom'

QUALITY_PRESET_DEFAULTS = {
    "denoise_sigma": 25,
    "chroma_resize_mode": "BICUBIC",
    "edge_chroma_protection": True,
    "edge_chroma_strength": 60,
    "line_ink_protection": True,
    "line_ink_protection_strength": 40,
    "screentone_chroma_smoothing": True,
    "screentone_smoothing_strength": 40,
}

# --- Path Setup ---
backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'Backend'))
sys.path.insert(0, backend_path)

# Persist TorchInductor compiled-kernel cache so torch.compile only pays the
# compile cost once (not every launch).
os.environ.setdefault(
    "TORCHINDUCTOR_CACHE_DIR",
    os.path.join(os.path.dirname(__file__), ".inductor_cache"),
)

# --- Re-exports from Backend ---
from transfer_quality import (
    transfer_luminance_from_source as _backend_transfer_luminance_from_source,
    compute_transfer_debug_masks as _backend_compute_transfer_debug_masks,
    apply_debug_mask_visual as _backend_apply_debug_mask_visual,
    DEBUG_MASK_UI_VALUES,
    DEBUG_MASK_COLOR_MAP,
    debug_mask_label_to_key,
)
from mask_bundle_export import export_current_mask_bundle
from precision_utils import (
    supports_fp8,
    supports_bf16,
    POLICY_FP8,
)
from pipeline import (
    PIPELINE_PROFILE_VALUES,
    INSTANCE_CAPS,
    THREADS_PER_INSTANCE,
    calculate_max_instances_from_vram,
    estimate_vram_breakdown_mb,
    get_default_instance_vram_profile_mb,
    get_cuda_memory_stats_mb,
    get_system_memory_stats_mb,
    get_gpu_vram_mb,
    vram_warning_text,
)

def transfer_luminance_from_source(source_rgb, colorized_rgb, config=None):
    return _backend_transfer_luminance_from_source(source_rgb, colorized_rgb, config)

def compute_transfer_debug_masks(source_rgb):
    return _backend_compute_transfer_debug_masks(source_rgb)

def apply_debug_mask_visual(base_rgb, mask, mode='overlay', color=(255, 90, 50), opacity=0.55):
    return _backend_apply_debug_mask_visual(base_rgb, mask, mode=mode, color=color, opacity=opacity)

def sanitize_input_width_limit(value, fallback=DEFAULT_INPUT_WIDTH):
    try:
        requested = int(value)
    except (TypeError, ValueError):
        requested = int(fallback)
    if requested <= 0:
        requested = int(fallback)
    return max(MIN_INPUT_WIDTH, min(MAX_INPUT_WIDTH, requested))

def clamp_image_to_input_width(image, input_limit):
    limit = sanitize_input_width_limit(input_limit)
    if image.shape[1] <= limit:
        return image, False, image.shape[1], limit

    original_width = image.shape[1]
    scale = limit / original_width
    resized_height = max(1, int(round(image.shape[0] * scale)))
    resized_image = np.array(
        PIL.Image.fromarray(image).resize(
            (limit, resized_height),
            PIL.Image.Resampling.LANCZOS,
        )
    )
    return resized_image, True, original_width, limit

# Import the actual GUI application main window from our new module
from ui import ColorizerAppMainWindow
ColorizerApp = ColorizerAppMainWindow

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Standalone Manga Colorizer')
    default_device = 'cuda'
    if sys.platform == 'darwin':
        default_device = 'mps'
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default=default_device, help='Device to use for processing')

    config = parser.parse_args()

    # --- MSVC toolchain bootstrap (enables torch.compile + CUDA-Sage) ---
    # Makes cl.exe + INCLUDE/LIB usable in-process so we don't need to launch
    # from an "x64 Native Tools" prompt. Fixes "Failed to find C compiler" and
    # the storm of empty cmd windows Inductor spawns while probing.
    try:
        from msvc_env import ensure_msvc
        _msvc_ok, _msvc_msg = ensure_msvc(verbose=True)
        config.enable_torch_compile = bool(_msvc_ok)
        if not _msvc_ok:
            print("[!] torch.compile disabled (no C++ compiler). " + str(_msvc_msg))
    except Exception as _e:
        config.enable_torch_compile = False
        print(f"[!] MSVC bootstrap skipped: {_e}")

    # --- Global GPU perf flags (set once, process-wide) ---
    # cudnn autotuner picks fastest conv algos for our fixed tile sizes.
    # NOTE: per-pass precision (set at pass start) owns TF32/matmul precision;
    # we only enable the autotuner here.
    if torch.cuda.is_available():
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass
    config.upscaler_tile_size = 256
    config.colorizer_tile_size = 0
    config.tile_pad = 8
    config.lp_generator_patch_size = 16
    config.lp_num_patches_h = 3
    config.lp_num_patches_w = 3
    config.colorized_image_size = 576
    config.upscale_factor = 4
    config.input_image_size = sanitize_input_width_limit(DEFAULT_INPUT_WIDTH)
    config.force_safe_colorizer_width = True
    config.detailed_debug_logs = False
    config.export_ocr_debug = False
    config.chroma_resize_mode = 'BICUBIC'
    config.edge_chroma_protection = True
    config.edge_chroma_strength = 60
    config.line_ink_protection = True
    config.line_ink_protection_strength = 40
    config.screentone_chroma_smoothing = True
    config.screentone_smoothing_strength = 40
    config.pipeline_fail_fast = True
    config.pipeline_worker_profile = 'Balanced'
    config.pipeline_manual_override = False
    config.pipeline_queue_ram_budget_mb = 0
    config.pipeline_writer_threads = 1
    config._denoiser_weights_dir = os.path.abspath(os.path.join(backend_path, 'denoising', 'models'))
    config.denoiser_weights_dir = config._denoiser_weights_dir
    config.pipeline_denoise_instances = PIPELINE_PROFILE_VALUES['Balanced']['denoise_inst']
    config.pipeline_colorize_instances = PIPELINE_PROFILE_VALUES['Balanced']['colorize_inst']
    config.pipeline_upscale_instances = PIPELINE_PROFILE_VALUES['Balanced']['upscale_inst']
    config.pipeline_instances_denoise = config.pipeline_denoise_instances
    config.pipeline_instances_colorize = config.pipeline_colorize_instances
    config.pipeline_instances_upscale = config.pipeline_upscale_instances
    config.pipeline_workers_denoise = config.pipeline_denoise_instances
    config.pipeline_workers_colorize = config.pipeline_colorize_instances
    config.pipeline_workers_upscale = config.pipeline_upscale_instances

    app = ColorizerApp(config)
    app.mainloop()
