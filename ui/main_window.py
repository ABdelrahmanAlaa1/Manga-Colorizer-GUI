import os
import sys
import argparse
import copy
import time
import tkinter as tk
from tkinter import filedialog, ttk, simpledialog, messagebox
from tkinter.font import Font
import threading
import queue
import json
from collections import deque, OrderedDict
import webbrowser
import random
import glob
import PIL.Image
import PIL.ImageTk
import PIL.ImageDraw
from PIL import ImageTk
import numpy as np
import torch

MAX_INPUT_WIDTH = 1792
MIN_INPUT_WIDTH = 256
DEFAULT_INPUT_WIDTH = 800
PREVIEW_SOURCE_CACHE_SIZE = 12
PREVIEW_DENOISE_CACHE_SIZE = 24
PREVIEW_RERUN_DEBOUNCE_MS = 280
MAX_LOG_BUFFER_SIZE = 50000
APP_CLOSE_WAIT_TIMEOUT_S = 12.0
PIPELINE_CPU_MPS_INSTANCE_CAPS = {'denoise': 4, 'colorize': 2, 'upscale': 1}

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

# --- Path Setup (Adjusted for root directory) ---
# Get the directory of the current script (which is now the project root)
# and join it with 'Backend' to get the correct path.
backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Backend'))
sys.path.insert(0, backend_path)

# --- Now we can import from Backend ---
from colorizator import MangaColorizator
from denoisator import MangaDenoiser
from upscalator import MangaUpscaler
from utils.utils import distance_from_grayscale, save_image, clear_torch_cache
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
    format_cuda_device_summary,
    get_cuda_device_summary,
    normalize_fallback_mode,
    normalize_precision_policy,
    precision_vram_scale,
    resolve_precision,
    supports_fp8,
    supports_bf16,
    POLICY_FP8,
)
from pipeline import (
    ProcessingPipeline,
    ModelInstancePool,
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


PIPELINE_PROFILE_CUSTOM = 'Custom'


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


class HoverTooltip:
    def __init__(self, widget, text, delay_ms=450):
        self.widget = widget
        self.text = text
        self.delay_ms = delay_ms
        self._after_id = None
        self._tip_window = None

        self.widget.bind("<Enter>", self._on_enter, add="+")
        self.widget.bind("<Leave>", self._on_leave, add="+")
        self.widget.bind("<ButtonPress>", self._on_leave, add="+")

    def _on_enter(self, _event=None):
        self._cancel_schedule()
        self._after_id = self.widget.after(self.delay_ms, self._show)

    def _on_leave(self, _event=None):
        self._cancel_schedule()
        self._hide()

    def _cancel_schedule(self):
        if self._after_id is not None:
            try:
                self.widget.after_cancel(self._after_id)
            except tk.TclError:
                pass
            self._after_id = None

    def _show(self):
        if self._tip_window is not None or not self.text:
            return

        try:
            x = self.widget.winfo_rootx() + 12
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 10
        except tk.TclError:
            return

        self._tip_window = tk.Toplevel(self.widget)
        self._tip_window.wm_overrideredirect(True)
        self._tip_window.wm_geometry(f"+{x}+{y}")

        label = tk.Label(
            self._tip_window,
            text=self.text,
            justify="left",
            relief="solid",
            borderwidth=1,
            background="#fffbe6",
            padx=6,
            pady=4,
            wraplength=360,
        )
        label.pack()

    def _hide(self):
        if self._tip_window is not None:
            try:
                self._tip_window.destroy()
            except tk.TclError:
                pass
            self._tip_window = None


# --- Core Processing Logic ---
def process_image(image_path, output_folder, colorizer, upscaler, denoiser, config, progress_queue, overwrite_existing, relative_subdir="", detailed_logs=False):
    """
    Processes a single image: denoises, colorizes, and upscales based on the config.
    Sends progress updates back to the GUI via a queue.
    Returns the time taken to process the image.
    """
    try:
        image_name = os.path.basename(image_path)
        output_dir = os.path.join(output_folder, relative_subdir) if relative_subdir else output_folder
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, image_name)

        if not overwrite_existing and os.path.exists(output_path):
            if detailed_logs:
                progress_queue.put({'type': 'log', 'message': f"[~] Skipping '{image_name}' as it already exists."})
            return 0 # Return 0 time taken for skipped files

        if detailed_logs:
            progress_queue.put({'type': 'log', 'message': f"\n--- Processing: {image_name} ---"})

        image = PIL.Image.open(image_path).convert("RGB")
        image = np.array(image)

        coloredness = distance_from_grayscale(image)
        if coloredness > 15 and config.colorize:
            if detailed_logs:
                progress_queue.put({'type': 'log', 'message': f"[!] '{image_name}' appears to be already colored. Skipping colorization step."})
            if not config.denoise and not config.upscale:
                return 0

        start_time = time.time()
        config.current_image_path = image_path

        raw_input_limit = getattr(config, 'input_image_size', DEFAULT_INPUT_WIDTH)
        input_limit = sanitize_input_width_limit(raw_input_limit)
        if detailed_logs:
            try:
                raw_input_limit_int = int(raw_input_limit)
            except (TypeError, ValueError):
                raw_input_limit_int = None

            if raw_input_limit_int != input_limit:
                progress_queue.put(
                    {
                        'type': 'log',
                        'message': f"[*] Input clamp sanitized to {input_limit}px (requested {raw_input_limit}).",
                    }
                )

        image, was_resized, pre_limit_width, _ = clamp_image_to_input_width(image, input_limit)
        if was_resized:
            if detailed_logs:
                progress_queue.put({'type': 'log', 'message': f"[*] Limiting input width: {pre_limit_width}px -> {input_limit}px"})

        if config.denoise and denoiser:
            if detailed_logs:
                progress_queue.put({'type': 'log', 'message': "[*] Denoising..."})
            image = denoiser.denoise(image, config.denoise_sigma, image_name=image_name)

        if config.colorize and colorizer:
            if detailed_logs:
                progress_queue.put({'type': 'log', 'message': "[*] Colorizing..."})
            luminance_source = image.copy()

            # This logic ensures the image width sent to the colorizer is a multiple of 32.
            original_width = image.shape[1]
            target_width = config.colorized_image_size
            if getattr(config, 'force_safe_colorizer_width', False):
                target_width = 576

            # Use the smaller of the original width or the user-defined target width.
            effective_width = min(original_width, target_width)

            # Calculate a new width that is a multiple of 32 by rounding down.
            adjusted_width = effective_width - (effective_width % 32)

            # Handle cases where the image is extremely narrow (less than 32px wide).
            if adjusted_width == 0:
                adjusted_width = 32
                if detailed_logs:
                    progress_queue.put({'type': 'log', 'message': f"[!] Warning: Image width is very small. Setting colorizer width to a minimum of {adjusted_width}px."})

            if adjusted_width != original_width:
                if detailed_logs:
                    progress_queue.put({'type': 'log', 'message': f"[*] Adjusting image width for colorizer: {original_width}px -> {adjusted_width}px"})

            # Set the image for the colorizer with the new, safe width.
            colorizer.set_image(image, adjusted_width, image_name=image_name)

            colorized_output = colorizer.colorize()
            image = transfer_luminance_from_source(luminance_source, colorized_output, config)

        if config.upscale and upscaler:
            if detailed_logs:
                progress_queue.put({'type': 'log', 'message': f"[*] Upscaling by {config.upscale_factor}x..."})
            image = upscaler.upscale(image, config.upscale_factor, image_name=image_name)

        save_image(image, output_path)

        end_time = time.time()
        duration = end_time - start_time
        if detailed_logs:
            progress_queue.put({'type': 'log', 'message': f"[+] Finished '{image_name}' in {duration:.2f}s."})
        return duration

    except Exception as e:
        progress_queue.put({'type': 'log', 'message': f"[!!!] FAILED to process {image_path}. Error: {e}"})
        import traceback
        print(f"--- ERROR TRACEBACK for {image_path} ---")
        traceback.print_exc()
        print("------------------------------------")
        return 0

# --- GUI Application ---
class ColorizerAppMainWindow(tk.Tk):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.settings_file = "settings.json"

        if not hasattr(self.config, 'force_safe_colorizer_width'):
            self.config.force_safe_colorizer_width = True
        if not hasattr(self.config, 'input_image_size'):
            self.config.input_image_size = DEFAULT_INPUT_WIDTH
        if not hasattr(self.config, 'detailed_debug_logs'):
            self.config.detailed_debug_logs = False
        if not hasattr(self.config, 'export_ocr_debug'):
            self.config.export_ocr_debug = False
        if not hasattr(self.config, 'max_ocr_dimension'):
            self.config.max_ocr_dimension = 3072
        if not hasattr(self.config, 'sfx_feather_radius'):
            self.config.sfx_feather_radius = 0
        if not hasattr(self.config, 'chroma_resize_mode'):
            self.config.chroma_resize_mode = 'BICUBIC'
        if not hasattr(self.config, 'edge_chroma_protection'):
            self.config.edge_chroma_protection = True
        if not hasattr(self.config, 'edge_chroma_strength'):
            self.config.edge_chroma_strength = 60
        legacy_ink_protection = getattr(self.config, 'ink_protection', True)
        legacy_ink_protection_strength = getattr(self.config, 'ink_protection_strength', 40)
        if not hasattr(self.config, 'line_ink_protection'):
            self.config.line_ink_protection = legacy_ink_protection
        if not hasattr(self.config, 'line_ink_protection_strength'):
            self.config.line_ink_protection_strength = legacy_ink_protection_strength
        if not hasattr(self.config, 'screentone_chroma_smoothing'):
            self.config.screentone_chroma_smoothing = True
        if not hasattr(self.config, 'screentone_smoothing_strength'):
            self.config.screentone_smoothing_strength = 40
        if not hasattr(self.config, 'pipeline_fail_fast'):
            self.config.pipeline_fail_fast = True
        if not hasattr(self.config, 'pipeline_worker_profile'):
            self.config.pipeline_worker_profile = 'Balanced'
        if not hasattr(self.config, 'pipeline_manual_override'):
            self.config.pipeline_manual_override = False
        if not hasattr(self.config, 'pipeline_queue_ram_budget_mb'):
            self.config.pipeline_queue_ram_budget_mb = 0
        if not hasattr(self.config, 'pipeline_writer_threads'):
            self.config.pipeline_writer_threads = 1
        if not hasattr(self.config, 'precision_policy'):
            self.config.precision_policy = 'auto'
        if not hasattr(self.config, 'colorize_precision_policy'):
            self.config.colorize_precision_policy = self.config.precision_policy
        if not hasattr(self.config, 'upscale_precision_policy'):
            self.config.upscale_precision_policy = self.config.precision_policy
        if not hasattr(self.config, 'precision_cast_weights'):
            self.config.precision_cast_weights = True
        if not hasattr(self.config, 'precision_fallback'):
            self.config.precision_fallback = 'per_image'
        if not hasattr(self.config, 'precision_allow_tf32'):
            self.config.precision_allow_tf32 = True
        if not hasattr(self.config, 'preview_fast_mode'):
            self.config.preview_fast_mode = False
        if not hasattr(self.config, 'external_color_source_enabled'):
            self.config.external_color_source_enabled = False
        if not hasattr(self.config, 'external_color_source_dir'):
            self.config.external_color_source_dir = ''

        legacy_workers_denoise = getattr(
            self.config,
            'pipeline_workers_denoise',
            PIPELINE_PROFILE_VALUES['Balanced']['denoise_inst'],
        )
        legacy_workers_colorize = getattr(
            self.config,
            'pipeline_workers_colorize',
            PIPELINE_PROFILE_VALUES['Balanced']['colorize_inst'],
        )
        legacy_workers_upscale = getattr(
            self.config,
            'pipeline_workers_upscale',
            PIPELINE_PROFILE_VALUES['Balanced']['upscale_inst'],
        )

        legacy_instances_denoise = getattr(self.config, 'pipeline_instances_denoise', legacy_workers_denoise)
        legacy_instances_colorize = getattr(self.config, 'pipeline_instances_colorize', legacy_workers_colorize)
        legacy_instances_upscale = getattr(self.config, 'pipeline_instances_upscale', legacy_workers_upscale)

        if not hasattr(self.config, 'pipeline_denoise_instances'):
            self.config.pipeline_denoise_instances = legacy_instances_denoise
        if not hasattr(self.config, 'pipeline_colorize_instances'):
            self.config.pipeline_colorize_instances = legacy_instances_colorize
        if not hasattr(self.config, 'pipeline_upscale_instances'):
            self.config.pipeline_upscale_instances = legacy_instances_upscale

        if not hasattr(self.config, '_denoiser_weights_dir'):
            fallback_weights_dir = getattr(
                self.config,
                'denoiser_weights_dir',
                os.path.join(backend_path, 'denoising', 'models'),
            )
            self.config._denoiser_weights_dir = os.path.abspath(fallback_weights_dir)

        # Keep legacy aliases in sync for older code paths/settings consumers.
        self.config.pipeline_instances_denoise = self.config.pipeline_denoise_instances
        self.config.pipeline_instances_colorize = self.config.pipeline_colorize_instances
        self.config.pipeline_instances_upscale = self.config.pipeline_upscale_instances
        self.config.pipeline_workers_denoise = self.config.pipeline_denoise_instances
        self.config.pipeline_workers_colorize = self.config.pipeline_colorize_instances
        self.config.pipeline_workers_upscale = self.config.pipeline_upscale_instances
        self.config.pipeline_writer_threads = self._coerce_int(
            getattr(self.config, 'pipeline_writer_threads', 1),
            1,
            1,
            8,
        )
        self.config.denoiser_weights_dir = self.config._denoiser_weights_dir
        self.config.input_image_size = sanitize_input_width_limit(self.config.input_image_size)

        self.title("Manga Colorizer")
        self.geometry("550x700")

        # --- Variables ---
        self.input_folder = tk.StringVar()
        self.output_folder = tk.StringVar()
        self.enable_colorize = tk.BooleanVar(value=True)
        self.enable_upscale = tk.BooleanVar(value=True)
        self.enable_denoise = tk.BooleanVar(value=True)
        self.overwrite_existing = tk.BooleanVar(value=False)
        self.external_color_source_enabled = tk.BooleanVar(value=False)
        self.external_color_source_dir = tk.StringVar()
        self.eta_text = tk.StringVar(value="ETA: N/A")

        # --- Advanced settings variables ---
        default_networks_path = os.path.join(backend_path, 'networks')
        self.colorizer_path = tk.StringVar(value=os.path.join(default_networks_path, 'generator.zip'))
        self.upscaler_path = tk.StringVar(value=os.path.join(default_networks_path, 'RealESRGAN_x4plus_anime_6B.pt'))
        self.upscaler_type = tk.StringVar(value='Auto-Detect')
        self.denoise_sigma = tk.IntVar(value=25)
        self.upscaler_tile_size = tk.IntVar(value=256)
        self.colorizer_tile_size = tk.IntVar(value=0)
        self.tile_pad = tk.IntVar(value=8)
        # LP upscaler settings
        self.lp_generator_patch_size = tk.IntVar(value=16)
        self.lp_num_patches_h = tk.IntVar(value=3)
        self.lp_num_patches_w = tk.IntVar(value=3)
        self.colorized_image_size = tk.IntVar(value=576)
        self.input_image_size = tk.IntVar(value=sanitize_input_width_limit(getattr(self.config, 'input_image_size', DEFAULT_INPUT_WIDTH)))
        self.force_safe_colorizer_width = tk.BooleanVar(value=self.config.force_safe_colorizer_width)
        self.detailed_debug_logs = tk.BooleanVar(value=self.config.detailed_debug_logs)
        self.export_ocr_debug = tk.BooleanVar(value=getattr(self.config, 'export_ocr_debug', False))
        self.max_ocr_dimension = tk.IntVar(value=getattr(self.config, 'max_ocr_dimension', 3072))
        # Normalize old boolean values to string mode
        yolo_raw = getattr(self.config, 'use_yolo_bubbles', 'full')
        if yolo_raw is True:
            yolo_raw = 'full'
        elif yolo_raw is False:
            yolo_raw = 'off'
        self.use_yolo_bubbles = tk.StringVar(value=yolo_raw)
        self.fp_strictness = tk.DoubleVar(value=getattr(self.config, 'fp_strictness', 0.5))
        self.sfx_feather_radius = tk.IntVar(value=getattr(self.config, 'sfx_feather_radius', 0))
        self.yolo_model_type = tk.StringVar(value=getattr(self.config, 'yolo_model_type', 'seg'))
        self.chroma_resize_mode = tk.StringVar(value=self.config.chroma_resize_mode)
        self.edge_chroma_protection = tk.BooleanVar(value=self.config.edge_chroma_protection)
        self.edge_chroma_strength = tk.IntVar(value=self.config.edge_chroma_strength)
        self.line_ink_protection = tk.BooleanVar(value=self.config.line_ink_protection)
        self.line_ink_protection_strength = tk.IntVar(value=self.config.line_ink_protection_strength)
        self.screentone_chroma_smoothing = tk.BooleanVar(value=self.config.screentone_chroma_smoothing)
        self.screentone_smoothing_strength = tk.IntVar(value=self.config.screentone_smoothing_strength)
        self.pipeline_fail_fast = tk.BooleanVar(value=self.config.pipeline_fail_fast)
        self.pipeline_worker_profile = tk.StringVar(value=self.config.pipeline_worker_profile)
        self.pipeline_denoise_instances = tk.IntVar(value=self.config.pipeline_denoise_instances)
        self.pipeline_colorize_instances = tk.IntVar(value=self.config.pipeline_colorize_instances)
        self.pipeline_upscale_instances = tk.IntVar(value=self.config.pipeline_upscale_instances)
        self.pipeline_writer_threads = tk.IntVar(value=self._sanitize_writer_threads(self.config.pipeline_writer_threads))
        self.colorize_precision_policy = tk.StringVar(value=normalize_precision_policy(self.config.colorize_precision_policy))
        self.upscale_precision_policy = tk.StringVar(value=normalize_precision_policy(self.config.upscale_precision_policy))
        self.precision_cast_weights = tk.BooleanVar(value=bool(self.config.precision_cast_weights))
        self.precision_fallback = tk.StringVar(value=normalize_fallback_mode(self.config.precision_fallback))
        self.precision_allow_tf32 = tk.BooleanVar(value=bool(self.config.precision_allow_tf32))

        # Mixed precision strategy unified selection variable
        init_precision = normalize_precision_policy(getattr(self.config, 'precision_policy', 'auto'))
        if init_precision == 'auto':
            init_precision = 'fp16'
        gui_precision = 'FP16 (Recommended / Fast)'
        if init_precision == 'fp32':
            gui_precision = 'FP32 (Highest Quality / Slowest)'
        elif init_precision == 'fp8':
            gui_precision = 'FP8 (Experimental / High VRAM Savings)'
        self.precision_policy = tk.StringVar(value=gui_precision)

        self.torch_compile = tk.BooleanVar(value=bool(getattr(self.config, 'enable_torch_compile', False)))
        self.enable_flash_attention = tk.BooleanVar(value=bool(getattr(self.config, 'enable_flash_attention', False)))
        self.enable_cpu_offload = tk.BooleanVar(value=bool(getattr(self.config, 'enable_cpu_offload', False)))
        self.enable_sage_attention = tk.BooleanVar(value=bool(getattr(self.config, 'enable_sage_attention', False)))
        self.ocr_model_tier = tk.StringVar(value=getattr(self.config, 'ocr_model_tier', 'server'))
        self.denoise_max_side = tk.IntVar(value=getattr(self.config, 'denoise_max_side', 0))

        self.precision_hw_summary = tk.StringVar(value="")
        self.preview_fast_mode = tk.BooleanVar(value=bool(getattr(self.config, 'preview_fast_mode', False)))
        self.log_show_error = tk.BooleanVar(value=True)
        self.log_show_warning = tk.BooleanVar(value=True)
        self.log_show_debug = tk.BooleanVar(value=True)
        self.log_show_trace = tk.BooleanVar(value=False)
        self.log_show_verbose = tk.BooleanVar(value=False)
        self.vram_meter_text = tk.StringVar(value="VRAM: waiting for CUDA telemetry")


        self.processing_thread = None
        self.terminate_event = threading.Event() # Event to signal thread termination on window close
        self.pause_event = threading.Event() # Using a separate event for pausing
        self._close_deadline_mono = None
        self._dynamic_log_line = None
        self._full_log_buffer = []
        self._pipeline_pass_state = 'idle'
        self._preview_pass2_active = False

        # --- Model Management ---
        self.colorizer_model = None
        self.upscaler_model = None
        self.denoiser_model = None

        # --- State Tracking ---
        self.current_input_path = None
        self.preview_session = None
        self.advanced_window_ref = None
        self._model_signatures = {
            'colorizer': None,
            'upscaler': None,
            'denoiser': None,
        }

        self.quality_presets = {
            'Default': dict(QUALITY_PRESET_DEFAULTS),
        }
        self.active_quality_preset_global = 'Default'
        self.quality_preset_overrides = {}
        self._advanced_preset_selector_refresher = None
        self._advanced_pipeline_runtime_refresher = None
        self.pipeline_vram_calibration = {}
        self._last_cuda_stats = {}
        self._last_vram_poll_ts = 0.0
        self._folder_sample_size_cache = {
            'key': None,
            'size': None,
        }

        self._stage_toggle_trace_handles = []
        for stage_var in (self.enable_colorize, self.enable_upscale, self.enable_denoise):
            trace_id = stage_var.trace_add('write', self._on_stage_toggle_changed)
            self._stage_toggle_trace_handles.append((stage_var, trace_id))


        # --- Style for labels ---
        self.style = ttk.Style(self)
        italic_font = Font(family="Helvetica", size=10, slant="italic")
        self.style.configure("Italic.TLabel", font=italic_font)

        # --- Style for hyperlink ---
        link_font = Font(family="Helvetica", size=11, underline=True)
        self.style.configure("Link.TLabel", foreground="blue", font=link_font)


        self.create_widgets()
        self.load_settings()
        self._apply_pipeline_profile_if_needed(force=False)
        self._refresh_start_button_state()

        self.progress_queue = queue.Queue()
        self.after(100, self.check_queue)

        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def free_vram(self):
        """Releases all PyTorch models and clears VRAM."""
        import gc
        
        should_clear = False
        if getattr(self, 'colorizer_model', None) is not None:
            self.colorizer_model = None
            self._model_signatures['colorizer'] = None
            should_clear = True
            
        if getattr(self, 'denoiser_model', None) is not None:
            self.denoiser_model = None
            self._model_signatures['denoiser'] = None
            should_clear = True
            
        # We also release upscaler_model just in case
        if getattr(self, 'upscaler_model', None) is not None:
            self.upscaler_model = None
            self._model_signatures['upscaler'] = None
            should_clear = True

        from transfer_quality import free_ocr_reader
        if free_ocr_reader():
            should_clear = True

        if should_clear:
            gc.collect()
            clear_torch_cache()
            if self.detailed_debug_logs.get():
                self.progress_queue.put({'type': 'log', 'message': "[*] VRAM successfully released."})

    def create_widgets(self):
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        folder_frame = ttk.LabelFrame(main_frame, text="Folders", padding="10")
        folder_frame.pack(fill=tk.X, pady=5)
        folder_frame.columnconfigure(1, weight=1)
        ttk.Label(folder_frame, text="Input Folder:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(folder_frame, textvariable=self.input_folder).grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        ttk.Button(folder_frame, text="Browse...", command=self.select_input_folder).grid(row=0, column=2, padx=5, pady=5)
        ttk.Label(folder_frame, text="Output Folder:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
        ttk.Entry(folder_frame, textvariable=self.output_folder).grid(row=1, column=1, padx=5, pady=5, sticky="ew")
        ttk.Button(folder_frame, text="Browse...", command=self.select_output_folder).grid(row=1, column=2, padx=5, pady=5)

        # --- External Color Source ---
        self._ext_color_cb = ttk.Checkbutton(
            folder_frame,
            text="External Color Source",
            variable=self.external_color_source_enabled,
            command=self._on_external_color_toggled,
        )
        self._ext_color_cb.grid(row=2, column=0, padx=5, pady=5, sticky="w")
        self._ext_color_entry = ttk.Entry(folder_frame, textvariable=self.external_color_source_dir)
        self._ext_color_entry.grid(row=2, column=1, padx=5, pady=5, sticky="ew")
        self._ext_color_browse = ttk.Button(
            folder_frame, text="Browse...", command=self.select_external_color_dir,
        )
        self._ext_color_browse.grid(row=2, column=2, padx=5, pady=5)
        HoverTooltip(
            self._ext_color_cb,
            "Use pre-colorized images (e.g. from ComfyUI) as the color source.\n"
            "Colors (chroma) are transferred onto the grayscale output,\n"
            "preserving text and structural detail. If upscaling is disabled,\n"
            "colors are mapped directly to the original input images.\n\n"
            "When enabled, Denoise and Colorize are skipped (Pass 1 overruled).\n"
            "Files are fuzzy-matched by numeric sequences in filenames.",
        )
        self._sync_external_color_state()

        options_frame = ttk.LabelFrame(main_frame, text="Options", padding="10")
        options_frame.pack(fill=tk.X, pady=5)
        options_frame.columnconfigure(0, weight=1)
        options_frame.columnconfigure(1, weight=1)

        self._colorize_cb = ttk.Checkbutton(options_frame, text="Enable Colorizer", variable=self.enable_colorize)
        self._colorize_cb.grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(options_frame, text="Enable Upscaling", variable=self.enable_upscale).grid(row=0, column=1, sticky="w")
        self._denoise_cb = ttk.Checkbutton(options_frame, text="Enable Denoising", variable=self.enable_denoise)
        self._denoise_cb.grid(row=1, column=0, sticky="w")
        ttk.Checkbutton(options_frame, text="Overwrite existing files", variable=self.overwrite_existing).grid(row=1, column=1, sticky="w")

        advanced_button = ttk.Button(options_frame, text="Advanced Settings...", command=self.open_advanced_settings)
        advanced_button.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(10,0))

        progress_frame = ttk.LabelFrame(main_frame, text="Progress", padding="10")
        progress_frame.pack(fill=tk.X, pady=5)
        progress_frame.columnconfigure(0, weight=1)
        self.progress_bar = ttk.Progressbar(progress_frame, orient="horizontal", mode="determinate")
        self.progress_bar.grid(row=0, column=0, sticky="ew", padx=(0,10))
        ttk.Label(progress_frame, textvariable=self.eta_text).grid(row=0, column=1, sticky="e")
        self.vram_bar = ttk.Progressbar(progress_frame, orient="horizontal", mode="determinate", maximum=100)
        self.vram_bar.grid(row=1, column=0, sticky="ew", padx=(0, 10), pady=(6, 0))
        ttk.Label(progress_frame, textvariable=self.vram_meter_text, style="Italic.TLabel").grid(row=1, column=1, sticky="e", pady=(6, 0))

        action_frame = ttk.Frame(main_frame)
        action_frame.pack(fill=tk.X, pady=10)
        action_frame.columnconfigure(0, weight=1)
        action_frame.columnconfigure(1, weight=1)
        self.start_button = ttk.Button(action_frame, text="Start Processing", command=self.start_processing)
        self.start_button.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        self.pause_button = ttk.Button(action_frame, text="Pause", command=self.pause_processing, state="disabled")
        self.pause_button.grid(row=0, column=1, sticky="ew", padx=(5, 0))

        log_frame = ttk.LabelFrame(main_frame, text="Log", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)

        self.log_text = tk.Text(log_frame, height=10, state="disabled", wrap="word")
        self.log_text.grid(row=0, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log_text.config(yscrollcommand=scrollbar.set)

        log_filter_frame = ttk.Frame(log_frame)
        log_filter_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(2, 0))

        for label_text, var in [
            ("Error", self.log_show_error),
            ("Warn", self.log_show_warning),
            ("Debug", self.log_show_debug),
            ("Trace", self.log_show_trace),
            ("Verbose", self.log_show_verbose),
        ]:
            ttk.Checkbutton(
                log_filter_frame,
                text=label_text,
                variable=var,
                command=self._on_log_filter_changed,
            ).pack(side=tk.LEFT, padx=3)

        # --- Bottom bar for links and buttons ---
        bottom_bar_frame = ttk.Frame(log_frame)
        bottom_bar_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(5,0))
        bottom_bar_frame.columnconfigure(1, weight=1) # Make column 1 (where button is) expand to push button right

        # Bug report link
        bug_report_label = ttk.Label(bottom_bar_frame, text="Found a bug? Report @sepTN", style="Link.TLabel", cursor="hand2")
        bug_report_label.grid(row=0, column=0, sticky="w", padx=5) # Align to the west (left)
        bug_report_label.bind("<Button-1>", self.open_bug_report_link)

        # --- Button container for the right side ---
        button_container = ttk.Frame(bottom_bar_frame)
        button_container.grid(row=0, column=1, sticky="e")

        # Preview button
        preview_button = ttk.Button(button_container, text="Preview", command=self.open_preview_window)
        preview_button.pack(side=tk.LEFT, padx=(0, 5))

        # Clear log button
        clear_button = ttk.Button(button_container, text="Clear Log", command=self.clear_log)
        clear_button.pack(side=tk.LEFT)

        self._refresh_start_button_state()

    def _has_enabled_processing_stage(self):
        if self.external_color_source_enabled.get():
            return True
        return bool(self.enable_colorize.get() or self.enable_upscale.get() or self.enable_denoise.get())

    def _refresh_start_button_state(self):
        if not hasattr(self, 'start_button'):
            return

        has_stage = self._has_enabled_processing_stage()
        is_running = bool(self.processing_thread and self.processing_thread.is_alive())
        can_start = has_stage and not is_running
        self.start_button.config(state="normal" if can_start else "disabled")

    def _on_stage_toggle_changed(self, *_args):
        self._refresh_start_button_state()

    def _is_batch_processing_active(self):
        return bool(self.processing_thread and self.processing_thread.is_alive())

    def _pass_label(self, pass_key, compact=False):
        key = str(pass_key or '').strip().lower()

        if key == 'pass1':
            if compact:
                prefix = 'Pass1'
                joiner = '+'
                denoise_name = 'D'
                colorize_name = 'C'
                inactive_label = 'inactive'
            else:
                prefix = 'Pass 1'
                joiner = ' + '
                denoise_name = 'Denoise'
                colorize_name = 'Colorize'
                inactive_label = 'inactive'

            stage_names = []
            if self.enable_denoise.get():
                stage_names.append(denoise_name)
            if self.enable_colorize.get():
                stage_names.append(colorize_name)

            if stage_names:
                joined = joiner.join(stage_names)
                return f"{prefix}({joined})" if compact else f"{prefix} ({joined})"

            return f"{prefix}({inactive_label})" if compact else f"{prefix} ({inactive_label})"

        if key == 'pass2':
            if self.enable_upscale.get():
                return 'Pass2(U)' if compact else 'Pass 2 (Upscale)'
            return 'Pass2(inactive)' if compact else 'Pass 2 (inactive)'

        return 'batch' if compact else 'batch processing'

    def _preview_running_warning_text(self):
        phase = str(getattr(self, '_pipeline_pass_state', '') or '').strip().lower()
        if phase == 'pass1':
            phase_text = self._pass_label('pass1')
        elif phase == 'pass2':
            phase_text = self._pass_label('pass2')
        else:
            phase_text = 'batch processing'
        return f"Live tuning is disabled during {phase_text}. Showing input vs current output only."

    def _refresh_preview_runtime_mode(self):
        session = self.preview_session
        if not isinstance(session, dict):
            return

        updater = session.get('update_runtime_mode')
        if not callable(updater):
            return

        try:
            updater()
        except tk.TclError:
            self.preview_session = None


    def _normalize_folder_key(self, folder_path):
        if not folder_path:
            return ""
        try:
            return os.path.normcase(os.path.abspath(os.path.normpath(folder_path)))
        except Exception:
            return folder_path

    @staticmethod
    def _coerce_bool(value, fallback):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"1", "true", "yes", "on"}:
                return True
            if lowered in {"0", "false", "no", "off"}:
                return False
        if isinstance(value, (int, float)):
            return bool(value)
        return fallback

    @staticmethod
    def _coerce_int(value, fallback, min_value, max_value):
        try:
            coerced = int(value)
        except (TypeError, ValueError):
            return fallback
        return max(min_value, min(max_value, coerced))

    @staticmethod
    def _sanitize_queue_budget_mb(value):
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return 0
        if parsed <= 0:
            return 0
        return max(64, parsed)

    @staticmethod
    def _sanitize_writer_threads(value):
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return 1
        return max(1, min(8, parsed))

    def _pipeline_device_caps(self, device_name=None, instance_counts=None):
        device = str(device_name if device_name is not None else getattr(self.config, 'device', 'cpu')).lower()
        if device == 'cuda':
            gpu_vram_mb = get_gpu_vram_mb()
            if gpu_vram_mb <= 0:
                return dict(INSTANCE_CAPS)

            if not isinstance(instance_counts, dict):
                instance_counts = {
                    'denoise': int(self.pipeline_denoise_instances.get()),
                    'colorize': int(self.pipeline_colorize_instances.get()),
                    'upscale': int(self.pipeline_upscale_instances.get()),
                }

            inst_profile = self._pipeline_instance_vram_profile_mb(device_name=device)
            overhead_mb = self._pipeline_runtime_overhead_mb()
            caps = {}
            pass1_peer = {
                'denoise': 'colorize',
                'colorize': 'denoise',
            }
            for stage in ('denoise', 'colorize'):
                other_stage = pass1_peer[stage]
                other_mb = int(instance_counts.get(other_stage, 0)) * int(inst_profile.get(other_stage, 0))
                caps[stage] = calculate_max_instances_from_vram(
                    gpu_vram_mb,
                    stage,
                    other_stages_mb=other_mb,
                    per_instance_mb=inst_profile.get(stage, 0),
                    overhead_mb=overhead_mb,
                )

            caps['upscale'] = calculate_max_instances_from_vram(
                gpu_vram_mb,
                'upscale',
                other_stages_mb=0,
                per_instance_mb=inst_profile.get('upscale', 0),
                overhead_mb=overhead_mb,
            )
            return caps

        return {
            key: min(INSTANCE_CAPS[key], PIPELINE_CPU_MPS_INSTANCE_CAPS[key])
            for key in ('denoise', 'colorize', 'upscale')
        }

    def _pipeline_profile_values(self, profile_name=None, device_name=None):
        profile = str(profile_name or 'Balanced').strip().title()
        if profile not in PIPELINE_PROFILE_VALUES:
            profile = 'Balanced'

        profile_values = dict(PIPELINE_PROFILE_VALUES[profile])
        values = {
            'denoise': int(profile_values.get('denoise_inst', profile_values.get('denoise', 1))),
            'colorize': int(profile_values.get('colorize_inst', profile_values.get('colorize', 1))),
            'upscale': int(profile_values.get('upscale_inst', profile_values.get('upscale', 1))),
        }
        caps = self._pipeline_device_caps(device_name=device_name, instance_counts=values)
        for key in ('denoise', 'colorize', 'upscale'):
            values[key] = max(1, min(caps[key], int(values[key])))
        return values

    def _clamp_pipeline_instances(self, denoise_instances, colorize_instances, upscale_instances, device_name=None):
        caps = self._pipeline_device_caps(
            device_name=device_name,
            instance_counts={
                'denoise': denoise_instances,
                'colorize': colorize_instances,
                'upscale': upscale_instances,
            },
        )
        return {
            'denoise': max(1, min(caps['denoise'], int(denoise_instances))),
            'colorize': max(1, min(caps['colorize'], int(colorize_instances))),
            'upscale': max(1, min(caps['upscale'], int(upscale_instances))),
        }

    def _normalized_pipeline_profile(self, profile_name):
        profile = str(profile_name or 'Balanced').strip().title()
        if profile == PIPELINE_PROFILE_CUSTOM:
            return PIPELINE_PROFILE_CUSTOM
        if profile not in PIPELINE_PROFILE_VALUES:
            return 'Balanced'
        return profile

    def _effective_pipeline_instances(self, device_name=None):
        profile = self._normalized_pipeline_profile(self.pipeline_worker_profile.get())
        if profile == PIPELINE_PROFILE_CUSTOM:
            return self._clamp_pipeline_instances(
                self.pipeline_denoise_instances.get(),
                self.pipeline_colorize_instances.get(),
                self.pipeline_upscale_instances.get(),
                device_name=device_name,
            )
        return self._pipeline_profile_values(profile, device_name=device_name)

    def _active_pipeline_instances(self, device_name=None):
        configured = self._effective_pipeline_instances(device_name=device_name)
        return {
            'denoise': int(configured['denoise']) if self.enable_denoise.get() else 0,
            'colorize': int(configured['colorize']) if self.enable_colorize.get() else 0,
            'upscale': int(configured['upscale']) if self.enable_upscale.get() else 0,
        }

    def _probe_folder_sample_size(self, folder_path):
        normalized_key = self._normalize_folder_key(folder_path)
        cached = self._folder_sample_size_cache
        if cached.get('key') == normalized_key:
            return cached.get('size')

        sample_size = None
        supported_ext = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
        if folder_path and os.path.isdir(folder_path):
            for root, _, files in os.walk(folder_path):
                for filename in sorted(files):
                    if not filename.lower().endswith(supported_ext):
                        continue
                    sample_path = os.path.join(root, filename)
                    try:
                        with PIL.Image.open(sample_path) as img:
                            sample_size = (int(img.size[0]), int(img.size[1]))
                        break
                    except Exception:
                        continue
                if sample_size is not None:
                    break

        self._folder_sample_size_cache = {
            'key': normalized_key,
            'size': sample_size,
        }
        return sample_size

    def _estimated_pipeline_input_hw(self):
        input_limit = sanitize_input_width_limit(self.input_image_size.get())
        folder_path = self.input_folder.get().strip().strip("'\"")
        sample_size = self._probe_folder_sample_size(folder_path)

        if sample_size is None:
            # Fall back to a portrait-ish page ratio when no sample is available yet.
            fallback_h = max(1, int(round(input_limit * 1.45)))
            return fallback_h, int(input_limit)

        sample_w, sample_h = sample_size
        if sample_w > input_limit and sample_w > 0:
            ratio = input_limit / float(sample_w)
            sample_w = int(input_limit)
            sample_h = max(1, int(round(sample_h * ratio)))

        return max(1, int(sample_h)), max(1, int(sample_w))

    def _cold_pipeline_instance_profile_mb(self):
        input_h, input_w = self._estimated_pipeline_input_hw()
        input_mp = max(0.15, (float(input_h) * float(input_w)) / 1_000_000.0)
        upscale_factor = max(1, int(getattr(self.config, 'upscale_factor', 4)))
        output_mp = input_mp * float(upscale_factor * upscale_factor)

        defaults = get_default_instance_vram_profile_mb()
        profile = dict(defaults)

        profile['denoise'] = max(defaults['denoise'], int(round(120.0 + 220.0 * input_mp)))
        profile['colorize'] = max(defaults['colorize'], int(round(310.0 + 430.0 * input_mp)))

        up_tile_size = max(0, int(self.upscaler_tile_size.get()))
        if up_tile_size > 0:
            upscale_slope = 8.0
            upscale_base = 620.0
        else:
            upscale_slope = 12.0
            upscale_base = 680.0
        profile['upscale'] = max(defaults['upscale'], int(round(upscale_base + upscale_slope * output_mp)))
        return profile

    def _scaled_calibrated_profile_mb(self, observed_profile):
        observed = dict(observed_profile or {})
        cold_profile = self._cold_pipeline_instance_profile_mb()
        defaults = get_default_instance_vram_profile_mb()

        warmup_hw = self.pipeline_vram_calibration.get('warmup_input_hw', []) if isinstance(self.pipeline_vram_calibration, dict) else []
        warmup_upscale_factor = 4
        if isinstance(self.pipeline_vram_calibration, dict):
            try:
                warmup_upscale_factor = max(1, int(self.pipeline_vram_calibration.get('warmup_upscale_factor', 4)))
            except (TypeError, ValueError):
                warmup_upscale_factor = 4

        if not isinstance(warmup_hw, list) or len(warmup_hw) < 2:
            return {
                stage: max(cold_profile[stage], int(observed.get(stage, cold_profile[stage])))
                for stage in ('denoise', 'colorize', 'upscale')
            }

        try:
            warmup_h = max(1, int(warmup_hw[0]))
            warmup_w = max(1, int(warmup_hw[1]))
        except (TypeError, ValueError):
            return {
                stage: max(cold_profile[stage], int(observed.get(stage, cold_profile[stage])))
                for stage in ('denoise', 'colorize', 'upscale')
            }

        current_h, current_w = self._estimated_pipeline_input_hw()
        current_upscale_factor = max(1, int(getattr(self.config, 'upscale_factor', 4)))

        warmup_input_mp = max(0.1, (warmup_h * warmup_w) / 1_000_000.0)
        current_input_mp = max(0.1, (current_h * current_w) / 1_000_000.0)
        input_ratio = current_input_mp / warmup_input_mp

        warmup_output_mp = max(0.1, warmup_input_mp * float(warmup_upscale_factor * warmup_upscale_factor))
        current_output_mp = max(0.1, current_input_mp * float(current_upscale_factor * current_upscale_factor))
        output_ratio = current_output_mp / warmup_output_mp

        scaled = {}
        for stage in ('denoise', 'colorize', 'upscale'):
            observed_value = int(observed.get(stage, 0) or 0)
            if observed_value <= 0:
                scaled[stage] = cold_profile[stage]
                continue

            default_value = defaults[stage]
            extra = max(0.0, float(observed_value - default_value))

            if stage == 'upscale':
                stage_ratio = max(0.45, min(2.8, output_ratio))
            else:
                stage_ratio = max(0.55, min(2.3, input_ratio))

            scaled_value = int(round(default_value + extra * stage_ratio))
            scaled[stage] = max(cold_profile[stage], scaled_value)

        return scaled

    def _precision_scale_map(self, device_name=None):
        device = device_name if device_name is not None else getattr(self.config, 'device', 'cpu')
        precision_config = copy.copy(self.config)
        precision_config.colorize_precision_policy = normalize_precision_policy(self.colorize_precision_policy.get())
        precision_config.upscale_precision_policy = normalize_precision_policy(self.upscale_precision_policy.get())
        precision_config.precision_cast_weights = bool(self.precision_cast_weights.get())
        precision_config.precision_fallback = normalize_fallback_mode(self.precision_fallback.get())
        precision_config.precision_allow_tf32 = bool(self.precision_allow_tf32.get())
        scales = {}
        for stage in ('denoise', 'colorize', 'upscale'):
            spec = resolve_precision(precision_config, device_name=device, stage=stage)
            scales[stage] = float(precision_vram_scale(spec))
        return scales

    def _precision_adjusted_profile_mb(self, profile_mb, device_name=None):
        profile = dict(profile_mb or {})
        scales = self._precision_scale_map(device_name=device_name)
        adjusted = {}
        for stage in ('denoise', 'colorize', 'upscale'):
            value = int(profile.get(stage, 0) or 0)
            scale = float(scales.get(stage, 1.0))
            if value <= 0:
                adjusted[stage] = 0
            else:
                adjusted[stage] = max(1, int(round(value * scale)))
        return adjusted

    def _pipeline_instance_vram_profile_mb(self, device_name=None):
        cold_profile = self._cold_pipeline_instance_profile_mb()
        observed = self.pipeline_vram_calibration.get('instance_vram_profile_mb', {}) if isinstance(self.pipeline_vram_calibration, dict) else {}
        if not isinstance(observed, dict) or not observed:
            return self._precision_adjusted_profile_mb(cold_profile, device_name=device_name)

        scaled = self._scaled_calibrated_profile_mb(observed)
        return self._precision_adjusted_profile_mb(scaled, device_name=device_name)

    def _pipeline_runtime_overhead_mb(self):
        if not isinstance(self.pipeline_vram_calibration, dict):
            return 0
        try:
            return max(0, int(self.pipeline_vram_calibration.get('runtime_overhead_mb', 0)))
        except (TypeError, ValueError):
            return 0

    def _pipeline_profile_source_label(self):
        if not isinstance(self.pipeline_vram_calibration, dict):
            base_label = 'cold-heuristic'
            scales = self._precision_scale_map()
            if any(abs(scale - 1.0) > 1e-6 for scale in scales.values()):
                return f"{base_label}+precision"
            return base_label
        observed = self.pipeline_vram_calibration.get('instance_vram_profile_mb', {})
        if not isinstance(observed, dict):
            base_label = 'cold-heuristic'
            scales = self._precision_scale_map()
            if any(abs(scale - 1.0) > 1e-6 for scale in scales.values()):
                return f"{base_label}+precision"
            return base_label
        if any(int(observed.get(stage, 0) or 0) > 0 for stage in ('denoise', 'colorize', 'upscale')):
            base_label = 'calibrated+scaled'
        else:
            base_label = 'cold-heuristic'

        scales = self._precision_scale_map()
        if any(abs(scale - 1.0) > 1e-6 for scale in scales.values()):
            return f"{base_label}+precision"
        return base_label

    def _effective_pipeline_vram_mb(self, device_name=None):
        breakdown = self._effective_pipeline_vram_breakdown_mb(device_name=device_name)
        return int(breakdown.get('peak_total_mb', 0))

    def _pipeline_shared_stage_peak_mb(self):
        if not isinstance(self.pipeline_vram_calibration, dict):
            return {'denoise': 0, 'colorize': 0, 'upscale': 0}

        shared = self.pipeline_vram_calibration.get('shared_stage_peak_mb')
        if not isinstance(shared, dict):
            shared = self.pipeline_vram_calibration.get('stage_peak_inference_mb', {})

        if not isinstance(shared, dict):
            shared = {}

        return {
            stage: max(0, int(shared.get(stage, 0) or 0))
            for stage in ('denoise', 'colorize', 'upscale')
        }

    def _effective_pipeline_vram_breakdown_mb(self, device_name=None):
        inst = self._active_pipeline_instances(device_name=device_name)
        breakdown = estimate_vram_breakdown_mb(
            inst['denoise'],
            inst['colorize'],
            inst['upscale'],
            instance_profile_mb=self._pipeline_instance_vram_profile_mb(device_name=device_name),
            overhead_mb=self._pipeline_runtime_overhead_mb(),
        )

        shared_stage_peak = self._pipeline_shared_stage_peak_mb()
        pass1_shared_mb = 0
        pass2_shared_mb = 0

        if int(breakdown.get('pass1_total_mb', 0) or 0) > 0:
            pass1_shared_mb = max(
                int(shared_stage_peak.get('denoise', 0) or 0),
                int(shared_stage_peak.get('colorize', 0) or 0),
            )
        if int(breakdown.get('pass2_total_mb', 0) or 0) > 0:
            pass2_shared_mb = int(shared_stage_peak.get('upscale', 0) or 0)

        pass1_total = int(breakdown.get('pass1_total_mb', 0) or 0)
        pass2_total = int(breakdown.get('pass2_total_mb', 0) or 0)

        if pass1_total > 0:
            pass1_total += pass1_shared_mb
        if pass2_total > 0:
            pass2_total += pass2_shared_mb

        if pass1_total <= 0 and pass2_total <= 0:
            peak_pass = 'none'
            peak_total = 0
        elif pass1_total >= pass2_total:
            peak_pass = 'pass1'
            peak_total = pass1_total
        else:
            peak_pass = 'pass2'
            peak_total = pass2_total

        breakdown['pass1_base_mb'] = int(breakdown.get('pass1_total_mb', 0) or 0)
        breakdown['pass2_base_mb'] = int(breakdown.get('pass2_total_mb', 0) or 0)
        breakdown['pass1_shared_mb'] = int(pass1_shared_mb)
        breakdown['pass2_shared_mb'] = int(pass2_shared_mb)
        breakdown['pass1_total_mb'] = int(pass1_total)
        breakdown['pass2_total_mb'] = int(pass2_total)
        breakdown['peak_pass'] = str(peak_pass)
        breakdown['peak_total_mb'] = int(peak_total)
        return breakdown

    def _apply_pipeline_profile_if_needed(self, *_args, force=False):
        profile = self._normalized_pipeline_profile(self.pipeline_worker_profile.get())
        if profile != self.pipeline_worker_profile.get():
            self.pipeline_worker_profile.set(profile)

        if profile in PIPELINE_PROFILE_VALUES:
            instances = self._pipeline_profile_values(profile)
            if force:
                self.pipeline_denoise_instances.set(instances['denoise'])
                self.pipeline_colorize_instances.set(instances['colorize'])
                self.pipeline_upscale_instances.set(instances['upscale'])

        clamped = self._clamp_pipeline_instances(
            self.pipeline_denoise_instances.get(),
            self.pipeline_colorize_instances.get(),
            self.pipeline_upscale_instances.get(),
        )
        if clamped['denoise'] != self.pipeline_denoise_instances.get():
            self.pipeline_denoise_instances.set(clamped['denoise'])
        if clamped['colorize'] != self.pipeline_colorize_instances.get():
            self.pipeline_colorize_instances.set(clamped['colorize'])
        if clamped['upscale'] != self.pipeline_upscale_instances.get():
            self.pipeline_upscale_instances.set(clamped['upscale'])

    def _coerce_quality_preset_payload(self, payload):
        payload = payload if isinstance(payload, dict) else {}
        merged = dict(QUALITY_PRESET_DEFAULTS)

        merged['denoise_sigma'] = self._coerce_int(
            payload.get('denoise_sigma', merged['denoise_sigma']),
            merged['denoise_sigma'],
            1,
            100,
        )

        chroma_mode = str(payload.get('chroma_resize_mode', merged['chroma_resize_mode'])).upper()
        if chroma_mode not in {'BICUBIC', 'BILINEAR', 'LANCZOS'}:
            chroma_mode = 'BICUBIC'
        merged['chroma_resize_mode'] = chroma_mode

        merged['edge_chroma_protection'] = self._coerce_bool(
            payload.get('edge_chroma_protection', merged['edge_chroma_protection']),
            merged['edge_chroma_protection'],
        )
        merged['edge_chroma_strength'] = self._coerce_int(
            payload.get('edge_chroma_strength', merged['edge_chroma_strength']),
            merged['edge_chroma_strength'],
            0,
            100,
        )

        merged['line_ink_protection'] = self._coerce_bool(
            payload.get('line_ink_protection', payload.get('ink_protection', merged['line_ink_protection'])),
            merged['line_ink_protection'],
        )
        merged['line_ink_protection_strength'] = self._coerce_int(
            payload.get('line_ink_protection_strength', payload.get('ink_protection_strength', merged['line_ink_protection_strength'])),
            merged['line_ink_protection_strength'],
            0,
            100,
        )

        merged['screentone_chroma_smoothing'] = self._coerce_bool(
            payload.get('screentone_chroma_smoothing', merged['screentone_chroma_smoothing']),
            merged['screentone_chroma_smoothing'],
        )
        merged['screentone_smoothing_strength'] = self._coerce_int(
            payload.get('screentone_smoothing_strength', merged['screentone_smoothing_strength']),
            merged['screentone_smoothing_strength'],
            0,
            100,
        )

        return merged

    def _snapshot_quality_preset_from_vars(self):
        return self._coerce_quality_preset_payload({
            'denoise_sigma': self.denoise_sigma.get(),
            'chroma_resize_mode': self.chroma_resize_mode.get(),
            'edge_chroma_protection': self.edge_chroma_protection.get(),
            'edge_chroma_strength': self.edge_chroma_strength.get(),
            'line_ink_protection': self.line_ink_protection.get(),
            'line_ink_protection_strength': self.line_ink_protection_strength.get(),
            'screentone_chroma_smoothing': self.screentone_chroma_smoothing.get(),
            'screentone_smoothing_strength': self.screentone_smoothing_strength.get(),
        })

    def _apply_quality_preset_payload_to_vars(self, payload):
        merged = self._coerce_quality_preset_payload(payload)
        self.denoise_sigma.set(merged['denoise_sigma'])
        self.chroma_resize_mode.set(merged['chroma_resize_mode'])
        self.edge_chroma_protection.set(merged['edge_chroma_protection'])
        self.edge_chroma_strength.set(merged['edge_chroma_strength'])
        self.line_ink_protection.set(merged['line_ink_protection'])
        self.line_ink_protection_strength.set(merged['line_ink_protection_strength'])
        self.screentone_chroma_smoothing.set(merged['screentone_chroma_smoothing'])
        self.screentone_smoothing_strength.set(merged['screentone_smoothing_strength'])

    def _ensure_quality_preset_integrity(self):
        sanitized_presets = {}
        if isinstance(self.quality_presets, dict):
            for name, payload in self.quality_presets.items():
                if not isinstance(name, str):
                    continue
                clean_name = name.strip()
                if not clean_name:
                    continue
                sanitized_presets[clean_name] = self._coerce_quality_preset_payload(payload)

        if 'Default' not in sanitized_presets:
            sanitized_presets['Default'] = dict(QUALITY_PRESET_DEFAULTS)

        self.quality_presets = sanitized_presets

        if self.active_quality_preset_global not in self.quality_presets:
            self.active_quality_preset_global = 'Default'

        cleaned_overrides = {}
        if isinstance(self.quality_preset_overrides, dict):
            for folder_key, preset_name in self.quality_preset_overrides.items():
                normalized_key = self._normalize_folder_key(folder_key)
                if normalized_key and preset_name in self.quality_presets:
                    cleaned_overrides[normalized_key] = preset_name
        self.quality_preset_overrides = cleaned_overrides

    def _ordered_preset_names(self):
        self._ensure_quality_preset_integrity()
        names = sorted((name for name in self.quality_presets if name != 'Default'), key=str.lower)
        return ['Default'] + names

    def _active_quality_preset_for_folder(self, folder_path):
        self._ensure_quality_preset_integrity()
        normalized_key = self._normalize_folder_key(folder_path)
        if normalized_key:
            preset_name = self.quality_preset_overrides.get(normalized_key)
            if preset_name in self.quality_presets:
                return preset_name
        if self.active_quality_preset_global in self.quality_presets:
            return self.active_quality_preset_global
        return 'Default'

    def _quality_context_folder(self):
        folder_value = self.input_folder.get().strip().strip("'\"")
        if folder_value:
            return folder_value

        session = self.preview_session
        if isinstance(session, dict):
            fallback_folder = str(session.get('image_folder', '')).strip().strip("'\"")
            if fallback_folder:
                return fallback_folder

        return ''

    def _set_active_quality_preset(self, preset_name, folder_path=None, scope='global'):
        self._ensure_quality_preset_integrity()
        if preset_name not in self.quality_presets:
            return False

        if scope == 'folder':
            normalized_key = self._normalize_folder_key(folder_path)
            if not normalized_key:
                return False
            self.quality_preset_overrides[normalized_key] = preset_name
        elif scope == 'clear_folder':
            normalized_key = self._normalize_folder_key(folder_path)
            if normalized_key in self.quality_preset_overrides:
                self.quality_preset_overrides.pop(normalized_key, None)
        else:
            self.active_quality_preset_global = preset_name

        self._ensure_quality_preset_integrity()
        return True

    def _refresh_preview_preset_selector(self, selected_name=None):
        session = self.preview_session
        if not session:
            return

        window = session.get('window')
        refresher = session.get('refresh_preset_selector')
        if not window or not callable(refresher):
            return

        try:
            if not window.winfo_exists():
                return
        except tk.TclError:
            return

        refresher(selected_name)

    def _sync_open_preset_selectors(self, source=None, selected_name=None):
        if source != 'advanced':
            refresher = self._advanced_preset_selector_refresher
            if callable(refresher):
                try:
                    refresher(selected_name)
                except tk.TclError:
                    self._advanced_preset_selector_refresher = None

        if source != 'preview':
            self._refresh_preview_preset_selector(selected_name=selected_name)

    def _can_autosync_preset(self, preset_name):
        self._ensure_quality_preset_integrity()
        if preset_name != 'Default':
            return True
        return len(self.quality_presets) <= 1

    def _save_current_values_as_preset(self, preset_name, allow_overwrite=False):
        self._ensure_quality_preset_integrity()
        clean_name = (preset_name or '').strip()
        if not clean_name:
            return False, 'Preset name cannot be empty.'
        if clean_name == 'Default':
            return False, "'Default' is reserved and cannot be overwritten."
        if clean_name in self.quality_presets and not allow_overwrite:
            return False, f"Preset '{clean_name}' already exists."

        self.quality_presets[clean_name] = self._snapshot_quality_preset_from_vars()
        self._ensure_quality_preset_integrity()
        return True, clean_name

    def _rename_quality_preset(self, old_name, new_name):
        self._ensure_quality_preset_integrity()
        source = (old_name or '').strip()
        target = (new_name or '').strip()

        if source not in self.quality_presets:
            return False, f"Preset '{source}' was not found."
        if source == 'Default':
            return False, "'Default' cannot be renamed."
        if not target:
            return False, 'New preset name cannot be empty.'
        if target == 'Default':
            return False, "'Default' is reserved."
        if target in self.quality_presets and target != source:
            return False, f"Preset '{target}' already exists."

        self.quality_presets[target] = self.quality_presets.pop(source)
        if self.active_quality_preset_global == source:
            self.active_quality_preset_global = target

        for folder_key, preset_name in list(self.quality_preset_overrides.items()):
            if preset_name == source:
                self.quality_preset_overrides[folder_key] = target

        self._ensure_quality_preset_integrity()
        return True, target

    def _delete_quality_preset(self, preset_name):
        self._ensure_quality_preset_integrity()
        clean_name = (preset_name or '').strip()
        if clean_name not in self.quality_presets:
            return False, f"Preset '{clean_name}' was not found."
        if clean_name == 'Default':
            return False, "'Default' cannot be deleted."

        self.quality_presets.pop(clean_name, None)
        if self.active_quality_preset_global == clean_name:
            self.active_quality_preset_global = 'Default'

        for folder_key, active_name in list(self.quality_preset_overrides.items()):
            if active_name == clean_name:
                self.quality_preset_overrides[folder_key] = 'Default'

        self._ensure_quality_preset_integrity()
        return True, clean_name

    def _build_unique_preset_name(self, candidate_name):
        base = (candidate_name or 'Imported Preset').strip() or 'Imported Preset'
        if base not in self.quality_presets:
            return base

        suffix = 2
        while True:
            next_name = f"{base} ({suffix})"
            if next_name not in self.quality_presets:
                return next_name
            suffix += 1

    def _export_quality_presets(self, export_path):
        self._ensure_quality_preset_integrity()
        payload = {
            'version': 1,
            'active_global': self.active_quality_preset_global,
            'folder_overrides': dict(self.quality_preset_overrides),
            'items': self.quality_presets,
        }
        with open(export_path, 'w', encoding='utf-8') as output_file:
            json.dump(payload, output_file, indent=4)

    def _import_quality_presets(self, import_path):
        self._ensure_quality_preset_integrity()
        with open(import_path, 'r', encoding='utf-8') as input_file:
            imported = json.load(input_file)

        if not isinstance(imported, dict):
            raise ValueError('Invalid preset file format.')

        imported_items = None
        imported_active_global = None
        imported_overrides = {}

        if isinstance(imported.get('items'), dict):
            imported_items = imported['items']
            imported_active_global = imported.get('active_global')
            imported_overrides = imported.get('folder_overrides', {})
        elif isinstance(imported.get('presets'), dict):
            imported_items = imported['presets']
            imported_active_global = imported.get('active_global')
            imported_overrides = imported.get('folder_overrides', {})
        else:
            imported_items = imported

        if not isinstance(imported_items, dict):
            raise ValueError('Preset payload must be an object map.')

        alias_map = {}
        merged_count = 0
        for raw_name, raw_payload in imported_items.items():
            if not isinstance(raw_name, str):
                continue
            clean_name = raw_name.strip()
            if not clean_name:
                continue

            clean_payload = self._coerce_quality_preset_payload(raw_payload)
            target_name = clean_name
            if target_name in self.quality_presets and self.quality_presets[target_name] != clean_payload:
                target_name = self._build_unique_preset_name(target_name)

            self.quality_presets[target_name] = clean_payload
            alias_map[clean_name] = target_name
            merged_count += 1

        if merged_count == 0:
            raise ValueError('No valid presets were found in the selected file.')

        if isinstance(imported_active_global, str):
            mapped_global = alias_map.get(imported_active_global.strip(), imported_active_global.strip())
            if mapped_global in self.quality_presets:
                self.active_quality_preset_global = mapped_global

        if isinstance(imported_overrides, dict):
            for folder_key, preset_name in imported_overrides.items():
                normalized_key = self._normalize_folder_key(folder_key)
                if not normalized_key or not isinstance(preset_name, str):
                    continue
                mapped_name = alias_map.get(preset_name.strip(), preset_name.strip())
                if mapped_name in self.quality_presets:
                    self.quality_preset_overrides[normalized_key] = mapped_name

        self._ensure_quality_preset_integrity()
        return merged_count


    from ui.preview_window import open_preview_window
    from ui.settings_panel import open_advanced_settings
    def open_bug_report_link(self, event):
        """Opens the GitHub issues page in a new browser tab."""
        webbrowser.open_new_tab("https://github.com/sepTN/Manga-Colorizer-GUI/issues")

    def select_model_path(self, path_var):
        file_path = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=(("Model files", "*.pt *.pth *.zip"), ("All files", "*.*"))
        )
        if file_path:
            path_var.set(file_path)

    def select_input_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.input_folder.set(folder_path)
            current_output = self.output_folder.get()
            if not current_output or current_output.startswith(os.path.dirname(self.input_folder.get())):
                 self.output_folder.set(os.path.join(folder_path, 'output'))

    def select_output_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.output_folder.set(folder_path)

    def select_external_color_dir(self):
        folder_path = filedialog.askdirectory(title="Select External Color Source Folder")
        if folder_path:
            self.external_color_source_dir.set(folder_path)

    def _on_external_color_toggled(self):
        self._sync_external_color_state()
        self._refresh_start_button_state()

    def _sync_external_color_state(self):
        """Grey out denoise/colorize and enable/disable ext dir entry based on external checkbox."""
        ext_on = self.external_color_source_enabled.get()
        if ext_on:
            # Disable denoise and colorize checkboxes
            if hasattr(self, '_colorize_cb'):
                self._colorize_cb.config(state='disabled')
            if hasattr(self, '_denoise_cb'):
                self._denoise_cb.config(state='disabled')
            # Enable external dir entry and browse
            if hasattr(self, '_ext_color_entry'):
                self._ext_color_entry.config(state='normal')
            if hasattr(self, '_ext_color_browse'):
                self._ext_color_browse.config(state='normal')
        else:
            # Re-enable denoise and colorize checkboxes
            if hasattr(self, '_colorize_cb'):
                self._colorize_cb.config(state='normal')
            if hasattr(self, '_denoise_cb'):
                self._denoise_cb.config(state='normal')
            # Disable external dir entry and browse
            if hasattr(self, '_ext_color_entry'):
                self._ext_color_entry.config(state='disabled')
            if hasattr(self, '_ext_color_browse'):
                self._ext_color_browse.config(state='disabled')

    def _precision_hardware_summary_text(self):
        device_name = str(getattr(self.config, 'device', 'cpu')).strip().lower()
        summary = get_cuda_device_summary()
        if device_name != 'cuda':
            if summary.get('available'):
                name = summary.get('name', 'CUDA')
                capability = summary.get('capability') or 'unknown'
                bf16 = 'yes' if summary.get('bf16_supported') else 'no'
                fp8 = 'yes' if summary.get('fp8_supported') else 'no'
                tf32 = 'yes' if summary.get('tf32_available') else 'no'
                return (
                    f"Device: {device_name}\n"
                    f"CUDA available: yes ({name}, CC {capability})\n"
                    f"BF16 supported: {bf16}\n"
                    f"FP8 supported: {fp8}\n"
                    f"TF32 available: {tf32}"
                )
            return f"Device: {device_name}\nCUDA not available. Using CPU (FP32)."

        return format_cuda_device_summary(summary)

    def _precision_log_callback(self, message, level='W'):
        if not message:
            return
        payload = {'type': 'log', 'level': level, 'message': str(message)}
        try:
            self.progress_queue.put(payload)
        except Exception:
            self.log(str(message), level=level)

    def log(self, message, track_dynamic=False, record_full=True, level='D'):
        if record_full:
            self._append_full_log_event(message, level)
        if not track_dynamic:
            self._dynamic_log_line = None
        self.log_text.config(state="normal")
        self.log_text.insert(tk.END, message + "\n")
        if track_dynamic:
            self._dynamic_log_line = self.log_text.index("end-2l linestart")
        self.log_text.config(state="disabled")
        self.log_text.see(tk.END)

    def update_dynamic_log(self, message):
        self.log_text.config(state="normal")
        if self._dynamic_log_line is None:
            self.log_text.insert(tk.END, message + "\n")
            self._dynamic_log_line = self.log_text.index("end-2l linestart")
        else:
            line_end = self.log_text.index(f"{self._dynamic_log_line} lineend")
            self.log_text.delete(self._dynamic_log_line, line_end)
            self.log_text.insert(self._dynamic_log_line, message)
        self.log_text.config(state="disabled")
        self.log_text.see(tk.END)

    def _on_log_filter_changed(self):
        self._rebuild_log_from_buffer()

    def _rebuild_log_from_buffer(self):
        self.log_text.config(state="normal")
        self.log_text.delete(1.0, tk.END)
        self._dynamic_log_line = None

        for event in self._full_log_buffer:
            level = self._normalize_event_level(event.get('level', 'D'))
            if not self._should_show_log_level(level):
                continue
            message = str(event.get('message', ''))
            self.log_text.insert(tk.END, message + "\n")

        self.log_text.config(state="disabled")
        self.log_text.see(tk.END)

    def clear_log(self):
        self.log_text.config(state="normal")
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state="disabled")
        self._dynamic_log_line = None
        self._full_log_buffer.clear()

    def _normalize_event_level(self, level):
        normalized = str(level or 'D').strip().upper()
        if normalized in {'E', 'W', 'D', 'T', 'V'}:
            return normalized
        return 'D'

    def _should_show_log_level(self, level):
        normalized = self._normalize_event_level(level)
        return {
            'E': bool(self.log_show_error.get()),
            'W': bool(self.log_show_warning.get()),
            'D': bool(self.log_show_debug.get()),
            'T': bool(self.log_show_trace.get()),
            'V': bool(self.log_show_verbose.get()),
        }.get(normalized, True)

    def _append_full_log_event(self, message, level):
        if not message:
            return

        self._full_log_buffer.append({'level': self._normalize_event_level(level), 'message': str(message)})
        overflow = len(self._full_log_buffer) - MAX_LOG_BUFFER_SIZE
        if overflow > 0:
            del self._full_log_buffer[:overflow]

    def _set_preview_pause_state(self, active, note=''):
        self._preview_pass2_active = bool(active)
        if not isinstance(self.preview_session, dict):
            return

        if self._is_batch_processing_active():
            return

        status_var = self.preview_session.get('status_var')
        if status_var is None:
            return

        try:
            if self._preview_pass2_active:
                pause_message = str(note).strip() or "Upscale pass running; showing cached preview."
                status_var.set(pause_message)
                return

            base_status = str(self.preview_session.get('last_render_status_base', '')).strip()
            if base_status:
                status_var.set(base_status)
            elif str(note).strip():
                status_var.set(str(note).strip())
            else:
                status_var.set("Ready.")
        except tk.TclError:
            self.preview_session = None

    def _refresh_live_vram_meter(self, force=False):
        now = time.time()
        if not force and (now - self._last_vram_poll_ts) < 0.5:
            return
        self._last_vram_poll_ts = now

        stats = get_cuda_memory_stats_mb()
        self._last_cuda_stats = dict(stats) if stats else {}

        if not stats:
            self.vram_bar['value'] = 0
            device_name = str(getattr(self.config, 'device', 'cpu')).lower()
            if device_name != 'cuda':
                self.vram_meter_text.set("VRAM: CUDA telemetry unavailable on current device")
            else:
                self.vram_meter_text.set("VRAM: CUDA telemetry unavailable")
            return

        app_used = float(stats.get('app_used_mb', stats.get('process_reserved_mb', 0.0)))
        available_for_process = float(stats.get('available_for_process_mb', 0.0))
        free_now = float(stats.get('free_mb', 0.0))
        gpu_total = float(stats.get('total_mb', 0.0))

        percent = 0.0
        if available_for_process > 0:
            percent = min(100.0, (app_used / available_for_process) * 100.0)
        self.vram_bar['value'] = percent

        self.vram_meter_text.set(
            f"VRAM(app): {app_used / 1024:.1f}/{available_for_process / 1024:.1f} GB | free now {free_now / 1024:.1f} GB (GPU {gpu_total / 1024:.1f} GB)"
        )

    def _consume_pipeline_calibration(self, report):
        if not isinstance(report, dict):
            return

        self.pipeline_vram_calibration = dict(report)
        profile = report.get('instance_vram_profile_mb', {})
        shared_stage_peaks = report.get('shared_stage_peak_mb', {})
        worker_counts = report.get('worker_counts', {})
        stage_peaks = report.get('stage_peak_inference_mb', {})
        warmup_input_hw = report.get('warmup_input_hw', [])
        projection_source = str(report.get('projection_source', 'init_delta'))
        adaptive_upscale = report.get('adaptive_upscale_limit', {})
        overhead_mb = int(report.get('runtime_overhead_mb', 0) or 0)
        projected_peak_mb = int(report.get('projected_peak_mb', report.get('projected_total_mb', 0)) or 0)
        projected_pass1_mb = int(report.get('projected_pass1_mb', 0) or 0)
        projected_pass2_mb = int(report.get('projected_pass2_mb', 0) or 0)
        shared_pass1_mb = int(report.get('shared_pass1_mb', 0) or 0)
        shared_pass2_mb = int(report.get('shared_pass2_mb', 0) or 0)
        writer_threads = int(report.get('writer_threads', getattr(self.config, 'pipeline_writer_threads', 1)) or 1)
        projected_peak_pass = str(report.get('projected_peak_pass', 'none')).strip().lower()
        peak_label = {
            'pass1': self._pass_label('pass1'),
            'pass2': self._pass_label('pass2'),
        }.get(projected_peak_pass, 'none')

        if isinstance(profile, dict):
            denoise_mb = int(profile.get('denoise', 0) or 0)
            colorize_mb = int(profile.get('colorize', 0) or 0)
            upscale_mb = int(profile.get('upscale', 0) or 0)
            denoise_workers = int(worker_counts.get('denoise', 0) or 0) if isinstance(worker_counts, dict) else 0
            colorize_workers = int(worker_counts.get('colorize', 0) or 0) if isinstance(worker_counts, dict) else 0
            upscale_workers = int(worker_counts.get('upscale', 0) or 0) if isinstance(worker_counts, dict) else 0
            self.log(
                (
                    "[*] Runtime VRAM calibration updated "
                    f"(per-instance MB: D={denoise_mb}, C={colorize_mb}, U={upscale_mb}; "
                    f"workers: D={denoise_workers}, C={colorize_workers}, U={upscale_workers}; "
                    f"writer={writer_threads}; "
                    f"overhead={overhead_mb} MB; pass1={projected_pass1_mb} MB; "
                    f"pass2={projected_pass2_mb} MB; shared(P1={shared_pass1_mb}, P2={shared_pass2_mb}) MB; "
                    f"peak={projected_peak_mb} MB [{peak_label}])."
                )
            )

        if isinstance(stage_peaks, dict) and stage_peaks:
            denoise_peak = int(stage_peaks.get('denoise', 0) or 0)
            colorize_peak = int(stage_peaks.get('colorize', 0) or 0)
            upscale_peak = int(stage_peaks.get('upscale', 0) or 0)
            shared_denoise_peak = int(shared_stage_peaks.get('denoise', denoise_peak) or 0) if isinstance(shared_stage_peaks, dict) else denoise_peak
            shared_colorize_peak = int(shared_stage_peaks.get('colorize', colorize_peak) or 0) if isinstance(shared_stage_peaks, dict) else colorize_peak
            shared_upscale_peak = int(shared_stage_peaks.get('upscale', upscale_peak) or 0) if isinstance(shared_stage_peaks, dict) else upscale_peak
            warmup_h = int(warmup_input_hw[0]) if isinstance(warmup_input_hw, list) and len(warmup_input_hw) > 0 else 0
            warmup_w = int(warmup_input_hw[1]) if isinstance(warmup_input_hw, list) and len(warmup_input_hw) > 1 else 0
            self.log(
                (
                    "[*] Warmup peak inference MB "
                    f"(D={denoise_peak}, C={colorize_peak}, U={upscale_peak}; "
                    f"shared D={shared_denoise_peak}, C={shared_colorize_peak}, U={shared_upscale_peak}; "
                    f"warmup={warmup_w}x{warmup_h}; source={projection_source})."
                )
            )

        if isinstance(adaptive_upscale, dict) and adaptive_upscale.get('applied'):
            from_workers = int(adaptive_upscale.get('from_workers', adaptive_upscale.get('from', 0)) or 0)
            to_workers = int(adaptive_upscale.get('to_workers', adaptive_upscale.get('to', 0)) or 0)
            from_instances = int(adaptive_upscale.get('from_instances', 0) or 0)
            to_instances = int(adaptive_upscale.get('to_instances', from_instances) or 0)
            projected_peak_mb = int(adaptive_upscale.get('projected_peak_mb', 0) or 0)
            available_vram_mb = int(adaptive_upscale.get('available_vram_mb', 0) or 0)
            burst_window_mb = int(adaptive_upscale.get('burst_window_mb', 0) or 0)
            overhead_used_mb = int(adaptive_upscale.get('overhead_mb', 0) or 0)
            shared_peak_mb = int(adaptive_upscale.get('shared_peak_mb', 0) or 0)
            trigger = str(adaptive_upscale.get('trigger', '')).strip()
            reason = str(adaptive_upscale.get('reason', '')).strip()

            message = (
                f"[!] Adaptive upscale VRAM gate applied: instances {from_instances} -> {to_instances}, "
                f"workers {from_workers} -> {to_workers}; peak={projected_peak_mb} MB; "
                f"avail={available_vram_mb} MB; burst={burst_window_mb} MB; "
                f"overhead={overhead_used_mb} MB; shared={shared_peak_mb} MB"
            )
            if trigger:
                message = f"{message}; trigger={trigger}"
            if reason:
                message = f"{message}. {reason}"
            self.log(message)

        refresher = self._advanced_pipeline_runtime_refresher
        if callable(refresher):
            try:
                refresher()
            except tk.TclError:
                self._advanced_pipeline_runtime_refresher = None

    def check_queue(self):
        while not self.progress_queue.empty():
            data = self.progress_queue.get()
            event_type = str(data.get('type', '')).strip().lower()

            if event_type == 'log':
                message = str(data.get('message', ''))
                level = self._normalize_event_level(data.get('level', 'D'))
                self._append_full_log_event(message, level)
                if self._should_show_log_level(level):
                    track_dynamic = message.startswith("[*] Processing:") or message.startswith("[*] Pass ")
                    self.log(message, track_dynamic=track_dynamic, record_full=False)

            elif event_type == 'log_update':
                message = str(data.get('message', ''))
                level = self._normalize_event_level(data.get('level', 'D'))
                self._append_full_log_event(message, level)
                if self._should_show_log_level(level):
                    self.update_dynamic_log(message)

            elif event_type == 'progress':
                self.progress_bar['value'] = float(data.get('value', 0) or 0)
                self.eta_text.set(str(data.get('eta', 'ETA: N/A')))

            elif event_type == 'pipeline_calibration':
                self._consume_pipeline_calibration(data.get('report', {}))

            elif event_type == 'pass_state':
                state = str(data.get('state', '')).strip().lower()
                message = str(data.get('message', '')).strip()
                self._pipeline_pass_state = state or 'idle'

                if state == 'pass2':
                    self._set_preview_pause_state(True, message)
                else:
                    was_paused = self._preview_pass2_active
                    self._set_preview_pause_state(False, message)
                    if was_paused and isinstance(self.preview_session, dict):
                        schedule_rerun = self.preview_session.get('schedule_preview_rerun')
                        if callable(schedule_rerun):
                            self.preview_session['pending_reset_scroll'] = False
                            schedule_rerun("Batch resumed", delay_ms=0)

                self._refresh_preview_runtime_mode()

            elif event_type == 'terminal':
                status = str(data.get('status', '')).strip().lower()
                message = str(data.get('message', '')).strip()
                if message:
                    prefix = "[*]"
                    level = 'D'
                    if status in {'failed', 'stopped', 'cancelled'}:
                        prefix = "[!]"
                        level = 'W'
                    self.log(f"{prefix} {message}", level=level)
                self._refresh_preview_runtime_mode()

            elif event_type:
                self.log(f"[!] Unrecognized pipeline event type: {event_type}")

        self._refresh_live_vram_meter()
        self.after(100, self.check_queue)

    def _update_config_from_gui(self):
        """Updates the self.config object with the current values from the GUI widgets."""
        # Basic options
        self.config.input_folder = self.input_folder.get().strip().strip("'\"")
        self.config.output_folder = self.output_folder.get().strip().strip("'\"")
        self.config.colorize = self.enable_colorize.get()
        self.config.upscale = self.enable_upscale.get()
        self.config.denoise = self.enable_denoise.get()

        # Advanced settings
        self.config.colorizer_path = self.colorizer_path.get()
        self.config.upscaler_path = self.upscaler_path.get()
        self.config.upscaler_type = self.upscaler_type.get()
        denoiser_weights_dir = os.path.abspath(os.path.join(backend_path, 'denoising', 'models'))
        self.config._denoiser_weights_dir = denoiser_weights_dir
        self.config.denoiser_weights_dir = denoiser_weights_dir
        self.config.denoise_sigma = self.denoise_sigma.get()
        self.config.upscaler_tile_size = self.upscaler_tile_size.get()
        self.config.colorizer_tile_size = self.colorizer_tile_size.get()
        self.config.tile_pad = self.tile_pad.get()
        # LP settings
        self.config.lp_generator_patch_size = self.lp_generator_patch_size.get()
        self.config.lp_num_patches_h = self.lp_num_patches_h.get()
        self.config.lp_num_patches_w = self.lp_num_patches_w.get()
        self.config.colorized_image_size = self.colorized_image_size.get()
        sanitized_input_size = sanitize_input_width_limit(self.input_image_size.get())
        if sanitized_input_size != self.input_image_size.get():
            self.input_image_size.set(sanitized_input_size)
        self.config.input_image_size = sanitized_input_size
        self.config.force_safe_colorizer_width = self.force_safe_colorizer_width.get()
        self.config.detailed_debug_logs = self.detailed_debug_logs.get()
        self.config.export_ocr_debug = self.export_ocr_debug.get()
        self.config.max_ocr_dimension = self.max_ocr_dimension.get()
        self.config.use_yolo_bubbles = self.use_yolo_bubbles.get()
        self.config.fp_strictness = self.fp_strictness.get()
        self.config.sfx_feather_radius = self.sfx_feather_radius.get()
        self.config.yolo_model_type = self.yolo_model_type.get()
        self.config.chroma_resize_mode = self.chroma_resize_mode.get()
        self.config.edge_chroma_protection = self.edge_chroma_protection.get()
        self.config.edge_chroma_strength = self.edge_chroma_strength.get()
        self.config.line_ink_protection = self.line_ink_protection.get()
        self.config.line_ink_protection_strength = self.line_ink_protection_strength.get()
        self.config.screentone_chroma_smoothing = self.screentone_chroma_smoothing.get()
        self.config.screentone_smoothing_strength = self.screentone_smoothing_strength.get()
        self.config.pipeline_fail_fast = self.pipeline_fail_fast.get()
        self.config.pipeline_worker_profile = self._normalized_pipeline_profile(self.pipeline_worker_profile.get())
        self.config.pipeline_manual_override = (self.config.pipeline_worker_profile == PIPELINE_PROFILE_CUSTOM)
        self.config.pipeline_queue_ram_budget_mb = self._sanitize_queue_budget_mb(
            getattr(self.config, 'pipeline_queue_ram_budget_mb', 0)
        )
        try:
            raw_writer_threads = self.pipeline_writer_threads.get()
        except (tk.TclError, TypeError, ValueError):
            raw_writer_threads = getattr(self.config, 'pipeline_writer_threads', 1)
        self.config.pipeline_writer_threads = self._sanitize_writer_threads(raw_writer_threads)
        self.pipeline_writer_threads.set(self.config.pipeline_writer_threads)
        self.config.colorize_precision_policy = normalize_precision_policy(self.colorize_precision_policy.get())
        self.config.upscale_precision_policy = normalize_precision_policy(self.upscale_precision_policy.get())
        self.config.precision_cast_weights = bool(self.precision_cast_weights.get())
        self.config.precision_fallback = normalize_fallback_mode(self.precision_fallback.get())
        self.config.precision_allow_tf32 = bool(self.precision_allow_tf32.get())
        self.config.preview_fast_mode = bool(self.preview_fast_mode.get())
        self.config.external_color_source_enabled = bool(self.external_color_source_enabled.get())
        self.config.external_color_source_dir = self.external_color_source_dir.get().strip().strip("'\"")
        self.config.enable_sage_attention = self.enable_sage_attention.get()
        self.config.ocr_model_tier = self.ocr_model_tier.get()
        self.config.denoise_max_side = self.denoise_max_side.get()
        self.config.log_callback = self._precision_log_callback
        self.config.enable_torch_compile = self.torch_compile.get()
        self.config.enable_flash_attention = self.enable_flash_attention.get()
        self.config.enable_cpu_offload = self.enable_cpu_offload.get()

        # SageAttention: monkey-patch F.scaled_dot_product_attention for RTX 30/40/50
        if self.config.enable_sage_attention:
            try:
                from Backend.sage_attention import apply_monkey_patch, get_installed_version
                if get_installed_version() is None:
                    # Not installed — silently disable toggle (user can install from Advanced Settings)
                    self.enable_sage_attention.set(False)
                    self.config.enable_sage_attention = False
                else:
                    ok, msg = apply_monkey_patch()
                    if not ok:
                        self.log(f"[WARNING] SageAttention: {msg}", level='W')
            except Exception as e:
                self.log(f"[WARNING] SageAttention patch failed: {e}", level='W')
        else:
            try:
                from Backend.sage_attention import remove_monkey_patch
                remove_monkey_patch()
            except Exception:
                pass

        effective_instances = self._effective_pipeline_instances(device_name=getattr(self.config, 'device', 'cpu'))
        self.config.pipeline_denoise_instances = effective_instances['denoise']
        self.config.pipeline_colorize_instances = effective_instances['colorize']
        self.config.pipeline_upscale_instances = effective_instances['upscale']

        # Legacy aliases for compatibility with older settings/code paths.
        self.config.pipeline_instances_denoise = self.config.pipeline_denoise_instances
        self.config.pipeline_instances_colorize = self.config.pipeline_colorize_instances
        self.config.pipeline_instances_upscale = self.config.pipeline_upscale_instances
        self.config.pipeline_workers_denoise = self.config.pipeline_denoise_instances
        self.config.pipeline_workers_colorize = self.config.pipeline_colorize_instances
        self.config.pipeline_workers_upscale = self.config.pipeline_upscale_instances

    def _get_and_manage_models(self, precision_policy_override=None):
        """
        Checks GUI toggles and loads/returns models as needed.
        This is called by the processing thread before each image.
        """
        try:
            device_name = getattr(self.config, 'device', 'cpu')
            precision_signature_base = (
                normalize_precision_policy(getattr(self.config, 'colorize_precision_policy', 'auto')),
                normalize_precision_policy(getattr(self.config, 'upscale_precision_policy', 'auto')),
                bool(getattr(self.config, 'precision_cast_weights', False)),
                normalize_fallback_mode(getattr(self.config, 'precision_fallback', 'per_image')),
                bool(getattr(self.config, 'precision_allow_tf32', True)),
            )
            precision_override = normalize_precision_policy(precision_policy_override) if precision_policy_override else None
            precision_signature_override = precision_signature_base + (precision_override,)
            current_signatures = {
                'colorizer': (self.config.colorizer_path, device_name, precision_signature_override),
                'upscaler': (self.config.upscaler_path, self.config.upscaler_type, device_name, precision_signature_base),
                'denoiser': (device_name, precision_signature_override),
            }

            def _config_with_precision_override(policy_override):
                if not policy_override:
                    return self.config
                preview_config = copy.copy(self.config)
                preview_config.precision_policy_override = policy_override
                return preview_config

            colorizer_config = _config_with_precision_override(precision_override)
            denoiser_config = _config_with_precision_override(precision_override)

            should_clear_cache = False

            if self.colorizer_model is not None and self._model_signatures.get('colorizer') != current_signatures['colorizer']:
                if self.detailed_debug_logs.get():
                    self.progress_queue.put({'type': 'log', 'message': "[*] Reloading Colorizer model due to model path/device change..."})
                self.colorizer_model = None
                self._model_signatures['colorizer'] = None
                should_clear_cache = True

            if self.upscaler_model is not None and self._model_signatures.get('upscaler') != current_signatures['upscaler']:
                if self.detailed_debug_logs.get():
                    self.progress_queue.put({'type': 'log', 'message': "[*] Reloading Upscaler model due to path/type/device change..."})
                self.upscaler_model = None
                self._model_signatures['upscaler'] = None
                should_clear_cache = True

            if self.denoiser_model is not None and self._model_signatures.get('denoiser') != current_signatures['denoiser']:
                if self.detailed_debug_logs.get():
                    self.progress_queue.put({'type': 'log', 'message': "[*] Reloading Denoiser model due to device change..."})
                self.denoiser_model = None
                self._model_signatures['denoiser'] = None
                should_clear_cache = True

            if should_clear_cache:
                clear_torch_cache()

            # --- Colorizer ---
            if self.config.colorize and self.colorizer_model is None:
                if self.detailed_debug_logs.get():
                    self.progress_queue.put({'type': 'log', 'message': "[*] Colorizing enabled. Initializing Colorizer model..."})
                self.colorizer_model = MangaColorizator(colorizer_config)
                self._model_signatures['colorizer'] = current_signatures['colorizer']

            # --- Upscaler ---
            if self.config.upscale and self.upscaler_model is None:
                if self.detailed_debug_logs.get():
                    self.progress_queue.put({'type': 'log', 'message': "[*] Upscaling enabled. Initializing Upscaler model..."})
                self.upscaler_model = MangaUpscaler(self.config)
                self._model_signatures['upscaler'] = current_signatures['upscaler']

            # --- Denoiser ---
            if self.config.denoise and self.denoiser_model is None:
                if self.detailed_debug_logs.get():
                    self.progress_queue.put({'type': 'log', 'message': "[*] Denoising enabled. Initializing Denoiser model..."})
                self.denoiser_model = MangaDenoiser(denoiser_config)
                self._model_signatures['denoiser'] = current_signatures['denoiser']

            # Return the currently active models based on toggles
            active_colorizer = self.colorizer_model if self.config.colorize else None
            active_upscaler = self.upscaler_model if self.config.upscale else None
            active_denoiser = self.denoiser_model if self.config.denoise else None

            return active_colorizer, active_upscaler, active_denoiser

        except Exception as e:
            self.progress_queue.put({'type': 'log', 'message': f"[!!!] FAILED to initialize a model. Error: {e}"})
            # Return None for all models to stop processing for the current image
            return None, None, None


    def start_processing(self):
        if not self._has_enabled_processing_stage():
            self.log("[!] No processing stages selected. Enable at least one of Denoise, Colorize, or Upscale.")
            self._refresh_start_button_state()
            return

        # Validate external color source directory if enabled
        if self.external_color_source_enabled.get():
            ext_dir = self.external_color_source_dir.get().strip().strip("'\"")
            if not ext_dir:
                self.log("[!] External Color Source is enabled but no directory is selected.")
                return
            if not os.path.isdir(ext_dir):
                self.log(f"[!] External Color Source directory not found: '{ext_dir}'")
                return

        # Check if another process is already running
        if self.processing_thread and self.processing_thread.is_alive():
            self.log("[!] A process is already running. Please wait or pause it.")
            return

        input_path = self.input_folder.get().strip().strip("'\"")
        output_path = self.output_folder.get().strip().strip("'\"")

        if not input_path or not output_path:
            self.log("[!] Please select both input and output folders.")
            return
        if not os.path.isdir(input_path):
            self.log(f"[!] Error: Input folder not found at '{input_path}'")
            return

        os.makedirs(output_path, exist_ok=True)
        
        # Store the input path for the new process
        self.current_input_path = input_path

        self.start_button.config(state="disabled")
        self.pause_button.config(state="normal", text="Pause", command=self.pause_processing)
        self.terminate_event.clear()
        self.pause_event.clear()
        self._close_deadline_mono = None
        self._pipeline_pass_state = 'idle'
        self._set_preview_pause_state(False)

        # Reset progress bar for a new run
        self.progress_bar['value'] = 0
        self.eta_text.set("ETA: N/A")

        self.processing_thread = threading.Thread(
            target=self.run_batch_processing,
            args=(input_path, output_path),
            daemon=True
        )
        self.processing_thread.start()
        self.after(0, self._refresh_preview_runtime_mode)

    def pause_processing(self):
        self.log("\n[!] Pausing intake... in-flight pipeline items will finish, then workers will idle.")
        self.pause_event.set()
        self.pause_button.config(text="Continue", command=self.continue_processing)

    def continue_processing(self):
        new_input_path = self.input_folder.get().strip().strip("'\"")

        # Check if the input folder has changed, requiring a full restart
        if new_input_path != self.current_input_path:
            self.log("\n[!] Input folder has changed. Restarting the entire batch process...")

            # Signal the current thread to stop.
            self.terminate_event.set()
            # Un-pause the thread so it can see the terminate signal and exit its loop.
            self.pause_event.clear()

            def restart_after_termination():
                if self.processing_thread and self.processing_thread.is_alive():
                    # Check again in 100ms
                    self.after(100, restart_after_termination)
                else:
                    # The old thread is gone, now we can start a new process.
                    self.start_processing()

            # Start the check to wait for the old thread to terminate.
            self.after(100, restart_after_termination)
        else:
            # Path is the same, just resume.
            self.log("\n[*] Resuming...")
            self.pause_event.clear()
            self.pause_button.config(text="Pause", command=self.pause_processing)

    def run_batch_processing(self, input_path, output_path):
        batch_stopped_on_failure = False
        batch_pipeline = None

        try:
            self._pipeline_pass_state = 'idle'

            supported_formats = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
            images_to_process = []

            for root, _, files in os.walk(input_path):
                for filename in files:
                    if filename.lower().endswith(supported_formats):
                        rel_dir = os.path.relpath(root, input_path)
                        if rel_dir == '.':
                            rel_dir = ''
                        images_to_process.append((os.path.join(root, filename), rel_dir))

            images_to_process.sort(key=lambda item: item[0])
            total_images = len(images_to_process)

            if not images_to_process:
                self.progress_queue.put({'type': 'log', 'level': 'W', 'message': f"[!] No supported image files found in '{input_path}'."})
                return

            self._full_log_buffer.clear()
            self.progress_queue.put({'type': 'log', 'level': 'D', 'message': f"\n[*] Found {total_images} images to process."})

            grouped_by_folder = OrderedDict()
            for full_path, rel_dir in images_to_process:
                if rel_dir not in grouped_by_folder:
                    grouped_by_folder[rel_dir] = []
                grouped_by_folder[rel_dir].append((full_path, rel_dir))

            # Refresh config and instance settings once per batch run.
            self._update_config_from_gui()

            # External color source overrides Pass 1 entirely
            external_enabled = bool(getattr(self.config, 'external_color_source_enabled', False))
            if external_enabled:
                self.config.denoise = False
                self.config.colorize = False
                ext_dir = str(getattr(self.config, 'external_color_source_dir', '') or '')
                if not ext_dir or not os.path.isdir(ext_dir):
                    self.progress_queue.put(
                        {
                            'type': 'log',
                            'level': 'E',
                            'message': f"[!!!] External color source directory not found: '{ext_dir}'",
                        }
                    )
                    return
                self.progress_queue.put(
                    {
                        'type': 'log',
                        'level': 'D',
                        'message': f"[*] External color source mode: Pass 1 skipped, colors from '{ext_dir}'",
                    }
                )

            overwrite = self.overwrite_existing.get()
            detailed_logs = self.detailed_debug_logs.get()
            fail_fast = bool(getattr(self.config, 'pipeline_fail_fast', True))
            raw_input_limit = getattr(self.config, 'input_image_size', DEFAULT_INPUT_WIDTH)
            input_limit = sanitize_input_width_limit(raw_input_limit)
            self.config.input_image_size = input_limit
            device_name = getattr(self.config, 'device', 'cpu')
            effective_instances = self._effective_pipeline_instances(device_name=device_name)
            active_instances = self._active_pipeline_instances(device_name=device_name)

            self.config.pipeline_denoise_instances = effective_instances['denoise']
            self.config.pipeline_colorize_instances = effective_instances['colorize']
            self.config.pipeline_upscale_instances = effective_instances['upscale']
            self.config.pipeline_instances_denoise = self.config.pipeline_denoise_instances
            self.config.pipeline_instances_colorize = self.config.pipeline_colorize_instances
            self.config.pipeline_instances_upscale = self.config.pipeline_upscale_instances
            self.config.pipeline_workers_denoise = self.config.pipeline_denoise_instances
            self.config.pipeline_workers_colorize = self.config.pipeline_colorize_instances
            self.config.pipeline_workers_upscale = self.config.pipeline_upscale_instances
            try:
                raw_writer_threads = self.pipeline_writer_threads.get()
            except (tk.TclError, TypeError, ValueError):
                raw_writer_threads = getattr(self.config, 'pipeline_writer_threads', 1)
            self.config.pipeline_writer_threads = self._sanitize_writer_threads(raw_writer_threads)
            self.pipeline_writer_threads.set(self.config.pipeline_writer_threads)
            self.config.colorize_precision_policy = normalize_precision_policy(self.colorize_precision_policy.get())
            self.config.upscale_precision_policy = normalize_precision_policy(self.upscale_precision_policy.get())
            self.config.precision_cast_weights = bool(self.precision_cast_weights.get())
            self.config.precision_fallback = normalize_fallback_mode(self.precision_fallback.get())
            self.config.precision_allow_tf32 = bool(self.precision_allow_tf32.get())

            instance_summary = (
                f"D:{active_instances['denoise']} "
                f"C:{active_instances['colorize']} "
                f"U:{active_instances['upscale']}"
            )
            self.progress_queue.put(
                {
                    'type': 'log',
                    'level': 'D',
                    'message': (
                        f"[*] Pipeline instances => {instance_summary} "
                        f"(writer threads={int(self.config.pipeline_writer_threads)})"
                    ),
                }
            )

            projected = self._effective_pipeline_vram_breakdown_mb(device_name=device_name)
            projected_vram = int(projected.get('peak_total_mb', 0) or 0)
            projected_pass1 = int(projected.get('pass1_total_mb', 0) or 0)
            projected_pass2 = int(projected.get('pass2_total_mb', 0) or 0)
            projected_peak_pass = str(projected.get('peak_pass', 'none')).strip().lower()
            peak_label = {
                'pass1': self._pass_label('pass1'),
                'pass2': self._pass_label('pass2'),
            }.get(projected_peak_pass, 'none')

            current_cuda_stats = get_cuda_memory_stats_mb()
            warning = ""
            if current_cuda_stats:
                available_now = int(current_cuda_stats.get('available_for_process_mb', 0))
                if available_now > 0 and projected_vram > available_now:
                    warning = (
                        f"Projected peak VRAM ~{projected_vram} MB ({peak_label}; "
                        f"pass1={projected_pass1} MB, pass2={projected_pass2} MB) exceeds "
                        f"current available-for-app memory ~{available_now} MB."
                    )
                elif available_now > 0 and projected_vram > int(available_now * 0.8):
                    warning = (
                        f"Projected peak VRAM ~{projected_vram} MB ({peak_label}; "
                        f"pass1={projected_pass1} MB, pass2={projected_pass2} MB) is near "
                        f"the current available-for-app memory ~{available_now} MB."
                    )
            else:
                warning = vram_warning_text(
                    active_instances['denoise'],
                    active_instances['colorize'],
                    active_instances['upscale'],
                    instance_profile_mb=self._pipeline_instance_vram_profile_mb(device_name=device_name),
                    overhead_mb=self._pipeline_runtime_overhead_mb(),
                )

            if warning:
                self.progress_queue.put({'type': 'log', 'level': 'W', 'message': f"[!] {warning}"})
            elif detailed_logs:
                self.progress_queue.put(
                    {
                        'type': 'log',
                        'level': 'D',
                        'message': (
                            f"[*] Projected peak pipeline VRAM: ~{projected_vram} MB "
                            f"({peak_label}; pass1={projected_pass1} MB, pass2={projected_pass2} MB)."
                        ),
                    }
                )

            if detailed_logs:
                try:
                    raw_input_limit_int = int(raw_input_limit)
                except (TypeError, ValueError):
                    raw_input_limit_int = None

                if raw_input_limit_int != input_limit:
                    self.progress_queue.put(
                        {
                            'type': 'log',
                            'level': 'D',
                            'message': f"[*] Input clamp sanitized to {input_limit}px (requested {raw_input_limit}).",
                        }
                    )

            model_factories = {}
            base_factory_cfg = copy.copy(self.config)
            if getattr(base_factory_cfg, 'colorizer_path', ''):
                base_factory_cfg.colorizer_path = os.path.abspath(base_factory_cfg.colorizer_path)
            if getattr(base_factory_cfg, 'upscaler_path', ''):
                base_factory_cfg.upscaler_path = os.path.abspath(base_factory_cfg.upscaler_path)
            base_factory_cfg._denoiser_weights_dir = os.path.abspath(os.path.join(backend_path, 'denoising', 'models'))
            base_factory_cfg.denoiser_weights_dir = base_factory_cfg._denoiser_weights_dir

            if self.config.denoise:
                denoise_cfg = copy.copy(base_factory_cfg)
                model_factories['denoise'] = lambda _c=denoise_cfg: MangaDenoiser(_c)
            if self.config.colorize:
                colorize_cfg = copy.copy(base_factory_cfg)
                model_factories['colorize'] = lambda _c=colorize_cfg: MangaColorizator(_c)
            if self.config.upscale:
                upscale_cfg = copy.copy(base_factory_cfg)
                model_factories['upscale'] = lambda _c=upscale_cfg: MangaUpscaler(_c)

            warmup_h = min(1024, int(input_limit))
            warmup_w = int(input_limit)
            if images_to_process:
                sample_path = images_to_process[0][0]
                try:
                    with PIL.Image.open(sample_path) as sample_image:
                        sample_w, sample_h = sample_image.size

                    if sample_w > input_limit and sample_w > 0:
                        ratio = input_limit / float(sample_w)
                        sample_w = int(input_limit)
                        sample_h = max(1, int(round(sample_h * ratio)))

                    warmup_h = max(1, int(sample_h))
                    warmup_w = max(1, int(sample_w))
                except Exception as warmup_shape_err:
                    if detailed_logs:
                        self.progress_queue.put(
                            {
                                'type': 'log',
                                'level': 'W',
                                'message': f"[!] Warmup shape probe failed ({warmup_shape_err}); using fallback shape.",
                            }
                        )

            warmup_spec = {
                'input_hw': [warmup_h, warmup_w],
                'upscale_factor': int(getattr(self.config, 'upscale_factor', 4)),
                'denoise_sigma': int(getattr(self.config, 'denoise_sigma', 25)),
                'colorized_image_size': int(getattr(self.config, 'colorized_image_size', 576)),
                'force_safe_colorizer_width': bool(getattr(self.config, 'force_safe_colorizer_width', False)),
                # Representative tile sizes for warmup (small list keeps warmup quick)
                'tile_sizes': list(getattr(self.config, 'warmup_tile_sizes', [64, 128, 256])),
                'warmup_max_tile_runs': int(getattr(self.config, 'warmup_max_tile_runs', 3)),
            }

            try:
                batch_pipeline = ProcessingPipeline(
                    config=self.config,
                    model_factories=model_factories,
                    instance_counts=active_instances,
                    progress_callback=lambda payload: self.progress_queue.put(payload),
                    terminate_event=self.terminate_event,
                    pause_event=self.pause_event,
                    warmup_spec=warmup_spec,
                    fail_fast=fail_fast,
                )
                self.progress_queue.put(
                    {
                        'type': 'pipeline_calibration',
                        'report': batch_pipeline.get_vram_calibration_report(),
                    }
                )
            except Exception as pipeline_err:
                self.progress_queue.put(
                    {
                        'type': 'log',
                        'level': 'E',
                        'message': f"[!!!] Failed to initialize batch pipeline. Error: {pipeline_err}",
                    }
                )
                batch_stopped_on_failure = True

            if batch_pipeline is not None and not self.terminate_event.is_set() and not batch_stopped_on_failure:
                try:
                    batch_stats = batch_pipeline.run_batch(
                        grouped_by_folder=grouped_by_folder,
                        output_path=output_path,
                        overwrite=overwrite,
                        detailed_logs=detailed_logs,
                        transfer_config=self.config,
                    )
                    if fail_fast and int(batch_stats.get('failed', 0) or 0) > 0:
                        batch_stopped_on_failure = True
                except Exception as batch_err:
                    self.progress_queue.put(
                        {
                            'type': 'log',
                            'level': 'E',
                            'message': f"[!!!] Batch pipeline execution failed. Error: {batch_err}",
                        }
                    )
                    batch_stopped_on_failure = True

        except Exception as batch_outer_err:
            self.progress_queue.put(
                {
                    'type': 'log',
                    'level': 'E',
                    'message': f"[!!!] Batch processing failed before completion. Error: {batch_outer_err}",
                }
            )
            batch_stopped_on_failure = True

        finally:
            if batch_pipeline is not None:
                try:
                    batch_pipeline.shutdown()
                except Exception as shutdown_err:
                    self.progress_queue.put(
                        {
                            'type': 'log',
                            'level': 'W',
                            'message': f"[!] Pipeline shutdown warning: {shutdown_err}",
                        }
                    )

            clear_torch_cache()

            # Release VRAM safely based on user's smart condition
            # "except for upscale only case (C off, D off, CT off)"
            is_upscale_only = self.config.upscale and not self.config.colorize and not self.config.denoise and not self.config.external_color_source_enabled
            if not is_upscale_only:
                # We defer to the main thread since free_vram interacts with UI state safely
                self.after(0, self.free_vram)

            if self.terminate_event.is_set():
                self.progress_queue.put({'type': 'log', 'level': 'W', 'message': "\n--- Processing terminated by user. ---"})
            elif batch_stopped_on_failure:
                self.progress_queue.put({'type': 'log', 'level': 'W', 'message': "\n--- Processing stopped after failure policy trigger. ---"})
            else:
                self.progress_queue.put({'type': 'log', 'level': 'D', 'message': "\n--- Batch processing complete! ---"})
                self.progress_queue.put({'type': 'progress', 'value': 100, 'eta': 'ETA: Done!'})

            self.processing_finished()

    def processing_finished(self):
        try:
            self.pause_button.config(state="disabled", text="Pause") # Reset button text
        except tk.TclError:
            pass
        self.processing_thread = None
        self._close_deadline_mono = None
        try:
            self._refresh_start_button_state()
            self.after(0, self._refresh_preview_runtime_mode)
        except tk.TclError:
            pass


    def save_settings(self, snapshot_active_preset=False):
        self._ensure_quality_preset_integrity()
        active_folder = self._quality_context_folder()
        active_preset_name = self._active_quality_preset_for_folder(active_folder)
        if snapshot_active_preset and active_preset_name in self.quality_presets and self._can_autosync_preset(active_preset_name):
            self.quality_presets[active_preset_name] = self._snapshot_quality_preset_from_vars()

        profile_name = self._normalized_pipeline_profile(self.pipeline_worker_profile.get())
        self.pipeline_worker_profile.set(profile_name)
        effective_instances = self._effective_pipeline_instances(device_name=getattr(self.config, 'device', 'cpu'))
        queue_budget_mb = self._sanitize_queue_budget_mb(getattr(self.config, 'pipeline_queue_ram_budget_mb', 0))
        try:
            raw_writer_threads = self.pipeline_writer_threads.get()
        except (tk.TclError, TypeError, ValueError):
            raw_writer_threads = getattr(self.config, 'pipeline_writer_threads', 1)
        writer_threads = self._sanitize_writer_threads(raw_writer_threads)
        self.pipeline_writer_threads.set(writer_threads)
        self.config.pipeline_writer_threads = writer_threads
        queue_budget_mode = 'auto' if queue_budget_mb <= 0 else 'manual'

        settings = {
            "input_folder": self.input_folder.get(),
            "output_folder": self.output_folder.get(),
            "enable_colorize": self.enable_colorize.get(),
            "enable_upscale": self.enable_upscale.get(),
            "enable_denoise": self.enable_denoise.get(),
            "overwrite_existing": self.overwrite_existing.get(),
            "_denoiser_weights_dir": os.path.abspath(
                getattr(self.config, '_denoiser_weights_dir', os.path.join(backend_path, 'denoising', 'models'))
            ),
            "colorizer_path": self.colorizer_path.get(),
            "upscaler_path": self.upscaler_path.get(),
            "upscaler_type": self.upscaler_type.get(),
            "upscaler_tile_size": self.upscaler_tile_size.get(),
            "colorizer_tile_size": self.colorizer_tile_size.get(),
            "tile_pad": self.tile_pad.get(),
            "lp_generator_patch_size": self.lp_generator_patch_size.get(),
            "lp_num_patches_h": self.lp_num_patches_h.get(),
            "lp_num_patches_w": self.lp_num_patches_w.get(),
            "colorized_image_size": self.colorized_image_size.get(),
            "input_image_size": sanitize_input_width_limit(self.input_image_size.get()),
            "force_safe_colorizer_width": self.force_safe_colorizer_width.get(),
            "detailed_debug_logs": self.detailed_debug_logs.get(),
            "export_ocr_debug": self.export_ocr_debug.get(),
            "max_ocr_dimension": self.max_ocr_dimension.get(),

            "use_yolo_bubbles": self.use_yolo_bubbles.get(),
            "fp_strictness": self.fp_strictness.get(),
            "sfx_feather_radius": self.sfx_feather_radius.get(),
            "yolo_model_type": self.yolo_model_type.get(),
            "log_show_error": self.log_show_error.get(),
            "log_show_warning": self.log_show_warning.get(),
            "log_show_debug": self.log_show_debug.get(),
            "log_show_trace": self.log_show_trace.get(),
            "log_show_verbose": self.log_show_verbose.get(),
            "colorize_precision_policy": normalize_precision_policy(self.colorize_precision_policy.get()),
            "upscale_precision_policy": normalize_precision_policy(self.upscale_precision_policy.get()),
            "precision_cast_weights": bool(self.precision_cast_weights.get()),
            "precision_fallback": normalize_fallback_mode(self.precision_fallback.get()),
            "precision_allow_tf32": bool(self.precision_allow_tf32.get()),
            "preview_fast_mode": bool(self.preview_fast_mode.get()),
            "external_color_source_enabled": bool(self.external_color_source_enabled.get()),
            "external_color_source_dir": self.external_color_source_dir.get(),
            "enable_sage_attention": self.enable_sage_attention.get(),
            "enable_torch_compile": self.torch_compile.get(),
            "enable_flash_attention": self.enable_flash_attention.get(),
            "enable_cpu_offload": self.enable_cpu_offload.get(),
            "ocr_model_tier": self.ocr_model_tier.get(),
            "denoise_max_side": self.denoise_max_side.get(),
            "pipeline_fail_fast": self.pipeline_fail_fast.get(),
            "pipeline_worker_profile": profile_name,
            "pipeline_manual_override": (profile_name == PIPELINE_PROFILE_CUSTOM),
            "pipeline_queue_ram_budget_mb": queue_budget_mb,
            "pipeline_queue_ram_budget_mode": queue_budget_mode,
            "pipeline_writer_threads": writer_threads,
            "pipeline_denoise_instances": effective_instances['denoise'],
            "pipeline_colorize_instances": effective_instances['colorize'],
            "pipeline_upscale_instances": effective_instances['upscale'],
            # Legacy keys kept for backward compatibility.
            "pipeline_workers_denoise": effective_instances['denoise'],
            "pipeline_workers_colorize": effective_instances['colorize'],
            "pipeline_workers_upscale": effective_instances['upscale'],
            "presets": {
                "version": 1,
                "active_global": self.active_quality_preset_global,
                "folder_overrides": dict(self.quality_preset_overrides),
                "items": self.quality_presets,
            },
        }
        try:
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=4)
        except Exception as e:
            self.log(f"[!] Could not save settings: {e}")

    def load_settings(self):
        rewrite_settings_on_load = False
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                    rewrite_settings_on_load = (
                        any(
                            legacy_key in settings
                            for legacy_key in (
                                "pipeline_instances_denoise",
                                "pipeline_instances_colorize",
                                "pipeline_instances_upscale",
                            )
                        )
                        or ("_denoiser_weights_dir" not in settings and "denoiser_weights_dir" in settings)
                        or (
                            bool(settings.get("pipeline_manual_override", False))
                            and str(settings.get("pipeline_worker_profile", "Balanced")).strip().title() != PIPELINE_PROFILE_CUSTOM
                        )
                    )
                    self.input_folder.set(settings.get("input_folder", ""))
                    self.output_folder.set(settings.get("output_folder", ""))
                    self.enable_colorize.set(settings.get("enable_colorize", True))
                    self.enable_upscale.set(settings.get("enable_upscale", True))
                    self.enable_denoise.set(settings.get("enable_denoise", True))
                    self.overwrite_existing.set(settings.get("overwrite_existing", False))

                    default_networks_path = os.path.join(backend_path, 'networks')
                    self.colorizer_path.set(settings.get("colorizer_path", os.path.join(default_networks_path, 'generator.zip')))
                    self.upscaler_path.set(settings.get("upscaler_path", os.path.join(default_networks_path, 'RealESRGAN_x4plus_anime_6B.pt')))
                    self.upscaler_type.set(settings.get("upscaler_type", 'Auto-Detect'))

                    self.upscaler_tile_size.set(settings.get("upscaler_tile_size", 256))
                    self.colorizer_tile_size.set(settings.get("colorizer_tile_size", 0))
                    self.tile_pad.set(settings.get("tile_pad", 8))
                    self.lp_generator_patch_size.set(settings.get("lp_generator_patch_size", 16))
                    self.lp_num_patches_h.set(settings.get("lp_num_patches_h", 3))
                    self.lp_num_patches_w.set(settings.get("lp_num_patches_w", 3))
                    self.colorized_image_size.set(settings.get("colorized_image_size", 576))
                    loaded_input_size = settings.get("input_image_size", DEFAULT_INPUT_WIDTH)
                    self.input_image_size.set(sanitize_input_width_limit(loaded_input_size))
                    self.force_safe_colorizer_width.set(settings.get("force_safe_colorizer_width", True))
                    self.detailed_debug_logs.set(settings.get("detailed_debug_logs", False))
                    self.export_ocr_debug.set(settings.get("export_ocr_debug", False))
                    self.max_ocr_dimension.set(settings.get("max_ocr_dimension", 3072))

                    yolo_val = settings.get("use_yolo_bubbles", "full")
                    if yolo_val is True:
                        yolo_val = 'full'
                    elif yolo_val is False:
                        yolo_val = 'off'
                    self.use_yolo_bubbles.set(yolo_val)
                    self.fp_strictness.set(settings.get("fp_strictness", 0.5))
                    self.sfx_feather_radius.set(settings.get("sfx_feather_radius", 0))
                    self.yolo_model_type.set(settings.get("yolo_model_type", "seg"))
                    self.log_show_error.set(settings.get("log_show_error", True))
                    self.log_show_warning.set(settings.get("log_show_warning", True))
                    self.log_show_debug.set(settings.get("log_show_debug", True))
                    self.log_show_trace.set(settings.get("log_show_trace", False))
                    self.log_show_verbose.set(settings.get("log_show_verbose", False))
                    # Fallback to legacy 'precision_policy' if new specific policies are missing
                    self.colorize_precision_policy.set(
                        normalize_precision_policy(settings.get("colorize_precision_policy", settings.get("precision_policy", getattr(self.config, 'colorize_precision_policy', 'auto'))))
                    )
                    self.upscale_precision_policy.set(
                        normalize_precision_policy(settings.get("upscale_precision_policy", settings.get("precision_policy", getattr(self.config, 'upscale_precision_policy', 'auto'))))
                    )
                    self.precision_cast_weights.set(bool(settings.get("precision_cast_weights", True)))
                    self.precision_fallback.set(
                        normalize_fallback_mode(settings.get("precision_fallback", getattr(self.config, 'precision_fallback', 'per_image')))
                    )
                    self.precision_allow_tf32.set(bool(settings.get("precision_allow_tf32", True)))
                    self.preview_fast_mode.set(bool(settings.get("preview_fast_mode", False)))
                    self.external_color_source_enabled.set(bool(settings.get("external_color_source_enabled", False)))
                    self.external_color_source_dir.set(settings.get("external_color_source_dir", ""))
                    self.enable_sage_attention.set(settings.get("enable_sage_attention", False))
                    self.torch_compile.set(settings.get("enable_torch_compile", False))
                    self.enable_flash_attention.set(settings.get("enable_flash_attention", False))
                    self.enable_cpu_offload.set(settings.get("enable_cpu_offload", False))
                    self.ocr_model_tier.set(settings.get("ocr_model_tier", "server"))
                    self.denoise_max_side.set(settings.get("denoise_max_side", 0))
                    loaded_weights_dir = settings.get(
                        "_denoiser_weights_dir",
                        settings.get("denoiser_weights_dir", os.path.join(backend_path, 'denoising', 'models')),
                    )
                    self.config._denoiser_weights_dir = os.path.abspath(loaded_weights_dir)
                    self.config.denoiser_weights_dir = self.config._denoiser_weights_dir
                    raw_queue_budget = settings.get("pipeline_queue_ram_budget_mb", getattr(self.config, 'pipeline_queue_ram_budget_mb', 0))
                    queue_budget_mode = str(settings.get("pipeline_queue_ram_budget_mode", "")).strip().lower()
                    parsed_queue_budget = self._sanitize_queue_budget_mb(raw_queue_budget)
                    if queue_budget_mode == 'auto':
                        parsed_queue_budget = 0
                    elif queue_budget_mode == 'manual':
                        parsed_queue_budget = max(64, parsed_queue_budget) if parsed_queue_budget > 0 else 64
                    else:
                        # Backward-compat: treat prior default 384 as "auto" unless user explicitly set a mode.
                        if int(parsed_queue_budget) == 384:
                            parsed_queue_budget = 0
                            rewrite_settings_on_load = True

                    self.config.pipeline_queue_ram_budget_mb = parsed_queue_budget
                    loaded_writer_threads = self._sanitize_writer_threads(
                        settings.get("pipeline_writer_threads", getattr(self.config, 'pipeline_writer_threads', 1))
                    )
                    self.config.pipeline_writer_threads = loaded_writer_threads
                    self.pipeline_writer_threads.set(loaded_writer_threads)
                    self.pipeline_fail_fast.set(settings.get("pipeline_fail_fast", True))
                    loaded_profile = settings.get("pipeline_worker_profile", "Balanced")
                    loaded_manual_override = bool(settings.get("pipeline_manual_override", False))
                    normalized_profile = self._normalized_pipeline_profile(loaded_profile)
                    if loaded_manual_override and normalized_profile in PIPELINE_PROFILE_VALUES:
                        normalized_profile = PIPELINE_PROFILE_CUSTOM
                    self.pipeline_worker_profile.set(normalized_profile)
                    self.pipeline_denoise_instances.set(
                        settings.get(
                            "pipeline_denoise_instances",
                            settings.get(
                                "pipeline_instances_denoise",
                                settings.get("pipeline_workers_denoise", PIPELINE_PROFILE_VALUES['Balanced']['denoise_inst'])
                            )
                        )
                    )
                    self.pipeline_colorize_instances.set(
                        settings.get(
                            "pipeline_colorize_instances",
                            settings.get(
                                "pipeline_instances_colorize",
                                settings.get("pipeline_workers_colorize", PIPELINE_PROFILE_VALUES['Balanced']['colorize_inst'])
                            )
                        )
                    )
                    self.pipeline_upscale_instances.set(
                        settings.get(
                            "pipeline_upscale_instances",
                            settings.get(
                                "pipeline_instances_upscale",
                                settings.get("pipeline_workers_upscale", PIPELINE_PROFILE_VALUES['Balanced']['upscale_inst'])
                            )
                        )
                    )

                    presets_blob = settings.get("presets")
                    if isinstance(presets_blob, dict) and presets_blob:
                        raw_items = presets_blob.get("items")
                        if not isinstance(raw_items, dict):
                            raw_items = presets_blob.get("presets", {})

                        self.quality_presets = raw_items if isinstance(raw_items, dict) else {
                            'Default': dict(QUALITY_PRESET_DEFAULTS)
                        }
                        self.active_quality_preset_global = presets_blob.get(
                            "active_global",
                            settings.get("active_quality_preset_global", "Default"),
                        )
                        self.quality_preset_overrides = presets_blob.get(
                            "folder_overrides",
                            settings.get("quality_preset_overrides", {}),
                        )
                    else:
                        legacy_default_payload = self._coerce_quality_preset_payload({
                            "denoise_sigma": settings.get("denoise_sigma", QUALITY_PRESET_DEFAULTS['denoise_sigma']),
                            "chroma_resize_mode": settings.get("chroma_resize_mode", QUALITY_PRESET_DEFAULTS['chroma_resize_mode']),
                            "edge_chroma_protection": settings.get("edge_chroma_protection", QUALITY_PRESET_DEFAULTS['edge_chroma_protection']),
                            "edge_chroma_strength": settings.get("edge_chroma_strength", QUALITY_PRESET_DEFAULTS['edge_chroma_strength']),
                            "line_ink_protection": settings.get("line_ink_protection", settings.get("ink_protection", QUALITY_PRESET_DEFAULTS['line_ink_protection'])),
                            "line_ink_protection_strength": settings.get("line_ink_protection_strength", settings.get("ink_protection_strength", QUALITY_PRESET_DEFAULTS['line_ink_protection_strength'])),
                            "screentone_chroma_smoothing": settings.get("screentone_chroma_smoothing", QUALITY_PRESET_DEFAULTS['screentone_chroma_smoothing']),
                            "screentone_smoothing_strength": settings.get("screentone_smoothing_strength", QUALITY_PRESET_DEFAULTS['screentone_smoothing_strength']),
                        })

                        legacy_items = settings.get("quality_presets", {})
                        if isinstance(legacy_items, dict) and legacy_items:
                            self.quality_presets = dict(legacy_items)
                            self.quality_presets.setdefault('Default', legacy_default_payload)
                        else:
                            self.quality_presets = {
                                'Default': legacy_default_payload
                            }

                        self.active_quality_preset_global = settings.get("active_quality_preset_global", "Default")
                        self.quality_preset_overrides = settings.get("quality_preset_overrides", {})

            self._ensure_quality_preset_integrity()
            active_profile = self._normalized_pipeline_profile(self.pipeline_worker_profile.get())
            self.pipeline_worker_profile.set(active_profile)
            self._apply_pipeline_profile_if_needed(force=(active_profile in PIPELINE_PROFILE_VALUES))

            effective_instances = self._effective_pipeline_instances(device_name=getattr(self.config, 'device', 'cpu'))
            self.config.pipeline_denoise_instances = effective_instances['denoise']
            self.config.pipeline_colorize_instances = effective_instances['colorize']
            self.config.pipeline_upscale_instances = effective_instances['upscale']
            self.config.pipeline_worker_profile = active_profile
            self.config.pipeline_manual_override = (active_profile == PIPELINE_PROFILE_CUSTOM)
            self.config.pipeline_instances_denoise = self.config.pipeline_denoise_instances
            self.config.pipeline_instances_colorize = self.config.pipeline_colorize_instances
            self.config.pipeline_instances_upscale = self.config.pipeline_upscale_instances
            self.config.pipeline_workers_denoise = self.config.pipeline_denoise_instances
            self.config.pipeline_workers_colorize = self.config.pipeline_colorize_instances
            self.config.pipeline_workers_upscale = self.config.pipeline_upscale_instances
            try:
                raw_writer_threads = self.pipeline_writer_threads.get()
            except (tk.TclError, TypeError, ValueError):
                raw_writer_threads = getattr(self.config, 'pipeline_writer_threads', 1)
            self.config.pipeline_writer_threads = self._sanitize_writer_threads(raw_writer_threads)
            self.pipeline_writer_threads.set(self.config.pipeline_writer_threads)

            active_folder = self._quality_context_folder()
            active_preset = self._active_quality_preset_for_folder(active_folder)
            if active_preset in self.quality_presets:
                self._apply_quality_preset_payload_to_vars(self.quality_presets[active_preset])

            if rewrite_settings_on_load:
                self.save_settings(snapshot_active_preset=False)

            self._refresh_live_vram_meter(force=True)
            self._sync_external_color_state()

        except Exception as e:
            self.log(f"[!] Could not load settings: {e}")
            self.quality_presets = {'Default': dict(QUALITY_PRESET_DEFAULTS)}
            self.active_quality_preset_global = 'Default'
            self.quality_preset_overrides = {}
            self._ensure_quality_preset_integrity()

    def on_closing(self):
        self.terminate_event.set() # Signal thread to stop
        self.pause_event.clear() # Ensure paused workers can observe terminate.
        self.save_settings(snapshot_active_preset=False)
        if self.processing_thread and self.processing_thread.is_alive():
             if self._close_deadline_mono is None:
                 self._close_deadline_mono = time.monotonic() + APP_CLOSE_WAIT_TIMEOUT_S
                 self.log("[!] Waiting for processing thread to stop before closing...")
             self.after(100, self.check_thread_and_close)
        else:
             self._close_deadline_mono = None
             self.destroy()

    def check_thread_and_close(self):
        if self.processing_thread and self.processing_thread.is_alive():
            if (
                self._close_deadline_mono is not None
                and time.monotonic() >= float(self._close_deadline_mono)
            ):
                self.log("[!] Close timeout reached; forcing window shutdown.")
                self.processing_thread = None
                self._close_deadline_mono = None
                try:
                    self.destroy()
                except Exception:
                    pass
                return
            self.after(100, self.check_thread_and_close)
        else:
            self._close_deadline_mono = None
            self.destroy()


