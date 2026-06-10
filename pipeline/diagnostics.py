import ctypes
import math
import os
import queue
import torch
import numpy as np
import PIL.Image


VRAM_PER_DENOISER_MB = 25
VRAM_PER_COLORIZER_MB = 325
VRAM_PER_UPSCALER_MB = 650
VRAM_OVERHEAD_MB = 2048

INSTANCE_CAPS = {
    'denoise': 32,
    'colorize': 16,
    'upscale': 8,
}

THREADS_PER_INSTANCE = {
    'denoise': 3,
    'colorize': 3,
    'upscale': 3,
}

PIPELINE_PROFILE_VALUES = {
    'Safe': {'denoise_inst': 2, 'colorize_inst': 1, 'upscale_inst': 1},
    'Balanced': {'denoise_inst': 4, 'colorize_inst': 2, 'upscale_inst': 1},
    'Performance': {'denoise_inst': 8, 'colorize_inst': 4, 'upscale_inst': 2},
    'Aggressive': {'denoise_inst': 16, 'colorize_inst': 8, 'upscale_inst': 2},
}

MAX_INPUT_WIDTH = 1792
MIN_INPUT_WIDTH = 256
DEFAULT_INPUT_WIDTH = 800
QUEUE_MEMORY_BUDGET_MB_DEFAULT = 384
QUEUE_MEMORY_BUDGET_MB_MIN = 128
QUEUE_MEMORY_BUDGET_MB_MAX = 1536
QUEUE_MEMORY_BUDGET_AUTO_FRACTION = 0.18
QUEUE_MEMORY_BUDGET_AUTO_HEADROOM_MB = 1024
QUEUE_HARD_SAFETY_CAP = 48
QUEUE_MAX_ITEMS_CAP = QUEUE_HARD_SAFETY_CAP
QUEUE_MIN_ITEMS = 1
QUEUE_RAM_HEADROOM_MB = 2048
QUEUE_RAM_PRESSURE_WARN_MB = 1024
QUEUE_DIAG_WAIT_WARN_S = 2.0
QUEUE_DIAG_WAIT_REPORT_INTERVAL_S = 5.0
QUEUE_DIAG_JOIN_REPORT_INTERVAL_S = 10.0
QUEUE_DIAG_HEARTBEAT_INTERVAL_S = 10.0
QUEUE_DIAG_WRITER_REPORT_INTERVAL_S = 10.0
QUEUE_DIAG_JOIN_ABORT_ON_STOP_S = 5.0
QUEUE_DIAG_SLOW_STAGE_S = 2.0


class LogLevel:
    ERROR = 'E'
    WARNING = 'W'
    DEBUG = 'D'
    TRACE = 'T'
    VERBOSE = 'V'


def _bytes_to_mb(value):
    return int(value // (1024 * 1024))


def _queue_qsize(q):
    try:
        return int(q.qsize())
    except Exception:
        return -1


def _queue_maxsize(q):
    try:
        return int(getattr(q, 'maxsize', 0) or 0)
    except Exception:
        return 0


def _queue_state_text(q):
    size = _queue_qsize(q)
    cap = _queue_maxsize(q)
    if size < 0:
        return '?'
    if cap > 0:
        return f'{size}/{cap}'
    return str(size)


def get_default_instance_vram_profile_mb():
    return {
        'denoise': VRAM_PER_DENOISER_MB,
        'colorize': VRAM_PER_COLORIZER_MB,
        'upscale': VRAM_PER_UPSCALER_MB,
    }


def _normalized_instance_profile_mb(instance_profile_mb=None):
    profile = get_default_instance_vram_profile_mb()
    if isinstance(instance_profile_mb, dict):
        for stage in ('denoise', 'colorize', 'upscale'):
            if stage in instance_profile_mb:
                try:
                    profile[stage] = max(1, int(instance_profile_mb[stage]))
                except (TypeError, ValueError):
                    continue
    return profile


def estimate_vram_breakdown_mb(denoise_inst, colorize_inst, upscale_inst, instance_profile_mb=None, overhead_mb=0):
    profile = _normalized_instance_profile_mb(instance_profile_mb)

    pass1_stage_mb = (
        int(denoise_inst) * profile['denoise']
        + int(colorize_inst) * profile['colorize']
    )
    pass2_stage_mb = int(upscale_inst) * profile['upscale']

    overhead = max(0, int(overhead_mb))
    pass1_total_mb = pass1_stage_mb + overhead if pass1_stage_mb > 0 else 0
    pass2_total_mb = pass2_stage_mb + overhead if pass2_stage_mb > 0 else 0

    if pass1_total_mb <= 0 and pass2_total_mb <= 0:
        peak_pass = 'none'
        peak_total_mb = 0
    elif pass1_total_mb >= pass2_total_mb:
        peak_pass = 'pass1'
        peak_total_mb = pass1_total_mb
    else:
        peak_pass = 'pass2'
        peak_total_mb = pass2_total_mb

    return {
        'pass1_stage_mb': int(pass1_stage_mb),
        'pass2_stage_mb': int(pass2_stage_mb),
        'pass1_total_mb': int(pass1_total_mb),
        'pass2_total_mb': int(pass2_total_mb),
        'peak_pass': peak_pass,
        'peak_total_mb': int(peak_total_mb),
    }


def estimate_vram_mb(denoise_inst, colorize_inst, upscale_inst, instance_profile_mb=None, overhead_mb=0):
    breakdown = estimate_vram_breakdown_mb(
        denoise_inst,
        colorize_inst,
        upscale_inst,
        instance_profile_mb=instance_profile_mb,
        overhead_mb=overhead_mb,
    )
    return int(breakdown['peak_total_mb'])

def _get_system_used_now():
    try:
        if torch.cuda.is_available():
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info()
            except Exception:
                free_bytes, total_bytes = torch.cuda.mem_get_info(0)
            return int((total_bytes - free_bytes) // (1024 * 1024))
    except Exception:
        pass
    return None

_baseline_system_used_mb = _get_system_used_now()

def get_cuda_memory_stats_mb(device_index=0):
    global _baseline_system_used_mb
    try:
        if not torch.cuda.is_available():
            return {}

        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
        except TypeError:
            free_bytes, total_bytes = torch.cuda.mem_get_info()
        try:
            allocated_bytes = torch.cuda.memory_allocated(device_index)
            reserved_bytes = torch.cuda.memory_reserved(device_index)
        except TypeError:
            allocated_bytes = torch.cuda.memory_allocated()
            reserved_bytes = torch.cuda.memory_reserved()

        total_mb = _bytes_to_mb(total_bytes)
        free_mb = _bytes_to_mb(free_bytes)
        system_used_mb = max(0, total_mb - free_mb)
        
        # Initialize baseline on first call
        if _baseline_system_used_mb is None:
            _baseline_system_used_mb = system_used_mb
            
        process_allocated_mb = _bytes_to_mb(allocated_bytes)
        process_reserved_mb = _bytes_to_mb(reserved_bytes)
        
        # Estimate total VRAM used by OUR app (including PaddleOCR, ONNX, etc)
        # It's at least PyTorch's reserved memory, plus any system usage increase since startup
        app_used_mb = max(process_reserved_mb, system_used_mb - _baseline_system_used_mb)
        
        # Available for process is whatever is currently free PLUS what we already occupy
        available_for_process_mb = max(0, free_mb + app_used_mb)

        try:
            max_allocated_bytes = torch.cuda.max_memory_allocated(device_index)
        except Exception:
            max_allocated_bytes = allocated_bytes

        return {
            'device_index': int(device_index),
            'total_mb': total_mb,
            'free_mb': free_mb,
            'used_mb': system_used_mb,
            'process_allocated_mb': process_allocated_mb,
            'process_reserved_mb': process_reserved_mb,
            'app_used_mb': app_used_mb,
            'max_allocated_mb': _bytes_to_mb(max_allocated_bytes),
            'available_for_process_mb': available_for_process_mb,
        }
    except Exception:
        return {}


def get_gpu_vram_mb():
    stats = get_cuda_memory_stats_mb()
    if stats:
        return int(stats.get('total_mb', 0))
    return 0


def get_system_memory_stats_mb():
    try:
        if os.name == 'nt':
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ('dwLength', ctypes.c_ulong),
                    ('dwMemoryLoad', ctypes.c_ulong),
                    ('ullTotalPhys', ctypes.c_ulonglong),
                    ('ullAvailPhys', ctypes.c_ulonglong),
                    ('ullTotalPageFile', ctypes.c_ulonglong),
                    ('ullAvailPageFile', ctypes.c_ulonglong),
                    ('ullTotalVirtual', ctypes.c_ulonglong),
                    ('ullAvailVirtual', ctypes.c_ulonglong),
                    ('ullAvailExtendedVirtual', ctypes.c_ulonglong),
                ]

            memory_status = MEMORYSTATUSEX()
            memory_status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory_status)):
                return {}

            total_mb = _bytes_to_mb(memory_status.ullTotalPhys)
            available_mb = _bytes_to_mb(memory_status.ullAvailPhys)
            return {
                'total_mb': int(total_mb),
                'available_mb': int(available_mb),
                'used_mb': int(max(0, total_mb - available_mb)),
            }

        if hasattr(os, 'sysconf'):
            page_size = int(os.sysconf('SC_PAGE_SIZE'))
            total_pages = int(os.sysconf('SC_PHYS_PAGES'))
            available_pages = int(os.sysconf('SC_AVPHYS_PAGES'))
            total_mb = _bytes_to_mb(page_size * total_pages)
            available_mb = _bytes_to_mb(page_size * available_pages)
            return {
                'total_mb': int(total_mb),
                'available_mb': int(available_mb),
                'used_mb': int(max(0, total_mb - available_mb)),
            }
    except Exception:
        return {}

    return {}


def vram_warning_text(denoise_inst, colorize_inst, upscale_inst, instance_profile_mb=None, overhead_mb=0):
    breakdown = estimate_vram_breakdown_mb(
        denoise_inst,
        colorize_inst,
        upscale_inst,
        instance_profile_mb=instance_profile_mb,
        overhead_mb=overhead_mb,
    )
    est = int(breakdown['peak_total_mb'])
    pass1_total_mb = int(breakdown['pass1_total_mb'])
    pass2_total_mb = int(breakdown['pass2_total_mb'])

    denoise_active = int(denoise_inst) > 0
    colorize_active = int(colorize_inst) > 0
    upscale_active = int(upscale_inst) > 0

    pass1_short_parts = []
    pass1_long_parts = []
    if denoise_active:
        pass1_short_parts.append('D')
        pass1_long_parts.append('Denoise')
    if colorize_active:
        pass1_short_parts.append('C')
        pass1_long_parts.append('Colorize')

    if pass1_long_parts:
        pass1_peak_label = f"Pass 1 ({' + '.join(pass1_long_parts)})"
        pass1_breakdown_label = f"Pass1({'+'.join(pass1_short_parts)})"
    else:
        pass1_peak_label = 'Pass 1 (inactive)'
        pass1_breakdown_label = 'Pass1(inactive)'

    if upscale_active:
        pass2_peak_label = 'Pass 2 (Upscale)'
        pass2_breakdown_label = 'Pass2(U)'
    else:
        pass2_peak_label = 'Pass 2 (inactive)'
        pass2_breakdown_label = 'Pass2(inactive)'

    if est <= 0:
        return ''

    peak_pass = str(breakdown.get('peak_pass', 'none')).strip().lower()
    peak_label = {
        'pass1': pass1_peak_label,
        'pass2': pass2_peak_label,
    }.get(peak_pass, 'No active pass')

    memory_stats = get_cuda_memory_stats_mb()
    if memory_stats:
        available = int(memory_stats.get('available_for_process_mb', 0))
        gpu_vram = int(memory_stats.get('total_mb', 0))
    else:
        gpu_vram = get_gpu_vram_mb()
        if gpu_vram == 0:
            return ''
        available = max(0, gpu_vram - VRAM_OVERHEAD_MB)

    if est > available:
        return (
            f"Est. peak pipeline VRAM: ~{est} MB "
            f"({peak_label}; {pass1_breakdown_label}={pass1_total_mb} MB, {pass2_breakdown_label}={pass2_total_mb} MB). "
            f"(GPU has {gpu_vram} MB, ~{available} MB usable). Risk of OOM errors."
        )

    if est > available * 0.8:
        return (
            f"Pipeline peak VRAM: ~{est}/{available} MB available ({peak_label}; "
            f"{pass1_breakdown_label}={pass1_total_mb} MB, {pass2_breakdown_label}={pass2_total_mb} MB) - near limit."
        )

    return ''


def calculate_max_instances_from_vram(
    gpu_vram_mb,
    stage,
    other_stages_mb=0,
    per_instance_mb=None,
    overhead_mb=None,
):
    if gpu_vram_mb <= 0:
        return 1

    defaults = get_default_instance_vram_profile_mb()
    inst_cost = int(per_instance_mb or defaults.get(stage, VRAM_PER_UPSCALER_MB))
    overhead = int(overhead_mb if overhead_mb is not None else VRAM_OVERHEAD_MB)

    available = int(gpu_vram_mb) - overhead - int(other_stages_mb)
    if available <= 0:
        return 1

    safe_available = int(available * 0.85)
    max_inst = max(1, safe_available // max(1, inst_cost))
    return min(max_inst, int(INSTANCE_CAPS.get(stage, 4)))


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

