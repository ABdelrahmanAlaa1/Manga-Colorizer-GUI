import copy
import gc
import math
import os
import queue
import threading
import time
from collections import OrderedDict

import numpy as np
import PIL.Image
import torch

from .diagnostics import (
    LogLevel,
    QUEUE_MAX_ITEMS_CAP,
    QUEUE_MIN_ITEMS,
    QUEUE_RAM_HEADROOM_MB,
    QUEUE_RAM_PRESSURE_WARN_MB,
    QUEUE_DIAG_SLOW_STAGE_S,
    QUEUE_DIAG_WRITER_REPORT_INTERVAL_S,
    QUEUE_DIAG_WAIT_WARN_S,
    QUEUE_DIAG_WAIT_REPORT_INTERVAL_S,
    QUEUE_DIAG_HEARTBEAT_INTERVAL_S,
    QUEUE_DIAG_JOIN_REPORT_INTERVAL_S,
    QUEUE_DIAG_JOIN_ABORT_ON_STOP_S,
    DEFAULT_INPUT_WIDTH,
    THREADS_PER_INSTANCE,
    get_cuda_memory_stats_mb,
    get_gpu_vram_mb,
    get_system_memory_stats_mb,
    estimate_vram_breakdown_mb,
    sanitize_input_width_limit,
    clamp_image_to_input_width,
    get_default_instance_vram_profile_mb,
    _queue_state_text,
    _bytes_to_mb,
)
from .stages import ModelInstancePool, _PipelineAborted
from .transfer import build_external_color_map
from .workers import _persistent_producer, _persistent_stage_worker, _persistent_writer


class ProcessingPipeline:
    _persistent_producer = _persistent_producer
    _persistent_stage_worker = _persistent_stage_worker
    _persistent_writer = _persistent_writer

    def __init__(
        self,
        config,
        model_factories,
        instance_counts,
        progress_callback,
        terminate_event,
        pause_event,
        warmup_spec=None,
        fail_fast=True,
    ):
        self.config = config
        self.model_factories = dict(model_factories or {})
        self.warmup_spec = dict(warmup_spec or {})
        self.instance_counts = {
            'denoise': max(0, int(instance_counts.get('denoise', 0))),
            'colorize': max(0, int(instance_counts.get('colorize', 0))),
            'upscale': max(0, int(instance_counts.get('upscale', 0))),
        }
        self.terminate_event = terminate_event
        self.pause_event = pause_event
        self.fail_fast = bool(fail_fast)
        self.progress_callback = progress_callback
        try:
            self.writer_threads = max(1, int(getattr(self.config, 'pipeline_writer_threads', 1) or 1))
        except (TypeError, ValueError):
            self.writer_threads = 1

        self.pools = {}

        self.instance_vram_observed_mb = {}
        self.stage_peak_inference_mb = {}
        self.runtime_overhead_observed_mb = 0
        self.cuda_baseline_stats = get_cuda_memory_stats_mb()
        self.cuda_after_pool_stats = {}
        self.cuda_after_warmup_stats = {}
        self.adaptive_upscale_limit = {
            'applied': False,
            'from_instances': 0,
            'to_instances': 0,
            'from_workers': 0,
            'to_workers': 0,
            'from': 0,
            'to': 0,
            'projected_peak_mb': 0,
            'available_vram_mb': 0,
            'burst_window_mb': 0,
            'headroom_mb': 0,
            'overhead_mb': 0,
            'shared_peak_mb': 0,
            'trigger': '',
            'reason': '',
        }

        self._external_color_map = {}

        self.worker_counts = {}
        for stage in ('denoise', 'colorize', 'upscale'):
            inst = int(self.instance_counts.get(stage, 0))
            tpi = int(THREADS_PER_INSTANCE.get(stage, 1))
            self.worker_counts[stage] = max(1, inst * tpi) if inst > 0 else 0

    def get_shared_stage_peak_mb(self):
        return {
            stage: max(0, int(self.stage_peak_inference_mb.get(stage, 0) or 0))
            for stage in ('denoise', 'colorize', 'upscale')
        }

    def _emit(self, payload, level=LogLevel.DEBUG):
        try:
            out_payload = payload
            if isinstance(payload, dict):
                out_payload = dict(payload)
                if out_payload.get('type') in ('log', 'log_update') and 'level' not in out_payload:
                    out_payload['level'] = str(level or LogLevel.DEBUG)
            self.progress_callback(out_payload)
        except Exception:
            pass

    def _stop_state_text(self, pass_stop_event=None):
        pass_stop_set = bool(pass_stop_event.is_set()) if isinstance(pass_stop_event, threading.Event) else False
        return f"terminate={int(self.terminate_event.is_set())}, pass_stop={int(pass_stop_set)}"

    def _is_stop_requested(self, pass_stop_event=None):
        pass_stop_set = bool(pass_stop_event.is_set()) if isinstance(pass_stop_event, threading.Event) else False
        return bool(self.terminate_event.is_set() or pass_stop_set)

    def _wait_while_paused(self, pass_stop_event=None, sleep_s=0.2):
        wait_step = max(0.01, float(sleep_s))
        while self.pause_event.is_set():
            if self._is_stop_requested(pass_stop_event):
                return False
            time.sleep(wait_step)
        return not self._is_stop_requested(pass_stop_event)

    def _make_safe_put_diag(self, detailed_logs, pass_stop_event, owner, queue_name, item_ref=None):
        if not detailed_logs:
            return None

        state_ref = item_ref if isinstance(item_ref, dict) else {}

        def _item_label():
            value = str(state_ref.get('item', '')).strip()
            return value if value else '?'

        def _on_wait(wait_seconds, queue_state):
            self._emit(
                {
                    'type': 'log',
                    'message': (
                        '[*] Queue wait => '
                        f'{owner}->{queue_name} '
                        f'item:{_item_label()} '
                        f'wait:{wait_seconds:.1f}s '
                        f'q:{queue_state} '
                        f'({self._stop_state_text(pass_stop_event)})'
                    ),
                },
                level=LogLevel.DEBUG,
            )

        def _on_abort(wait_seconds, queue_state):
            self._emit(
                {
                    'type': 'log',
                    'message': (
                        '[!] Queue enqueue aborted => '
                        f'{owner}->{queue_name} '
                        f'item:{_item_label()} '
                        f'wait:{wait_seconds:.1f}s '
                        f'q:{queue_state} '
                        f'({self._stop_state_text(pass_stop_event)})'
                    ),
                },
                level=LogLevel.WARNING,
            )

        def _on_success(wait_seconds, queue_state):
            if wait_seconds < QUEUE_DIAG_WAIT_WARN_S:
                return
            self._emit(
                {
                    'type': 'log',
                    'message': (
                        '[*] Queue wait cleared => '
                        f'{owner}->{queue_name} '
                        f'item:{_item_label()} '
                        f'wait:{wait_seconds:.1f}s '
                        f'q:{queue_state}'
                    ),
                },
                level=LogLevel.DEBUG,
            )

        return {
            'enabled': True,
            'warn_after_s': float(QUEUE_DIAG_WAIT_WARN_S),
            'report_interval_s': float(QUEUE_DIAG_WAIT_REPORT_INTERVAL_S),
            'on_wait': _on_wait,
            'on_abort': _on_abort,
            'on_success': _on_success,
        }

    @staticmethod
    def _latency_summary_ms(samples):
        values = [float(v) for v in (samples or []) if v is not None]
        if not values:
            return 0.0, 0.0, 0.0

        values.sort()
        count = len(values)
        avg_ms = float(sum(values) / count)
        p95_idx = max(0, min(count - 1, int(math.ceil(count * 0.95)) - 1))
        p95_ms = float(values[p95_idx])
        max_ms = float(values[-1])
        return avg_ms, p95_ms, max_ms

    def _emit_pass_state(self, state, message=''):
        self._emit(
            {
                'type': 'pass_state',
                'state': str(state),
                'message': str(message or ''),
            },
            level=LogLevel.DEBUG,
        )

    def _emit_terminal(self, status, message=''):
        self._emit(
            {
                'type': 'terminal',
                'status': str(status),
                'message': str(message or ''),
            },
            level=LogLevel.DEBUG,
        )

    def _get_warmup_input_hw(self):
        warmup_hw = self.warmup_spec.get('input_hw') if isinstance(self.warmup_spec, dict) else None
        if isinstance(warmup_hw, (list, tuple)) and len(warmup_hw) == 2:
            try:
                warmup_h = max(1, int(warmup_hw[0]))
                warmup_w = max(1, int(warmup_hw[1]))
                return warmup_h, warmup_w
            except (TypeError, ValueError):
                pass
        return 1024, 576

    def _apply_adaptive_upscale_worker_limit(self):
        current_instances = int(self.instance_counts.get('upscale', 0))
        current_workers = int(self.worker_counts.get('upscale', 0))
        self.adaptive_upscale_limit = {
            'applied': False,
            'from_instances': current_instances,
            'to_instances': current_instances,
            'from_workers': current_workers,
            'to_workers': current_workers,
            'from': current_workers,
            'to': current_workers,
            'projected_peak_mb': 0,
            'available_vram_mb': 0,
            'burst_window_mb': 0,
            'headroom_mb': 0,
            'overhead_mb': 0,
            'shared_peak_mb': 0,
            'trigger': '',
            'reason': '',
        }

        if 'upscale' not in self.pools or current_instances <= 0 or current_workers <= 0:
            return

        memory_stats = get_cuda_memory_stats_mb()
        if not memory_stats:
            self.adaptive_upscale_limit['reason'] = 'CUDA VRAM telemetry unavailable'
            return

        available_for_process = int(memory_stats.get('available_for_process_mb', 0) or 0)
        if available_for_process <= 0:
            available_for_process = int(memory_stats.get('free_mb', 0) or 0)
        if available_for_process <= 0:
            self.adaptive_upscale_limit['reason'] = 'CUDA VRAM telemetry unavailable'
            return

        profile = self.get_instance_vram_profile_mb()
        per_instance_mb = max(1, int(profile.get('upscale', 0) or 0))
        shared_stage_peak = self.get_shared_stage_peak_mb()
        shared_upscale_peak_mb = int(shared_stage_peak.get('upscale', 0) or 0)

        allocated_now = int(memory_stats.get('process_allocated_mb', 0) or 0)
        inst_total_mb = 0
        for stage in ('denoise', 'colorize', 'upscale'):
            inst_total_mb += int(self.instance_counts.get(stage, 0)) * int(profile.get(stage, 0))
        overhead_mb = max(0, int(allocated_now - inst_total_mb))
        self.runtime_overhead_observed_mb = int(overhead_mb)

        projected_breakdown = estimate_vram_breakdown_mb(
            0,
            0,
            self.instance_counts.get('upscale', 0),
            instance_profile_mb=profile,
            overhead_mb=overhead_mb,
        )
        base_projected_mb = int(projected_breakdown.get('pass2_total_mb', 0) or 0)
        projected_peak_mb = int(base_projected_mb + shared_upscale_peak_mb)

        baseline_stats = self.cuda_baseline_stats or {}
        after_pool_stats = self.cuda_after_pool_stats or {}
        after_warmup_stats = self.cuda_after_warmup_stats or {}
        baseline_reserved = int(baseline_stats.get('process_reserved_mb', 0) or 0)
        after_pool_reserved = int(after_pool_stats.get('process_reserved_mb', baseline_reserved) or baseline_reserved)
        after_warmup_reserved = int(after_warmup_stats.get('process_reserved_mb', after_pool_reserved) or after_pool_reserved)

        reserved_growth_pool = max(0, after_pool_reserved - baseline_reserved)
        reserved_growth_warmup = max(0, after_warmup_reserved - baseline_reserved)

        cache_stats = after_warmup_stats or after_pool_stats or memory_stats or {}
        cache_reserved = int(cache_stats.get('process_reserved_mb', 0) or 0)
        cache_allocated = int(cache_stats.get('process_allocated_mb', 0) or 0)
        cache_spread = max(0, cache_reserved - cache_allocated)

        burst_window_mb = max(
            reserved_growth_pool,
            reserved_growth_warmup,
            cache_spread,
            shared_upscale_peak_mb,
        )

        headroom_mb = int(available_for_process - projected_peak_mb)
        oversubscribed = projected_peak_mb > int(available_for_process * 0.95)
        near_limit = projected_peak_mb > 0 and headroom_mb < int(burst_window_mb * 0.5)

        trigger = ''
        if oversubscribed:
            trigger = 'oversubscription'
        elif near_limit:
            trigger = 'near-limit'

        reason_parts = []
        if oversubscribed:
            reason_parts.append(
                f'projected pass2 peak {projected_peak_mb} MB > available {available_for_process} MB'
            )
        if near_limit:
            reason_parts.append(f'headroom {headroom_mb} MB < burst window {burst_window_mb} MB')

        self.adaptive_upscale_limit.update(
            {
                'projected_peak_mb': int(projected_peak_mb),
                'available_vram_mb': int(available_for_process),
                'burst_window_mb': int(burst_window_mb),
                'headroom_mb': int(headroom_mb),
                'overhead_mb': int(overhead_mb),
                'shared_peak_mb': int(shared_upscale_peak_mb),
                'trigger': trigger,
                'reason': '; '.join(reason_parts),
            }
        )

        if not (oversubscribed or near_limit):
            return

        available_budget = max(0, int(available_for_process) - int(overhead_mb) - int(burst_window_mb))
        target_instances = max(1, int(available_budget // max(1, per_instance_mb)))
        target_instances = min(current_instances, target_instances)
        target_instances = max(target_instances, current_instances - 1)
        target_instances = max(1, int(target_instances))

        target_workers = max(1, int(target_instances * int(THREADS_PER_INSTANCE.get('upscale', 1))))

        if target_instances < current_instances:
            self.instance_counts['upscale'] = int(target_instances)
            self.worker_counts['upscale'] = int(target_workers)
            self.adaptive_upscale_limit.update(
                {
                    'applied': True,
                    'to_instances': int(target_instances),
                    'to_workers': int(target_workers),
                    'to': int(target_workers),
                }
            )

            if 'upscale' in self.pools:
                self._unload_pools(['upscale'])
                self._create_pools(['upscale'])
                self._run_stage_warmup_calibration()

            refreshed_stats = get_cuda_memory_stats_mb()
            if refreshed_stats:
                refreshed_profile = self.get_instance_vram_profile_mb()
                refreshed_allocated = int(refreshed_stats.get('process_allocated_mb', 0) or 0)
                refreshed_total = 0
                for stage in ('denoise', 'colorize', 'upscale'):
                    refreshed_total += int(self.instance_counts.get(stage, 0)) * int(refreshed_profile.get(stage, 0))
                self.runtime_overhead_observed_mb = max(0, int(refreshed_allocated - refreshed_total))

    def _run_with_peak_measure(self, run_fn):
        if not torch.cuda.is_available():
            return run_fn(), 0

        torch.cuda.synchronize()
        before_reserved = torch.cuda.memory_reserved()
        torch.cuda.reset_peak_memory_stats()
        result = run_fn()
        torch.cuda.synchronize()

        peak_reserved = torch.cuda.max_memory_reserved()
        after_reserved = torch.cuda.memory_reserved()
        delta_bytes = max(0, int(max(peak_reserved, after_reserved) - before_reserved))
        return result, _bytes_to_mb(delta_bytes)

    def _run_stage_warmup_calibration(self):
        self.stage_peak_inference_mb = {}
        if not torch.cuda.is_available():
            return
        if not isinstance(self.warmup_spec, dict) or not self.warmup_spec:
            return

        warmup_h, warmup_w = self._get_warmup_input_hw()
        warmup_image = np.zeros((warmup_h, warmup_w, 3), dtype=np.uint8)
        denoise_sigma = int(self.warmup_spec.get('denoise_sigma', getattr(self.config, 'denoise_sigma', 25)))
        target_width = int(self.warmup_spec.get('colorized_image_size', getattr(self.config, 'colorized_image_size', 576)))
        force_safe_width = bool(
            self.warmup_spec.get('force_safe_colorizer_width', getattr(self.config, 'force_safe_colorizer_width', False))
        )
        upscale_factor = max(1, int(self.warmup_spec.get('upscale_factor', getattr(self.config, 'upscale_factor', 4))))

        try:
            # Representative tile sizes for warmup (configurable)
            tile_sizes = list(self.warmup_spec.get('tile_sizes') or [64, 128, 256])
            max_tile_runs = max(1, int(self.warmup_spec.get('warmup_max_tile_runs', 3)))
            tile_sizes = tile_sizes[:max_tile_runs]

            # Helper to apply per-pass precision if available on the instance
            def _apply_instance_pass_precision(inst, stage_name):
                try:
                    from Backend.precision_utils import set_active_pass_precision
                except Exception:
                    set_active_pass_precision = None
                if set_active_pass_precision is None:
                    return
                model_dtype = None
                try:
                    model_dtype = getattr(inst, '_model_dtype', None)
                except Exception:
                    model_dtype = None
                try:
                    set_active_pass_precision(model_dtype, log=(lambda m, level=None: self._emit({'type':'log','message':m})) )
                except Exception:
                    pass

            # Denoise (pass1) warmup: iterate representative tiles
            if 'denoise' in self.pools:
                denoise_peak = 0
                for ts in tile_sizes:
                    h = min(int(warmup_h), int(ts))
                    w = min(int(warmup_w), int(ts))
                    img = np.zeros((h, w, 3), dtype=np.uint8)

                    def run_denoise_tile():
                        with torch.inference_mode():
                            with self.pools['denoise'].checkout(timeout=1.0) as denoiser:
                                _apply_instance_pass_precision(denoiser, 'denoise')
                                return denoiser.denoise(img, denoise_sigma)

                    _, peak_mb = self._run_with_peak_measure(run_denoise_tile)
                    denoise_peak = max(denoise_peak, int(peak_mb))

                self.stage_peak_inference_mb['denoise'] = max(0, int(denoise_peak))

            # Colorize (pass1) warmup: use adjusted widths and representative tiles
            if 'colorize' in self.pools:
                luminance_source = warmup_image.copy()
                colorize_peak = 0
                for ts in tile_sizes:
                    effective_w = min(int(ts), int(target_width), int(warmup_w))
                    if force_safe_width:
                        effective_w = min(effective_w, 576)
                    adjusted_w = effective_w - (effective_w % 32)
                    if adjusted_w == 0:
                        adjusted_w = 32
                    small_img = np.zeros((max(1, int(adjusted_w//2)), adjusted_w, 3), dtype=np.uint8)

                    def run_colorize_tile():
                        with torch.inference_mode():
                            with self.pools['colorize'].checkout(timeout=1.0) as colorizer:
                                _apply_instance_pass_precision(colorizer, 'colorize')
                                colorizer.set_image(small_img, adjusted_w)
                                colorized_output = colorizer.colorize()
                        try:
                            from transfer_quality import transfer_luminance_from_source as _trans_lum
                        except ImportError:
                            from Backend.transfer_quality import transfer_luminance_from_source as _trans_lum
                        return _trans_lum(luminance_source, colorized_output, self.config)

                    _, peak_mb = self._run_with_peak_measure(run_colorize_tile)
                    colorize_peak = max(colorize_peak, int(peak_mb))

                self.stage_peak_inference_mb['colorize'] = max(0, int(colorize_peak))

            # Upscale (pass2) warmup: representative tiles to trigger kernels / compilation
            if 'upscale' in self.pools:
                upscale_peak = 0
                for ts in tile_sizes:
                    h = min(int(warmup_h), int(ts))
                    w = min(int(warmup_w), int(ts))
                    img = np.zeros((h, w, 3), dtype=np.uint8)

                    def run_upscale_tile():
                        with torch.inference_mode():
                            with self.pools['upscale'].checkout(timeout=1.0) as upscaler:
                                _apply_instance_pass_precision(upscaler, 'upscale')
                                return upscaler.upscale(img, upscale_factor)

                    _, peak_mb = self._run_with_peak_measure(run_upscale_tile)
                    upscale_peak = max(upscale_peak, int(peak_mb))

                self.stage_peak_inference_mb['upscale'] = max(0, int(upscale_peak))
        except Exception:
            # Keep original behavior on any unexpected failure
            try:
                if 'denoise' in self.pools:
                    def run_denoise_stage():
                        with torch.inference_mode():
                            with self.pools['denoise'].checkout(timeout=1.0) as denoiser:
                                return denoiser.denoise(warmup_image, denoise_sigma)

                    warmup_image, denoise_peak_mb = self._run_with_peak_measure(run_denoise_stage)
                    self.stage_peak_inference_mb['denoise'] = max(0, int(denoise_peak_mb))

                if 'colorize' in self.pools:
                    luminance_source = warmup_image.copy()

                    effective_width = min(int(warmup_image.shape[1]), target_width)
                    if force_safe_width:
                        effective_width = min(int(warmup_image.shape[1]), 576)
                    adjusted_width = effective_width - (effective_width % 32)
                    if adjusted_width == 0:
                        adjusted_width = 32

                    def run_colorize_stage():
                        with torch.inference_mode():
                            with self.pools['colorize'].checkout(timeout=1.0) as colorizer:
                                colorizer.set_image(warmup_image, adjusted_width)
                                colorized_output = colorizer.colorize()
                        try:
                            from transfer_quality import transfer_luminance_from_source as _trans_lum
                        except ImportError:
                            from Backend.transfer_quality import transfer_luminance_from_source as _trans_lum
                        return _trans_lum(luminance_source, colorized_output, self.config)

                    warmup_image, colorize_peak_mb = self._run_with_peak_measure(run_colorize_stage)
                    self.stage_peak_inference_mb['colorize'] = max(0, int(colorize_peak_mb))

                if 'upscale' in self.pools:
                    def run_upscale_stage():
                        with torch.inference_mode():
                            with self.pools['upscale'].checkout(timeout=1.0) as upscaler:
                                return upscaler.upscale(warmup_image, upscale_factor)

                    warmup_image, upscale_peak_mb = self._run_with_peak_measure(run_upscale_stage)
                    self.stage_peak_inference_mb['upscale'] = max(0, int(upscale_peak_mb))
            except Exception:
                pass
        finally:
            self.cuda_after_warmup_stats = get_cuda_memory_stats_mb()

    @staticmethod
    def _queue_worker_bound(worker_count):
        base_workers = max(1, int(worker_count))
        return max(base_workers + 4, int(math.ceil(base_workers * 2.0)))

    @staticmethod
    def _queue_capacity_from_budget(stage_budget_bytes, item_bytes, worker_count):
        worker_bound = ProcessingPipeline._queue_worker_bound(worker_count)
        if stage_budget_bytes <= 0 or item_bytes <= 0:
            return max(QUEUE_MIN_ITEMS, min(QUEUE_MAX_ITEMS_CAP, worker_bound))

        budget_items = max(1, int(stage_budget_bytes // item_bytes))
        return max(QUEUE_MIN_ITEMS, min(QUEUE_MAX_ITEMS_CAP, worker_bound, budget_items))

    @staticmethod
    def _dynamic_ram_queue_cap(available_ram_mb, active_worker_count):
        if int(available_ram_mb) <= 0:
            return int(QUEUE_MAX_ITEMS_CAP)

        active_workers = max(1, int(active_worker_count))
        usable_ram = max(0, int(available_ram_mb) - int(QUEUE_RAM_HEADROOM_MB))
        ram_scaled = max(3, int(round(float(usable_ram) / 512.0)))
        floor_scaled = max(3, active_workers + 2)
        pressure_ceiling = max(floor_scaled, int(math.ceil(active_workers * 3.0)))

        cap = max(floor_scaled, min(ram_scaled, pressure_ceiling))
        return max(QUEUE_MIN_ITEMS, min(QUEUE_MAX_ITEMS_CAP, int(cap)))

    @staticmethod
    def _estimate_upscale_working_set_mb(input_h, input_w, upscale_factor):
        upscale_pixels = max(1, int(input_h) * int(input_w) * int(upscale_factor) * int(upscale_factor))
        working_bytes = max(1, upscale_pixels * 3 * 4)
        return max(1, int(round(working_bytes / (1024 * 1024))))

    def _effective_queue_budget_mb(self, transfer_config, input_h, input_w, worker_counts, upscale_factor):
        raw_budget = getattr(transfer_config, 'pipeline_queue_ram_budget_mb', 0)
        try:
            configured_budget_mb = int(raw_budget)
        except (TypeError, ValueError):
            configured_budget_mb = 0

        if configured_budget_mb > 0:
            return max(64, configured_budget_mb), 'manual'

        system_stats = get_system_memory_stats_mb()
        if not system_stats:
            return 384, 'default'

        available_ram_mb = int(system_stats.get('available_mb', 0))
        if available_ram_mb <= 0:
            return 384, 'default'

        upscale_workers = max(1, int(worker_counts.get('upscale', 1)))
        upscale_working_set_mb = self._estimate_upscale_working_set_mb(input_h, input_w, upscale_factor)

        from_available = int(available_ram_mb * 0.18)
        from_working_set = int(upscale_working_set_mb * max(1, upscale_workers) * 0.55)
        headroom_ceiling = max(128, available_ram_mb - 1024)

        auto_budget_mb = max(128, from_available, from_working_set)
        auto_budget_mb = min(1536, headroom_ceiling, auto_budget_mb)
        auto_budget_mb = max(128, auto_budget_mb)
        return int(auto_budget_mb), 'auto'

    def _get_folder_input_hw(self, folder_images, transfer_config):
        fallback_h, fallback_w = self._get_warmup_input_hw()
        input_h = fallback_h
        input_w = fallback_w

        if folder_images:
            sample_path = folder_images[0][0]
            try:
                with PIL.Image.open(sample_path) as sample_image:
                    input_w, input_h = sample_image.size
            except Exception:
                input_h = fallback_h
                input_w = fallback_w

        input_limit = sanitize_input_width_limit(getattr(transfer_config, 'input_image_size', DEFAULT_INPUT_WIDTH))
        if input_w > input_limit and input_w > 0:
            ratio = input_limit / float(input_w)
            input_w = int(input_limit)
            input_h = max(1, int(round(input_h * ratio)))

        return max(1, int(input_h)), max(1, int(input_w))

    def _resolve_queue_probe_images(self, folder_images, output_path, source):
        probe_images = list(folder_images or [])
        if source != 'output' or not probe_images:
            return probe_images

        output_probe_images = []
        for source_path, rel_dir in probe_images:
            image_name = os.path.basename(source_path)
            out_dir = os.path.join(output_path, rel_dir) if rel_dir else output_path
            pass1_file = os.path.join(out_dir, image_name)
            if os.path.exists(pass1_file):
                output_probe_images.append((pass1_file, rel_dir))
                if len(output_probe_images) >= 8:
                    break

        if output_probe_images:
            return output_probe_images
        return probe_images

    def _queue_limits_for_folder(
        self,
        folder_images,
        transfer_config,
        worker_counts_override=None,
        input_hw=None,
        stages=None,
        writer_workers=None,
    ):
        if input_hw is None:
            input_h, input_w = self._get_folder_input_hw(folder_images, transfer_config)
        else:
            input_h = max(1, int(input_hw[0]))
            input_w = max(1, int(input_hw[1]))

        worker_counts = worker_counts_override if isinstance(worker_counts_override, dict) else self.worker_counts
        writer_worker_count = max(1, int(writer_workers if writer_workers is not None else self.writer_threads))

        requested_stages = set(stages or ('denoise', 'colorize', 'upscale'))
        active_chain = [
            stage
            for stage in ('denoise', 'colorize', 'upscale')
            if stage in requested_stages and stage in self.pools
        ]

        upscale_enabled = bool(getattr(transfer_config, 'upscale', False)) and 'upscale' in active_chain
        upscale_factor = max(1, int(getattr(transfer_config, 'upscale_factor', 4))) if upscale_enabled else 1

        queue_budget_mb, queue_budget_source = self._effective_queue_budget_mb(
            transfer_config,
            input_h,
            input_w,
            worker_counts,
            upscale_factor,
        )
        queue_budget_bytes = int(queue_budget_mb * 1024 * 1024)

        input_item_bytes = max(1, int(input_h * input_w * 3))
        output_item_bytes = max(1, int(input_h * input_w * 3 * upscale_factor * upscale_factor))

        edge_specs = []
        if active_chain:
            first_stage = active_chain[0]
            edge_specs.append(
                {
                    'key': 'decode_to_denoise',
                    'label': f'input->{first_stage}',
                    'item_bytes': input_item_bytes,
                    'worker_count': max(1, int(worker_counts.get(first_stage, 1))),
                }
            )

            for idx in range(1, len(active_chain)):
                previous_stage = active_chain[idx - 1]
                next_stage = active_chain[idx]
                edge_key = 'denoise_to_colorize' if previous_stage == 'denoise' else 'colorize_to_upscale'
                edge_specs.append(
                    {
                        'key': edge_key,
                        'label': f'{previous_stage}->{next_stage}',
                        'item_bytes': input_item_bytes,
                        'worker_count': max(1, int(worker_counts.get(next_stage, 1))),
                    }
                )

            last_stage = active_chain[-1]
            write_item_bytes = output_item_bytes if last_stage == 'upscale' else input_item_bytes
            edge_specs.append(
                {
                    'key': 'upscale_to_write',
                    'label': f'{last_stage}->write',
                    'item_bytes': write_item_bytes,
                    'worker_count': writer_worker_count,
                }
            )

        edge_budgets = {}
        if edge_specs:
            weighted_edges = []
            for edge in edge_specs:
                edge_weight = math.sqrt(max(1, int(edge['item_bytes']))) * float(
                    self._queue_worker_bound(edge['worker_count'])
                )
                weighted_edges.append((edge, max(1.0, float(edge_weight))))

            total_weight = sum(weight for _, weight in weighted_edges)
            remaining_budget = max(0, int(queue_budget_bytes))

            for edge, weight in weighted_edges[:-1]:
                budget = int(round(queue_budget_bytes * (weight / max(1.0, total_weight))))
                budget = max(int(edge['item_bytes']), budget)
                edge_budgets[edge['key']] = budget
                remaining_budget = max(0, remaining_budget - budget)

            last_edge = weighted_edges[-1][0]
            edge_budgets[last_edge['key']] = max(int(last_edge['item_bytes']), int(remaining_budget))

        system_stats = get_system_memory_stats_mb()
        available_ram_mb = int(system_stats.get('available_mb', 0)) if system_stats else 0

        limits = {
            'decode_to_denoise': int(QUEUE_MIN_ITEMS),
            'denoise_to_colorize': int(QUEUE_MIN_ITEMS),
            'colorize_to_upscale': int(QUEUE_MIN_ITEMS),
            'upscale_to_write': int(QUEUE_MIN_ITEMS),
            'queue_budget_mb': int(queue_budget_mb),
            'queue_budget_source': str(queue_budget_source),
            'available_ram_mb': int(available_ram_mb),
            'input_hw': [int(input_h), int(input_w)],
            'upscale_factor': int(upscale_factor),
            'input_item_mb': round(input_item_bytes / (1024 * 1024), 2),
            'output_item_mb': round(output_item_bytes / (1024 * 1024), 2),
            'writer_workers': int(writer_worker_count),
            'active_stages': list(active_chain),
        }

        edge_debug = []
        for edge in edge_specs:
            edge_key = edge['key']
            edge_item_bytes = max(1, int(edge['item_bytes']))
            edge_worker_count = max(1, int(edge['worker_count']))
            edge_budget = int(edge_budgets.get(edge_key, edge_item_bytes))
            edge_cap = self._queue_capacity_from_budget(edge_budget, edge_item_bytes, edge_worker_count)
            limits[edge_key] = max(int(limits.get(edge_key, QUEUE_MIN_ITEMS)), int(edge_cap))
            edge_debug.append(
                {
                    'key': edge_key,
                    'label': str(edge.get('label', edge_key)),
                    'item_mb': round(edge_item_bytes / (1024 * 1024), 2),
                    'budget_mb': round(edge_budget / (1024 * 1024), 2),
                    'workers': int(edge_worker_count),
                    'worker_bound': int(self._queue_worker_bound(edge_worker_count)),
                    'cap': int(limits[edge_key]),
                }
            )

        active_worker_count = int(
            writer_worker_count
            + sum(max(0, int(worker_counts.get(stage, 0))) for stage in active_chain)
        )
        ram_queue_cap = self._dynamic_ram_queue_cap(available_ram_mb, active_worker_count)

        edge_worker_floor = {
            str(edge.get('key', '')): max(QUEUE_MIN_ITEMS, int(edge.get('workers', 1)) + 2)
            for edge in edge_debug
            if isinstance(edge, dict)
        }
        if 'upscale_to_write' in edge_worker_floor:
            edge_worker_floor['upscale_to_write'] = max(
                edge_worker_floor['upscale_to_write'],
                max(QUEUE_MIN_ITEMS, int(edge_worker_floor.get('upscale_to_write', 1)) + 6),
            )

        for key in ('decode_to_denoise', 'denoise_to_colorize', 'colorize_to_upscale', 'upscale_to_write'):
            worker_floor = int(edge_worker_floor.get(key, QUEUE_MIN_ITEMS))
            limits[key] = max(worker_floor, min(ram_queue_cap, int(limits.get(key, QUEUE_MIN_ITEMS))))

        limits['ram_queue_cap'] = int(ram_queue_cap)
        limits['edge_debug'] = edge_debug
        return limits

    def get_instance_vram_profile_mb(self):
        profile = get_default_instance_vram_profile_mb()
        for stage in ('denoise', 'colorize', 'upscale'):
            observed = self.instance_vram_observed_mb.get(stage)
            if observed is not None and int(observed) > 0:
                profile[stage] = max(1, int(observed))
        return profile

    def get_vram_calibration_report(self):
        profile = self.get_instance_vram_profile_mb()
        shared_stage_peak = self.get_shared_stage_peak_mb()
        worker_counts = {
            'denoise': int(self.worker_counts.get('denoise', 0)) if 'denoise' in self.pools else 0,
            'colorize': int(self.worker_counts.get('colorize', 0)) if 'colorize' in self.pools else 0,
            'upscale': int(self.worker_counts.get('upscale', 0)) if 'upscale' in self.pools else 0,
        }
        projected_breakdown_base = estimate_vram_breakdown_mb(
            self.instance_counts.get('denoise', 0),
            self.instance_counts.get('colorize', 0),
            self.instance_counts.get('upscale', 0),
            instance_profile_mb=profile,
            overhead_mb=self.runtime_overhead_observed_mb,
        )

        pass1_base_mb = int(projected_breakdown_base.get('pass1_total_mb', 0) or 0)
        pass2_base_mb = int(projected_breakdown_base.get('pass2_total_mb', 0) or 0)
        pass1_shared_mb = 0
        pass2_shared_mb = 0

        if pass1_base_mb > 0:
            pass1_shared_mb = max(
                int(shared_stage_peak.get('denoise', 0) or 0),
                int(shared_stage_peak.get('colorize', 0) or 0),
            )
        if pass2_base_mb > 0:
            pass2_shared_mb = int(shared_stage_peak.get('upscale', 0) or 0)

        projected_pass1_mb = pass1_base_mb + pass1_shared_mb if pass1_base_mb > 0 else 0
        projected_pass2_mb = pass2_base_mb + pass2_shared_mb if pass2_base_mb > 0 else 0

        if projected_pass1_mb <= 0 and projected_pass2_mb <= 0:
            projected_peak_pass = 'none'
            projected_peak_mb = 0
        elif projected_pass1_mb >= projected_pass2_mb:
            projected_peak_pass = 'pass1'
            projected_peak_mb = projected_pass1_mb
        else:
            projected_peak_pass = 'pass2'
            projected_peak_mb = projected_pass2_mb

        observed_stages = {
            stage
            for stage in ('denoise', 'colorize', 'upscale')
            if int(self.instance_vram_observed_mb.get(stage, 0) or 0) > 0
        }
        heuristic_stages = {
            stage
            for stage in ('denoise', 'colorize', 'upscale')
            if stage not in observed_stages
        }
        if self.stage_peak_inference_mb:
            projection_source = 'warmup_peak'
        elif observed_stages and heuristic_stages:
            projection_source = 'init_delta+heuristic'
        elif observed_stages:
            projection_source = 'init_delta'
        else:
            projection_source = 'default_heuristic'

        return {
            'instance_counts': dict(self.instance_counts),
            'worker_counts': dict(worker_counts),
            'instance_vram_profile_mb': profile,
            'instance_vram_profile_base_mb': dict(profile),
            'stage_peak_inference_mb': dict(self.stage_peak_inference_mb),
            'shared_stage_peak_mb': dict(shared_stage_peak),
            'shared_pass1_mb': int(pass1_shared_mb),
            'shared_pass2_mb': int(pass2_shared_mb),
            'runtime_overhead_mb': int(self.runtime_overhead_observed_mb),
            'projected_total_mb': int(projected_peak_mb),
            'projected_peak_mb': int(projected_peak_mb),
            'projected_pass1_base_mb': int(pass1_base_mb),
            'projected_pass2_base_mb': int(pass2_base_mb),
            'projected_pass1_mb': int(projected_pass1_mb),
            'projected_pass2_mb': int(projected_pass2_mb),
            'projected_peak_pass': str(projected_peak_pass),
            'projection_source': str(projection_source),
            'adaptive_upscale_limit': dict(self.adaptive_upscale_limit),
            'warmup_input_hw': list(self._get_warmup_input_hw()),
            'warmup_upscale_factor': int(self.warmup_spec.get('upscale_factor', getattr(self.config, 'upscale_factor', 4))),
            'writer_threads': int(self.writer_threads),
            'cuda_baseline': dict(self.cuda_baseline_stats) if self.cuda_baseline_stats else {},
            'cuda_after_pool': dict(self.cuda_after_pool_stats) if self.cuda_after_pool_stats else {},
            'cuda_after_warmup': dict(self.cuda_after_warmup_stats) if self.cuda_after_warmup_stats else {},
        }

    def _create_pools(self, stages):
        for stage in stages:
            if stage in self.pools:
                continue

            factory = self.model_factories.get(stage)
            count = int(self.instance_counts.get(stage, 0))
            if not callable(factory) or count <= 0:
                continue

            stage_before = get_cuda_memory_stats_mb() if torch.cuda.is_available() else {}
            # Pass arch name for upscale stage so weight-sharing allowlist works
            arch_name = None
            if stage == 'upscale':
                arch_name = getattr(self.config, 'upscaler_type', None)
            self.pools[stage] = ModelInstancePool(factory, count, stage, arch_name=arch_name)
            stage_after = get_cuda_memory_stats_mb() if torch.cuda.is_available() else {}

            precision_summary = self.pools[stage].precision_summary
            if precision_summary:
                self._emit(
                    {
                        'type': 'log',
                        'message': f"[*] {stage} precision => {precision_summary}",
                    },
                    level=LogLevel.DEBUG,
                )

            if stage_before and stage_after:
                before_reserved = int(stage_before.get('process_reserved_mb', 0))
                after_reserved = int(stage_after.get('process_reserved_mb', before_reserved))
                delta = max(0, after_reserved - before_reserved)
                if delta <= 0:
                    before_alloc = int(stage_before.get('process_allocated_mb', 0))
                    after_alloc = int(stage_after.get('process_allocated_mb', before_alloc))
                    delta = max(0, after_alloc - before_alloc)
                if delta > 0:
                    self.instance_vram_observed_mb[stage] = max(1, int(round(delta / max(1, count))))

            if stage_after:
                self.cuda_after_pool_stats = stage_after

            observed = self.instance_vram_observed_mb.get(stage)
            observed_text = f' (~{observed} MB/inst)' if observed is not None else ''
            self._emit(
                {
                    'type': 'log',
                    'message': f"[*] Loaded {count} {stage} instance(s){observed_text}",
                },
                level=LogLevel.DEBUG,
            )

    def _unload_pools(self, stages):
        unloaded = []
        for stage in stages:
            pool = self.pools.pop(stage, None)
            if pool is None:
                continue
            instances = pool.drain()
            for instance in instances:
                del instance
            del instances
            unloaded.append(stage)

        if unloaded:
            gc.collect()
            try:
                from utils.utils import clear_torch_cache as _clear_cache
            except ImportError:
                from Backend.utils.utils import clear_torch_cache as _clear_cache
            _clear_cache()
            self._emit(
                {
                    'type': 'log',
                    'message': f"[*] Unloaded pools: {', '.join(unloaded)}. GC + CUDA cache cleared.",
                },
                level=LogLevel.DEBUG,
            )

    def _count_pass1_outputs(self, grouped_by_folder, output_path):
        count = 0
        for rel_dir, folder_images in grouped_by_folder.items():
            out_dir = os.path.join(output_path, rel_dir) if rel_dir else output_path
            for path, _ in folder_images:
                name = os.path.basename(path)
                if os.path.exists(os.path.join(out_dir, name)):
                    count += 1
        return count

    def _emit_calibration_report(self):
        report = self.get_vram_calibration_report()
        self._emit(
            {
                'type': 'pipeline_calibration',
                'report': report,
            },
            level=LogLevel.TRACE,
        )

    def _elide_path(self, rel_dir):
        rel_text = str(rel_dir or '').replace('\\', '/')
        parts = [p for p in rel_text.split('/') if p]
        if len(parts) <= 2:
            return rel_text
        return f"{parts[0]} -> {'/'.join(parts[-2:])}"

    def _check_folder_skip(self, folder_images, output_path, rel_dir, overwrite, pass_stop_event=None):
        if overwrite:
            return False, 0, ''

        skipped = []
        needs_processing = []
        out_dir = os.path.join(output_path, rel_dir) if rel_dir else output_path

        for path, _ in folder_images:
            if not self._wait_while_paused(pass_stop_event):
                return False, len(skipped), ''
            name = os.path.basename(path)
            out_file = os.path.join(out_dir, name)
            if os.path.exists(out_file):
                skipped.append(name)
            else:
                needs_processing.append(name)

        if skipped and not needs_processing:
            return True, len(skipped), f"all {len(skipped)} skipped ({skipped[0]}->{skipped[-1]})"

        return False, len(skipped), ''

    def _check_upscale_skip(self, folder_images, output_path, rel_dir, transfer_config, pass_stop_event=None):
        skipped = []
        needs_processing = []
        out_dir = os.path.join(output_path, rel_dir) if rel_dir else output_path

        for source_path, _ in folder_images:
            if not self._wait_while_paused(pass_stop_event):
                return False, len(skipped), ''
            name = os.path.basename(source_path)
            out_file = os.path.join(out_dir, name)

            if not os.path.exists(out_file):
                needs_processing.append(name)
                continue

            try:
                src_size = os.path.getsize(source_path)
                out_size = os.path.getsize(out_file)
                if src_size > 0 and out_size >= int(src_size * 1.5):
                    skipped.append(name)
                    continue
            except OSError:
                pass

            needs_processing.append(name)

        if skipped and not needs_processing:
            return True, len(skipped), f"all {len(skipped)} already upscaled ({skipped[0]}->{skipped[-1]})"

        return False, len(skipped), ''

    def _force_put(self, q, item, timeout=0.5, pass_stop_event=None, queue_name='queue', detailed_logs=False):
        started_at = time.monotonic()
        last_report_at = started_at

        while True:
            try:
                q.put(item, timeout=timeout)
                if detailed_logs:
                    waited = max(0.0, time.monotonic() - started_at)
                    if waited >= QUEUE_DIAG_WAIT_WARN_S:
                        self._emit(
                            {
                                'type': 'log',
                                'message': (
                                    '[*] Sentinel enqueue cleared => '
                                    f'{queue_name} '
                                    f'wait:{waited:.1f}s '
                                    f'q:{_queue_state_text(q)}'
                                ),
                            },
                            level=LogLevel.DEBUG,
                        )
                return True
            except queue.Full:
                terminate_set = self.terminate_event.is_set()
                pass_stop_set = bool(pass_stop_event.is_set()) if isinstance(pass_stop_event, threading.Event) else False
                if terminate_set or pass_stop_set:
                    waited = max(0.0, time.monotonic() - started_at)
                    self._emit(
                        {
                            'type': 'log',
                            'message': (
                                '[!] Sentinel enqueue aborted => '
                                f'{queue_name} '
                                f'wait:{waited:.1f}s '
                                f'q:{_queue_state_text(q)} '
                                f'({self._stop_state_text(pass_stop_event)})'
                            ),
                        },
                        level=LogLevel.WARNING,
                    )
                    return False

                if detailed_logs:
                    now = time.monotonic()
                    waited = max(0.0, now - started_at)
                    if waited >= QUEUE_DIAG_WAIT_WARN_S and (now - last_report_at) >= QUEUE_DIAG_WAIT_REPORT_INTERVAL_S:
                        self._emit(
                            {
                                'type': 'log',
                                'message': (
                                    '[*] Sentinel enqueue wait => '
                                    f'{queue_name} '
                                    f'wait:{waited:.1f}s '
                                    f'q:{_queue_state_text(q)} '
                                    f'({self._stop_state_text(pass_stop_event)})'
                                ),
                            },
                            level=LogLevel.DEBUG,
                        )
                        last_report_at = now
                continue

    def _drain_all_queues(self, *queues_to_drain):
        for work_q in queues_to_drain:
            while True:
                try:
                    item = work_q.get_nowait()
                    if isinstance(item, dict):
                        item['image'] = None
                        item.clear()
                    work_q.task_done()
                except queue.Empty:
                    break

    def _get_stage_queues(self, stage, stages, queues, write_queue):
        active_chain = [s for s in ('denoise', 'colorize', 'upscale') if s in stages and s in self.pools]
        stage_idx = active_chain.index(stage)

        if stage_idx == 0:
            in_q = queues.get('to_denoise') or queues.get('to_colorize') or queues.get('to_upscale')
        elif stage == 'colorize' and 'denoise' in active_chain:
            in_q = queues['from_denoise']
        elif stage == 'upscale' and 'colorize' in active_chain:
            in_q = queues['from_colorize']
        elif stage == 'upscale' and 'denoise' in active_chain:
            in_q = queues['from_denoise']
        else:
            in_q = queues.get(f'to_{stage}', write_queue)

        if stage_idx < len(active_chain) - 1:
            next_stage = active_chain[stage_idx + 1]
            out_q = queues.get(f'from_{stage}', queues.get(f'to_{next_stage}', write_queue))
        else:
            out_q = write_queue

        return in_q, out_q

    def _get_first_queue(self, stages, queues, write_queue=None):
        for stage in ('denoise', 'colorize', 'upscale'):
            if stage in stages:
                return queues.get(f'to_{stage}') or queues.get(f'from_{stage}')
        return write_queue

    def _get_first_stage_worker_count(self, stages):
        for stage in ('denoise', 'colorize', 'upscale'):
            if stage in stages and stage in self.pools:
                return max(1, int(self.worker_counts.get(stage, 1)))
        return 1

    def _run_pass(
        self,
        pass_name,
        stages,
        grouped_by_folder,
        output_path,
        overwrite,
        detailed_logs,
        transfer_config,
        source,
        total_images,
        progress_span,
    ):
        stats = {
            'completed': 0,
            'skipped': 0,
            'failed': 0,
            'cancelled': 0,
            'folders_done': 0,
        }

        if total_images <= 0:
            self._emit(
                {
                    'type': 'progress',
                    'value': float(progress_span[1]),
                    'eta': f'ETA: {pass_name} 0/0',
                    'pass': pass_name,
                    'done': 0,
                    'total': 0,
                },
                level=LogLevel.DEBUG,
            )
            return stats

        pass_stop_event = threading.Event()
        stop_events = (self.terminate_event, pass_stop_event)
        stats_lock = threading.Lock()
        pass_started_at = time.monotonic()
        diag_ctx = {
            'enabled': bool(detailed_logs),
            'pass_started_at': float(pass_started_at),
            'last_progress_at': float(pass_started_at),
            'inflight': {},
            'inflight_lock': threading.Lock(),
            'writer_lock': threading.Lock(),
            'writer_last_report_at': float(pass_started_at),
            'writer_last_done': 0,
            'writer_samples_ms': [],
        }

        first_folder = list(grouped_by_folder.values())[0] if grouped_by_folder else []
        queue_probe_folder = self._resolve_queue_probe_images(first_folder, output_path, source)
        writer_worker_count = max(1, int(self.writer_threads))
        if 'upscale' in stages:
            writer_worker_count = max(2, writer_worker_count)
        queue_limits = self._queue_limits_for_folder(
            queue_probe_folder,
            transfer_config,
            stages=stages,
            writer_workers=writer_worker_count,
        )

        edge_debug = queue_limits.get('edge_debug', []) if isinstance(queue_limits, dict) else []
        active_edge_keys = {
            str(edge.get('key', ''))
            for edge in edge_debug
            if isinstance(edge, dict)
        }
        edge_labels = {
            'decode_to_denoise': 'input->first-stage',
            'denoise_to_colorize': 'denoise->colorize',
            'colorize_to_upscale': 'colorize->upscale',
            'upscale_to_write': 'last-stage->write',
        }
        active_edge_label_list = [
            str(edge.get('label', edge_labels.get(edge.get('key', ''), edge.get('key', '?'))))
            for edge in edge_debug
            if isinstance(edge, dict)
        ]
        if active_edge_label_list:
            active_edge_summary = ', '.join(active_edge_label_list)
        else:
            active_edge_summary = 'none'

        def edge_cap_with_state(edge_key, short_name):
            state = 'A' if edge_key in active_edge_keys else 'I'
            return f"{short_name}:{int(queue_limits.get(edge_key, QUEUE_MIN_ITEMS))}[{state}]"

        self._emit(
            {
                'type': 'log',
                'message': (
                    '[*] Queue caps => '
                    f"{edge_cap_with_state('decode_to_denoise', 'decode')} "
                    f"{edge_cap_with_state('denoise_to_colorize', 'denoise')} "
                    f"{edge_cap_with_state('colorize_to_upscale', 'colorize')} "
                    f"{edge_cap_with_state('upscale_to_write', 'write')} "
                    f"(active edges: {active_edge_summary}; "
                    f"budget {queue_limits['queue_budget_mb']} MB/{queue_limits.get('queue_budget_source', 'manual')}, "
                    f"ram-cap {queue_limits.get('ram_queue_cap', QUEUE_MAX_ITEMS_CAP)})"
                ),
            },
            level=LogLevel.DEBUG,
        )

        queue_state_level = LogLevel.DEBUG if detailed_logs else LogLevel.TRACE

        for edge in edge_debug:
            self._emit(
                {
                    'type': 'log',
                    'message': (
                        '[*] Queue edge => '
                        f"{edge.get('label', edge.get('key', '?'))} "
                        f"cap:{edge.get('cap', '?')} "
                        f"workers:{edge.get('workers', '?')} "
                        f"bound:{edge.get('worker_bound', '?')} "
                        f"item:{edge.get('item_mb', '?')} MB "
                        f"budget:{edge.get('budget_mb', '?')} MB"
                    ),
                },
                level=queue_state_level,
            )

        for edge_key in ('decode_to_denoise', 'denoise_to_colorize', 'colorize_to_upscale', 'upscale_to_write'):
            if edge_key in active_edge_keys:
                continue
            self._emit(
                {
                    'type': 'log',
                    'message': (
                        '[*] Queue edge => '
                        f"{edge_labels.get(edge_key, edge_key)} inactive for this pass "
                        f"(floor cap {int(queue_limits.get(edge_key, QUEUE_MIN_ITEMS))})."
                    ),
                },
                level=queue_state_level,
            )

        queues = {}
        if 'denoise' in stages:
            queues['to_denoise'] = queue.Queue(maxsize=queue_limits.get('decode_to_denoise', 12))
            if 'colorize' in stages or 'upscale' in stages:
                queues['from_denoise'] = queue.Queue(maxsize=queue_limits.get('denoise_to_colorize', 12))

        if 'colorize' in stages:
            if 'denoise' not in stages:
                queues['to_colorize'] = queue.Queue(maxsize=queue_limits.get('decode_to_denoise', 12))
            queues['from_colorize'] = queue.Queue(maxsize=queue_limits.get('colorize_to_upscale', 4))

        has_postprocess = 'colorize' in stages
        if has_postprocess:
            if 'upscale' in stages:
                queues['from_postprocess'] = queue.Queue(maxsize=queue_limits.get('colorize_to_upscale', 4))

        if 'upscale' in stages and 'denoise' not in stages and 'colorize' not in stages:
            queues['to_upscale'] = queue.Queue(maxsize=queue_limits.get('decode_to_denoise', 4))

        write_queue = queue.Queue(maxsize=queue_limits.get('upscale_to_write', 4))
        queues['__write_queue'] = write_queue

        active_chain = [s for s in ('denoise', 'colorize', 'upscale') if s in stages and s in self.pools]
        if has_postprocess:
            colorize_idx = active_chain.index('colorize')
            active_chain.insert(colorize_idx + 1, 'postprocess')

        stage_threads = {}
        stage_in_queues = {}

        postprocess_worker_count = max(1, int(self.worker_counts.get('colorize', 1)))

        for stage in active_chain:
            if stage == 'postprocess':
                in_q = queues['from_colorize']
                out_q = queues.get('from_postprocess', write_queue)
                worker_count = postprocess_worker_count
            else:
                in_q, out_q = self._get_stage_queues(stage, stages, queues, write_queue)
                if stage == 'colorize' and has_postprocess:
                    out_q = queues['from_colorize']
                if stage == 'upscale' and has_postprocess:
                    in_q = queues['from_postprocess']
                worker_count = max(1, int(self.worker_counts.get(stage, 1)))

            stage_in_queues[stage] = in_q
            threads = []
            for _ in range(worker_count):
                t = threading.Thread(
                    target=self._persistent_stage_worker,
                    args=(
                        stage,
                        in_q,
                        out_q,
                        stop_events,
                        pass_stop_event,
                        transfer_config,
                        detailed_logs,
                        diag_ctx,
                    ),
                    daemon=True,
                )
                threads.append(t)
            stage_threads[stage] = threads

        producer_thread = threading.Thread(
            target=self._persistent_producer,
            args=(
                grouped_by_folder,
                output_path,
                overwrite,
                detailed_logs,
                transfer_config,
                source,
                stages,
                queues,
                stop_events,
                pass_stop_event,
                stats,
                stats_lock,
                diag_ctx,
            ),
            daemon=True,
        )

        writer_threads = []
        for _ in range(writer_worker_count):
            writer_thread = threading.Thread(
                target=self._persistent_writer,
                args=(
                    write_queue,
                    output_path,
                    stop_events,
                    stats,
                    stats_lock,
                    total_images,
                    pass_name,
                    detailed_logs,
                    float(progress_span[0]),
                    float(progress_span[1]),
                    diag_ctx,
                    transfer_config,
                ),
                daemon=True,
            )
            writer_threads.append(writer_thread)

        queue_state_pairs = [
            (name, q_obj)
            for name, q_obj in (
                ('to_denoise', queues.get('to_denoise')),
                ('from_denoise', queues.get('from_denoise')),
                ('to_colorize', queues.get('to_colorize')),
                ('from_colorize', queues.get('from_colorize')),
                ('from_postprocess', queues.get('from_postprocess')),
                ('to_upscale', queues.get('to_upscale')),
                ('write', write_queue),
            )
            if q_obj is not None
        ]

        heartbeat_stop_event = threading.Event()
        heartbeat_thread = None

        def _join_thread_with_diag(thread_obj, label):
            join_started = time.monotonic()
            last_report = join_started
            if detailed_logs:
                self._emit(
                    {
                        'type': 'log',
                        'message': f'[*] Join begin => {label}',
                    },
                    level=LogLevel.DEBUG,
                )

            while thread_obj.is_alive():
                thread_obj.join(timeout=0.5)
                now = time.monotonic()
                if (
                    thread_obj.is_alive()
                    and self.terminate_event.is_set()
                    and (now - join_started) >= QUEUE_DIAG_JOIN_ABORT_ON_STOP_S
                ):
                    if detailed_logs:
                        self._emit(
                            {
                                'type': 'log',
                                'message': (
                                    '[!] Join aborted on stop => '
                                    f'{label} '
                                    f'elapsed:{max(0.0, now - join_started):.1f}s '
                                    f'({self._stop_state_text(pass_stop_event)})'
                                ),
                            },
                            level=LogLevel.WARNING,
                        )
                    return False

                if detailed_logs and thread_obj.is_alive() and (now - last_report) >= QUEUE_DIAG_JOIN_REPORT_INTERVAL_S:
                    self._emit(
                        {
                            'type': 'log',
                            'message': (
                                f'[*] Join wait => {label} '
                                f'elapsed:{max(0.0, now - join_started):.1f}s '
                                f'({self._stop_state_text(pass_stop_event)})'
                            ),
                        },
                        level=LogLevel.DEBUG,
                    )
                    last_report = now

            if detailed_logs:
                self._emit(
                    {
                        'type': 'log',
                        'message': (
                            '[*] Join done => '
                            f'{label} '
                            f'elapsed:{max(0.0, time.monotonic() - join_started):.2f}s'
                        ),
                    },
                    level=LogLevel.DEBUG,
                )
            return True

        def _queue_join_with_diag(work_q, label):
            def _drain_queue_now():
                drained = 0
                while True:
                    try:
                        queued_item = work_q.get_nowait()
                    except queue.Empty:
                        break

                    if isinstance(queued_item, dict):
                        queued_item['image'] = None
                    try:
                        work_q.task_done()
                        drained += 1
                    except ValueError:
                        break
                return drained

            join_started = time.monotonic()
            last_report = join_started
            if detailed_logs:
                self._emit(
                    {
                        'type': 'log',
                        'message': f'[*] Queue join begin => {label} q:{_queue_state_text(work_q)}',
                    },
                    level=LogLevel.DEBUG,
                )

            while True:
                pending = int(getattr(work_q, 'unfinished_tasks', 0) or 0)
                if pending <= 0:
                    break
                time.sleep(0.2)
                now = time.monotonic()

                if (
                    self.terminate_event.is_set()
                    and (now - join_started) >= QUEUE_DIAG_JOIN_ABORT_ON_STOP_S
                ):
                    drained = _drain_queue_now()
                    if detailed_logs:
                        pending_after = int(getattr(work_q, 'unfinished_tasks', 0) or 0)
                        self._emit(
                            {
                                'type': 'log',
                                'message': (
                                    '[!] Queue join aborted on stop => '
                                    f'{label} '
                                    f'pending_before:{pending} '
                                    f'drained:{drained} '
                                    f'pending_after:{pending_after} '
                                    f'q:{_queue_state_text(work_q)} '
                                    f'elapsed:{max(0.0, now - join_started):.1f}s '
                                    f'({self._stop_state_text(pass_stop_event)})'
                                ),
                            },
                            level=LogLevel.WARNING,
                        )
                    return False

                if detailed_logs and (now - last_report) >= QUEUE_DIAG_JOIN_REPORT_INTERVAL_S:
                    self._emit(
                        {
                            'type': 'log',
                            'message': (
                                f'[*] Queue join wait => {label} '
                                f'pending:{pending} '
                                f'q:{_queue_state_text(work_q)} '
                                f'elapsed:{max(0.0, now - join_started):.1f}s '
                                f'({self._stop_state_text(pass_stop_event)})'
                            ),
                        },
                        level=LogLevel.DEBUG,
                    )
                    last_report = now

            if detailed_logs:
                self._emit(
                    {
                        'type': 'log',
                        'message': (
                            '[*] Queue join done => '
                            f'{label} '
                            f'elapsed:{max(0.0, time.monotonic() - join_started):.2f}s '
                            f'q:{_queue_state_text(work_q)}'
                        ),
                    },
                    level=LogLevel.DEBUG,
                )
            return True

        if bool(diag_ctx.get('enabled', False)):
            def _heartbeat_loop():
                while not heartbeat_stop_event.wait(QUEUE_DIAG_HEARTBEAT_INTERVAL_S):
                    now = time.monotonic()
                    last_progress_at = float(diag_ctx.get('last_progress_at', pass_started_at) or pass_started_at)
                    idle_seconds = max(0.0, now - last_progress_at)
                    if idle_seconds < QUEUE_DIAG_HEARTBEAT_INTERVAL_S:
                        continue

                    system_stats = get_system_memory_stats_mb()
                    if system_stats:
                        avail_ram = int(system_stats.get('available_mb', 0) or 0)
                        if avail_ram > 0 and avail_ram < int(QUEUE_RAM_PRESSURE_WARN_MB):
                            self._emit(
                                {
                                    'type': 'log',
                                    'message': (
                                        '[!] RAM pressure: '
                                        f'{avail_ram} MB free '
                                        f'(headroom target: {QUEUE_RAM_HEADROOM_MB} MB)'
                                    ),
                                },
                                level=LogLevel.WARNING,
                            )

                    with stats_lock:
                        done_total = int(stats['completed'] + stats['skipped'] + stats['failed'] + stats['cancelled'])

                    inflight = {}
                    inflight_lock = diag_ctx.get('inflight_lock')
                    raw_inflight = diag_ctx.get('inflight')
                    if isinstance(raw_inflight, dict) and hasattr(inflight_lock, 'acquire'):
                        with inflight_lock:
                            inflight = dict(raw_inflight)

                    oldest_inflight_s = 0.0
                    if inflight:
                        oldest_inflight_s = max(0.0, now - float(min(inflight.values())))

                    queue_summary = ', '.join(
                        f'{name}:{_queue_state_text(q_obj)}' for name, q_obj in queue_state_pairs
                    ) or 'none'
                    stage_summary = ', '.join(
                        f'{stage}:{sum(1 for th in threads if th.is_alive())}/{len(threads)}'
                        for stage, threads in stage_threads.items()
                    ) or 'none'
                    writer_alive = sum(1 for writer_th in writer_threads if writer_th.is_alive())
                    producer_alive = int(producer_thread.is_alive())

                    self._emit(
                        {
                            'type': 'log',
                            'message': (
                                '[*] Pass heartbeat => '
                                f'idle:{idle_seconds:.1f}s '
                                f'done:{done_total}/{total_images} '
                                f'inflight:{len(inflight)} '
                                f'oldest_inflight:{oldest_inflight_s:.1f}s '
                                f'producer:{producer_alive} '
                                f'stages:[{stage_summary}] '
                                f'writer:{writer_alive}/{writer_worker_count} '
                                f'queues:[{queue_summary}] '
                                f'({self._stop_state_text(pass_stop_event)})'
                            ),
                        },
                        level=LogLevel.DEBUG,
                    )

            heartbeat_thread = threading.Thread(target=_heartbeat_loop, daemon=True)

        for writer_thread in writer_threads:
            writer_thread.start()
        for stage in active_chain:
            for thread in stage_threads[stage]:
                thread.start()
        producer_thread.start()
        if heartbeat_thread is not None:
            heartbeat_thread.start()

        _join_thread_with_diag(producer_thread, f'{pass_name} producer')

        if self.terminate_event.is_set():
            self._drain_all_queues(write_queue, *queues.values())
            heartbeat_stop_event.set()
            for stage in active_chain:
                for thread in stage_threads[stage]:
                    thread.join(timeout=2.0)
            for writer_thread in writer_threads:
                writer_thread.join(timeout=2.0)
            if heartbeat_thread is not None:
                heartbeat_thread.join(timeout=1.0)

            with stats_lock:
                done_total = int(stats['completed'] + stats['skipped'] + stats['failed'] + stats['cancelled'])

            if total_images > 0:
                local_progress = min(100.0, (done_total / float(total_images)) * 100.0)
            else:
                local_progress = 100.0
            span = float(progress_span[1] - progress_span[0])
            final_scaled = float(progress_span[0]) + (local_progress / 100.0) * span
            final_scaled = max(0.0, min(100.0, final_scaled))
            self._emit(
                {
                    'type': 'progress',
                    'value': float(final_scaled),
                    'eta': f"ETA: {pass_name} {done_total}/{total_images}",
                    'pass': pass_name,
                    'done': done_total,
                    'total': int(total_images),
                },
                level=LogLevel.DEBUG,
            )
            return stats

        if active_chain:
            first_stage = active_chain[0]
            first_in_q = stage_in_queues[first_stage]
            _queue_join_with_diag(first_in_q, f'{pass_name} {first_stage} input')
            for thread in stage_threads[first_stage]:
                _join_thread_with_diag(thread, f'{pass_name} {first_stage} worker')

            for next_idx in range(1, len(active_chain)):
                next_stage = active_chain[next_idx]
                next_in_q = stage_in_queues[next_stage]
                next_worker_count = len(stage_threads[next_stage])

                sentinel_started = time.monotonic()
                sentinels_sent = 0
                for _ in range(next_worker_count):
                    if not self._force_put(
                        next_in_q,
                        None,
                        pass_stop_event=pass_stop_event,
                        queue_name=f'sentinel->{next_stage}',
                        detailed_logs=detailed_logs,
                    ):
                        break
                    sentinels_sent += 1

                if detailed_logs:
                    self._emit(
                        {
                            'type': 'log',
                            'message': (
                                '[*] Sentinel inject => '
                                f'{next_stage} '
                                f'sent:{sentinels_sent}/{next_worker_count} '
                                f'elapsed:{max(0.0, time.monotonic() - sentinel_started):.2f}s '
                                f'q:{_queue_state_text(next_in_q)}'
                            ),
                        },
                        level=LogLevel.DEBUG,
                    )

                _queue_join_with_diag(next_in_q, f'{pass_name} {next_stage} input')
                for thread in stage_threads[next_stage]:
                    _join_thread_with_diag(thread, f'{pass_name} {next_stage} worker')

        if pass_stop_event.is_set() and not self.terminate_event.is_set():
            _queue_join_with_diag(write_queue, f'{pass_name} write (draining completed)')

        writer_sentinel_started = time.monotonic()
        writer_sentinels_sent = 0
        for _ in range(writer_worker_count):
            if not self._force_put(
                write_queue,
                None,
                pass_stop_event=pass_stop_event,
                queue_name='sentinel->write',
                detailed_logs=detailed_logs,
            ):
                break
            writer_sentinels_sent += 1

        if detailed_logs:
            self._emit(
                {
                    'type': 'log',
                    'message': (
                        '[*] Sentinel inject => '
                        f'write '
                        f'sent:{writer_sentinels_sent}/{writer_worker_count} '
                        f'elapsed:{max(0.0, time.monotonic() - writer_sentinel_started):.2f}s '
                        f'q:{_queue_state_text(write_queue)}'
                    ),
                },
                level=LogLevel.DEBUG,
            )

        _queue_join_with_diag(write_queue, f'{pass_name} write')
        for writer_thread in writer_threads:
            _join_thread_with_diag(writer_thread, f'{pass_name} writer')

        heartbeat_stop_event.set()
        if heartbeat_thread is not None:
            _join_thread_with_diag(heartbeat_thread, f'{pass_name} heartbeat')

        if self.terminate_event.is_set() or pass_stop_event.is_set():
            self._drain_all_queues(write_queue, *queues.values())

        with stats_lock:
            done_total = int(stats['completed'] + stats['skipped'] + stats['failed'] + stats['cancelled'])

        if total_images > 0:
            local_progress = min(100.0, (done_total / float(total_images)) * 100.0)
        else:
            local_progress = 100.0
        span = float(progress_span[1] - progress_span[0])
        final_scaled = float(progress_span[0]) + (local_progress / 100.0) * span
        final_scaled = max(0.0, min(100.0, final_scaled))
        self._emit(
            {
                'type': 'progress',
                'value': float(final_scaled),
                'eta': f"ETA: {pass_name} {done_total}/{total_images}",
                'pass': pass_name,
                'done': done_total,
                'total': int(total_images),
            },
            level=LogLevel.DEBUG,
        )

        return stats

    def run_batch(
        self,
        grouped_by_folder,
        output_path,
        overwrite,
        detailed_logs=False,
        transfer_config=None,
    ):
        transfer_config = transfer_config or self.config
        grouped_by_folder = OrderedDict(grouped_by_folder or {})

        total_folders = len(grouped_by_folder)
        total_images = sum(len(images) for images in grouped_by_folder.values())

        batch_stats = {
            'completed': 0,
            'skipped': 0,
            'failed': 0,
            'cancelled': 0,
            'pass1_folders_done': 0,
            'pass2_folders_done': 0,
        }

        pass1_stages = []
        if callable(self.model_factories.get('denoise')) and int(self.instance_counts.get('denoise', 0)) > 0:
            pass1_stages.append('denoise')
        if callable(self.model_factories.get('colorize')) and int(self.instance_counts.get('colorize', 0)) > 0:
            pass1_stages.append('colorize')

        need_pass1 = bool(pass1_stages)
        need_pass2 = bool(callable(self.model_factories.get('upscale')) and int(self.instance_counts.get('upscale', 0)) > 0)
        
        external_enabled = bool(getattr(transfer_config, 'external_color_source_enabled', False))

        if not need_pass1 and not need_pass2 and not external_enabled:
            self._emit({'type': 'log', 'message': '[!] No stages enabled.'}, level=LogLevel.WARNING)
            self._emit_pass_state('complete', 'No stages enabled.')
            self._emit_terminal('complete', 'No stages enabled.')
            return batch_stats

        if need_pass1:
            self._emit_pass_state('pass1', 'Pass 1 started')
            self._emit(
                {
                    'type': 'log',
                    'message': f"[*] === Pass 1: {'+'.join(s.title() for s in pass1_stages)} ({total_folders} folders, {total_images} images) ===",
                },
                level=LogLevel.DEBUG,
            )
            self._create_pools(pass1_stages)
            self._run_stage_warmup_calibration()
            self._emit_calibration_report()

            pass1_progress_span = (0.0, 50.0) if need_pass2 else (0.0, 100.0)
            pass1_stats = self._run_pass(
                pass_name='Pass 1',
                stages=pass1_stages,
                grouped_by_folder=grouped_by_folder,
                output_path=output_path,
                overwrite=overwrite,
                detailed_logs=detailed_logs,
                transfer_config=transfer_config,
                source='input',
                total_images=total_images,
                progress_span=pass1_progress_span,
            )

            for key in ('completed', 'skipped', 'failed', 'cancelled'):
                batch_stats[key] += int(pass1_stats.get(key, 0))
            batch_stats['pass1_folders_done'] = int(pass1_stats.get('folders_done', 0))
            self._unload_pools(pass1_stages)
            gc.collect()
            try:
                from utils.utils import clear_torch_cache as _clear_cache
            except ImportError:
                from Backend.utils.utils import clear_torch_cache as _clear_cache
            _clear_cache()

        if need_pass2 and not self.terminate_event.is_set():
            if batch_stats['failed'] > 0 and self.fail_fast:
                self._emit(
                    {
                        'type': 'log',
                        'message': '[!] Skipping Pass 2 (upscale) due to Pass 1 failures.',
                    },
                    level=LogLevel.WARNING,
                )
            else:
                external_enabled = bool(
                    getattr(transfer_config, 'external_color_source_enabled', False)
                )
                if need_pass1:
                    pass2_source = 'output'
                    pass2_image_count = int(pass1_stats.get('completed', 0)) + int(pass1_stats.get('skipped', 0))
                    if pass2_image_count <= 0:
                        pass2_image_count = self._count_pass1_outputs(grouped_by_folder, output_path)
                else:
                    pass2_source = 'input'
                    pass2_image_count = total_images

                if external_enabled:
                    ext_dir = str(getattr(transfer_config, 'external_color_source_dir', '') or '')
                    self._external_color_map, ext_warnings = build_external_color_map(
                        grouped_by_folder, ext_dir,
                    )
                    for w in ext_warnings:
                        self._emit(
                            {'type': 'log', 'message': f'[!] External color: {w}'},
                            level=LogLevel.WARNING,
                        )
                    matched_count = len(self._external_color_map)
                    self._emit(
                        {
                            'type': 'log',
                            'message': f'[*] External color source: matched {matched_count}/{pass2_image_count} images',
                        },
                        level=LogLevel.DEBUG,
                    )

                pass2_label = 'Upscale + External Color' if external_enabled else 'Upscale'
                self._emit_pass_state('pass2', 'Pass 2 started')
                self._emit(
                    {
                        'type': 'log',
                        'message': f'[*] === Pass 2: {pass2_label} ({pass2_image_count} images) ===',
                    },
                    level=LogLevel.DEBUG,
                )

                self.cuda_baseline_stats = get_cuda_memory_stats_mb()
                self._create_pools(['upscale'])
                self._run_stage_warmup_calibration()
                self._apply_adaptive_upscale_worker_limit()
                self._emit_calibration_report()

                pass2_progress_span = (50.0, 100.0) if need_pass1 else (0.0, 100.0)
                pass2_stats = self._run_pass(
                    pass_name='Pass 2',
                    stages=['upscale'],
                    grouped_by_folder=grouped_by_folder,
                    output_path=output_path,
                    overwrite=True,
                    detailed_logs=detailed_logs,
                    transfer_config=transfer_config,
                    source=pass2_source,
                    total_images=pass2_image_count,
                    progress_span=pass2_progress_span,
                )

                for key in ('completed', 'skipped', 'failed', 'cancelled'):
                    batch_stats[key] += int(pass2_stats.get(key, 0))
                batch_stats['pass2_folders_done'] = int(pass2_stats.get('folders_done', 0))
                self._unload_pools(['upscale'])

        elif not need_pass2 and not self.terminate_event.is_set():
            external_enabled = bool(
                getattr(transfer_config, 'external_color_source_enabled', False)
            )
            if external_enabled:
                ext_dir = str(getattr(transfer_config, 'external_color_source_dir', '') or '')
                self._external_color_map, ext_warnings = build_external_color_map(
                    grouped_by_folder, ext_dir,
                )
                for w in ext_warnings:
                    self._emit(
                        {'type': 'log', 'message': f'[!] External color: {w}'},
                        level=LogLevel.WARNING,
                    )
                matched_count = len(self._external_color_map)
                ct_image_count = total_images if not need_pass1 else (
                    int(pass1_stats.get('completed', 0)) + int(pass1_stats.get('skipped', 0))
                )
                self._emit(
                    {
                        'type': 'log',
                        'message': f'[*] External color source: matched {matched_count}/{ct_image_count} images',
                    },
                    level=LogLevel.DEBUG,
                )

                ct_source = 'output' if need_pass1 else 'input'
                ct_label = 'Color Transfer Only'
                self._emit_pass_state('pass2', 'Color Transfer pass started')
                self._emit(
                    {
                        'type': 'log',
                        'message': f'[*] === {ct_label} ({ct_image_count} images) ===',
                    },
                    level=LogLevel.DEBUG,
                )

                ct_progress_span = (50.0, 100.0) if need_pass1 else (0.0, 100.0)
                ct_stats = self._run_pass(
                    pass_name='Color Transfer',
                    stages=[],
                    grouped_by_folder=grouped_by_folder,
                    output_path=output_path,
                    overwrite=True,
                    detailed_logs=detailed_logs,
                    transfer_config=transfer_config,
                    source=ct_source,
                    total_images=ct_image_count,
                    progress_span=ct_progress_span,
                )

                for key in ('completed', 'skipped', 'failed', 'cancelled'):
                    batch_stats[key] += int(ct_stats.get(key, 0))
                batch_stats['pass2_folders_done'] = int(ct_stats.get('folders_done', 0))

        if self.terminate_event.is_set():
            self._emit_pass_state('complete', 'Batch cancelled')
            self._emit_terminal('cancelled', 'Batch cancelled by user')
        elif batch_stats['failed'] > 0 and self.fail_fast:
            self._emit_pass_state('complete', 'Batch finished with failure policy stop')
            self._emit_terminal('failed', 'Batch stopped by failure policy')
        else:
            self._emit_pass_state('complete', 'Batch complete')
            self._emit_terminal('complete', 'Batch complete')

        self._emit(
            {
                'type': 'progress',
                'value': 100.0,
                'eta': 'ETA: Done!',
                'pass': 'complete',
                'done': int(batch_stats['completed'] + batch_stats['skipped'] + batch_stats['failed'] + batch_stats['cancelled']),
                'total': int(total_images),
            },
            level=LogLevel.DEBUG,
        )
        return batch_stats

    def process_folder(
        self,
        folder_images,
        output_path,
        rel_dir,
        overwrite,
        detailed_logs=False,
        transfer_config=None,
    ):
        grouped = OrderedDict()
        grouped[rel_dir] = list(folder_images or [])
        result = self.run_batch(
            grouped_by_folder=grouped,
            output_path=output_path,
            overwrite=overwrite,
            detailed_logs=detailed_logs,
            transfer_config=transfer_config,
        )
        return {
            'completed': int(result.get('completed', 0)),
            'skipped': int(result.get('skipped', 0)),
            'failed': int(result.get('failed', 0)),
            'cancelled': int(result.get('cancelled', 0)),
            'results': [],
            'stop_requested': bool(self.terminate_event.is_set()),
        }

    def shutdown(self):
        stages_to_drain = list(self.pools.keys())
        if stages_to_drain:
            self._unload_pools(stages_to_drain)
        self.model_factories.clear()
        self.instance_vram_observed_mb.clear()
        self.stage_peak_inference_mb.clear()
