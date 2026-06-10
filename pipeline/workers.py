import gc
import os
import queue
import time
import threading
import copy
import numpy as np
import PIL.Image
import torch

try:
    from utils.utils import clear_torch_cache, distance_from_grayscale, save_image
except ImportError:
    from Backend.utils.utils import clear_torch_cache, distance_from_grayscale, save_image

from .diagnostics import (
    _queue_state_text,
    sanitize_input_width_limit,
    clamp_image_to_input_width,
    DEFAULT_INPUT_WIDTH,
    get_system_memory_stats_mb,
    QUEUE_RAM_PRESSURE_WARN_MB,
    QUEUE_RAM_HEADROOM_MB,
    LogLevel,
    QUEUE_DIAG_SLOW_STAGE_S,
    QUEUE_DIAG_WRITER_REPORT_INTERVAL_S,
    QUEUE_DIAG_WAIT_WARN_S,
    QUEUE_DIAG_WAIT_REPORT_INTERVAL_S,
)
from .stages import _PipelineAborted
from .transfer import transfer_luminance_from_source


def safe_put(q, item, stop_events, timeout=0.25, diag=None):
    wait_started = time.monotonic()
    last_wait_report = wait_started

    while True:
        for evt in stop_events:
            if evt.is_set():
                if isinstance(diag, dict):
                    on_abort = diag.get('on_abort')
                    if callable(on_abort):
                        try:
                            on_abort(
                                wait_seconds=max(0.0, time.monotonic() - wait_started),
                                queue_state=_queue_state_text(q),
                            )
                        except Exception:
                            pass
                return False

        try:
            q.put(item, timeout=timeout)

            if isinstance(diag, dict):
                on_success = diag.get('on_success')
                if callable(on_success):
                    try:
                        on_success(
                            wait_seconds=max(0.0, time.monotonic() - wait_started),
                            queue_state=_queue_state_text(q),
                        )
                    except Exception:
                        pass

            return True
        except queue.Full:
            if isinstance(diag, dict) and bool(diag.get('enabled', False)):
                now = time.monotonic()
                elapsed = max(0.0, now - wait_started)
                warn_after_s = float(diag.get('warn_after_s', QUEUE_DIAG_WAIT_WARN_S))
                report_interval_s = float(diag.get('report_interval_s', QUEUE_DIAG_WAIT_REPORT_INTERVAL_S))

                if elapsed >= warn_after_s and (now - last_wait_report) >= report_interval_s:
                    on_wait = diag.get('on_wait')
                    if callable(on_wait):
                        try:
                            on_wait(wait_seconds=elapsed, queue_state=_queue_state_text(q))
                        except Exception:
                            pass
                    last_wait_report = now
            continue


def _persistent_producer(
    self,
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
    diag_ctx=None,
):
    first_queue = self._get_first_queue(stages, queues, write_queue=queues.get('__write_queue'))
    total_worker_count = self._get_first_stage_worker_count(stages)
    global_seq = 0

    is_passthrough = len(stages) == 0  # Color Transfer Only mode
    first_stage_name = 'writer' if is_passthrough else (
        'denoise' if 'denoise' in stages else 'colorize' if 'colorize' in stages else 'upscale'
    )
    first_queue_name = f'to_{first_stage_name}'
    producer_item_ref = {'item': '?'}
    producer_wait_diag = self._make_safe_put_diag(
        detailed_logs=detailed_logs,
        pass_stop_event=pass_stop_event,
        owner='producer',
        queue_name=first_queue_name,
        item_ref=producer_item_ref,
    )

    for rel_dir, folder_images in grouped_by_folder.items():
        if not self._wait_while_paused(pass_stop_event):
            break

        if self._is_stop_requested(pass_stop_event):
            break

        folder_name = os.path.basename(os.path.normpath(rel_dir)) if rel_dir else os.path.basename(
            os.path.normpath(getattr(self.config, 'input_folder', '') or output_path)
        )
        display_path = self._elide_path(rel_dir) if rel_dir else folder_name

        scan_started = time.monotonic()

        if source == 'input':
            all_skipped, skip_count, skip_summary = self._check_folder_skip(
                folder_images,
                output_path,
                rel_dir,
                overwrite,
                pass_stop_event=pass_stop_event,
            )
        else:
            all_skipped, skip_count, skip_summary = self._check_upscale_skip(
                folder_images,
                output_path,
                rel_dir,
                transfer_config,
                pass_stop_event=pass_stop_event,
            )

        scan_elapsed = max(0.0, time.monotonic() - scan_started)
        if detailed_logs:
            self._emit(
                {
                    'type': 'log',
                    'message': (
                        '[*] Skip scan => '
                        f'{display_path} '
                        f'files:{len(folder_images)} '
                        f'skipped:{int(skip_count)} '
                        f'all_skipped:{int(bool(all_skipped))} '
                        f'time:{scan_elapsed:.2f}s '
                        f'source:{source}'
                    ),
                },
                level=LogLevel.DEBUG,
            )

        if all_skipped:
            self._emit(
                {
                    'type': 'log',
                    'message': f"[~] {display_path}: {skip_summary}",
                },
                level=LogLevel.DEBUG,
            )
            with stats_lock:
                stats['skipped'] += int(skip_count)
                stats['folders_done'] += 1
            continue

        folder_item_count = 0
        for full_image_path, _rel in folder_images:
            if self.terminate_event.is_set() or pass_stop_event.is_set():
                break

            image_name = os.path.basename(full_image_path)

            if source == 'input':
                out_dir = os.path.join(output_path, rel_dir) if rel_dir else output_path
                out_file = os.path.join(out_dir, image_name)
                if not overwrite and os.path.exists(out_file):
                    with stats_lock:
                        stats['skipped'] += 1
                    continue

            try:
                if source == 'input':
                    with PIL.Image.open(full_image_path) as pil_img:
                        image = np.array(pil_img.convert('RGB'))
                    
                    # Do not clamp input size if we are in Color Transfer Only bypass mode
                    if not is_passthrough:
                        input_limit = sanitize_input_width_limit(
                            getattr(transfer_config, 'input_image_size', DEFAULT_INPUT_WIDTH)
                        )
                        image, _, _, _ = clamp_image_to_input_width(image, input_limit)
                else:
                    out_dir = os.path.join(output_path, rel_dir) if rel_dir else output_path
                    pass1_file = os.path.join(out_dir, image_name)
                    if not os.path.exists(pass1_file):
                        continue

                    # Fast per-image upscale skip: use file size ratio instead of
                    # opening the source image to read dimensions (which causes stalls).
                    try:
                        src_file_size = os.path.getsize(full_image_path)
                        pass1_file_size = os.path.getsize(pass1_file)
                        if src_file_size > 0 and pass1_file_size >= int(src_file_size * 1.5):
                            with stats_lock:
                                stats['skipped'] += 1
                            continue
                    except OSError:
                        pass

                    with PIL.Image.open(pass1_file) as pil_img:
                        image = np.array(pil_img.convert('RGB'))

                skip_colorize = False
                if source == 'input' and 'colorize' in stages:
                    coloredness = distance_from_grayscale(PIL.Image.fromarray(image))
                    if coloredness > 15:
                        skip_colorize = True
                        if not any(s in stages for s in ('denoise', 'upscale')):
                            with stats_lock:
                                stats['skipped'] += 1
                            continue

                created_mono = time.monotonic()

                item = {
                    'global_seq': int(global_seq),
                    'image_name': image_name,
                    'image_path': full_image_path,
                    'image': image,
                    'skip_colorize': bool(skip_colorize),
                    'rel_dir': rel_dir,
                    'output_path': output_path,
                    'started_at': time.time(),
                    'folder_name': display_path,
                    '_created_at_mono': created_mono,
                    '_timing': {},
                }

                # In passthrough mode (no GPU stages), items go directly to writer
                if is_passthrough:
                    item['status'] = 'ready'
                    item['duration'] = 0.0

                producer_item_ref['item'] = f'{display_path}/{image_name}'
                enqueue_started = time.monotonic()
                if not safe_put(first_queue, item, stop_events, diag=producer_wait_diag):
                    break

                enqueued_mono = time.monotonic()
                item['_last_enqueued_at'] = enqueued_mono
                item['_timing']['producer_outbound_wait_s'] = max(0.0, enqueued_mono - enqueue_started)

                if isinstance(diag_ctx, dict) and bool(diag_ctx.get('enabled', False)):
                    inflight_lock = diag_ctx.get('inflight_lock')
                    inflight = diag_ctx.get('inflight')
                    if isinstance(inflight, dict) and hasattr(inflight_lock, 'acquire'):
                        with inflight_lock:
                            inflight[item['global_seq']] = float(item.get('_created_at_mono', enqueued_mono))

                global_seq += 1
                folder_item_count += 1

            except Exception as decode_err:
                self._emit(
                    {
                        'type': 'log',
                        'message': f"[!!!] Failed to decode {image_name}: {decode_err}",
                    },
                    level=LogLevel.ERROR,
                )
                with stats_lock:
                    stats['failed'] += 1
                if self.fail_fast:
                    pass_stop_event.set()
                    break

        if folder_item_count > 0:
            self._emit(
                {
                    'type': 'log',
                    'message': f"[*] {display_path}: queued {folder_item_count} images",
                },
                level=LogLevel.DEBUG,
            )

        with stats_lock:
            stats['folders_done'] += 1

    # In passthrough mode, _run_pass handles writer sentinels directly.
    if not is_passthrough:
        for _ in range(total_worker_count):
            if not self._force_put(
                first_queue,
                None,
                pass_stop_event=pass_stop_event,
                queue_name=f'producer->{first_queue_name}',
                detailed_logs=detailed_logs,
            ):
                break


def _persistent_stage_worker(
    self,
    stage,
    in_q,
    out_q,
    stop_events,
    pass_stop_event,
    transfer_config,
    detailed_logs,
    diag_ctx=None,
):
    stage_item_ref = {'item': '?'}
    # New pass starting for this worker -> re-arm per-pass precision so the
    # selected model's (possibly changed) dtype is applied globally on first item.
    if stage in {'denoise', 'colorize'}:
        self._pass1_precision_set = False
    if stage == 'upscale':
        self._upscale_pass_precision_set = False
        self._upscale_since_flush = 0
    stage_out_diag = self._make_safe_put_diag(
        detailed_logs=detailed_logs,
        pass_stop_event=pass_stop_event,
        owner=f'{stage}-worker',
        queue_name=f'{stage}-out',
        item_ref=stage_item_ref,
    )
    stage_retry_diag = self._make_safe_put_diag(
        detailed_logs=detailed_logs,
        pass_stop_event=pass_stop_event,
        owner=f'{stage}-retry',
        queue_name=f'{stage}-in',
        item_ref=stage_item_ref,
    )

    while True:
        if not self._wait_while_paused(pass_stop_event):
            break

        if self._is_stop_requested(pass_stop_event):
            break

        try:
            item = in_q.get(timeout=0.5)
        except queue.Empty:
            if self._is_stop_requested(pass_stop_event):
                break
            continue

        if item is None:
            in_q.task_done()
            break

        stage_recv_mono = time.monotonic()
        stage_item_ref['item'] = str(item.get('image_name', '?'))
        timing = item.get('_timing') if isinstance(item, dict) else None
        if not isinstance(timing, dict):
            timing = {}
            if isinstance(item, dict):
                item['_timing'] = timing

        last_enqueued_at = float(item.get('_last_enqueued_at', stage_recv_mono))
        timing[f'{stage}_intake_wait_s'] = max(0.0, stage_recv_mono - last_enqueued_at)
        stage_compute_started = stage_recv_mono

        try:
            if stage == 'denoise' and 'denoise' in self.pools:
                with torch.inference_mode():
                    with self.pools['denoise'].checkout(stop_events=stop_events) as denoiser:
                        # Pass 1 owns global precision (denoise+colorize). Set it
                        # once per worker so pass1 math matches its model dtype.
                        if not getattr(self, '_pass1_precision_set', False):
                            try:
                                from precision_utils import set_active_pass_precision
                                set_active_pass_precision(
                                    getattr(denoiser, '_model_dtype', None),
                                    log=getattr(transfer_config, 'log_callback', None),
                                )
                            except Exception:
                                pass
                            self._pass1_precision_set = True
                        sigma = getattr(transfer_config, 'denoise_sigma', 25)
                        item['image'] = denoiser.denoise(
                            item['image'],
                            sigma,
                            image_name=item.get('image_name'),
                        )

            elif stage == 'colorize' and 'colorize' in self.pools:
                if not item.get('skip_colorize', False):
                    working_np = item['image']
                    luminance_source = working_np

                    target_width = int(getattr(transfer_config, 'colorized_image_size', 576))
                    if bool(getattr(transfer_config, 'force_safe_colorizer_width', False)):
                        target_width = 576

                    original_width = working_np.shape[1]
                    effective_width = min(original_width, target_width)
                    adjusted_width = effective_width - (effective_width % 32)
                    if adjusted_width == 0:
                        adjusted_width = 32

                    with torch.inference_mode():
                        with self.pools['colorize'].checkout(stop_events=stop_events) as colorizer:
                            if not getattr(self, '_pass1_precision_set', False):
                                try:
                                    from precision_utils import set_active_pass_precision
                                    set_active_pass_precision(
                                        getattr(colorizer, '_model_dtype', None),
                                        log=getattr(transfer_config, 'log_callback', None),
                                    )
                                except Exception:
                                    pass
                                self._pass1_precision_set = True
                            colorizer.set_image(
                                working_np,
                                adjusted_width,
                                image_name=item.get('image_name'),
                            )
                            colorized_output = colorizer.colorize()

                    # Stash luminance source and colorized output for postprocess stage
                    item['_luminance_source'] = luminance_source
                    item['_colorized_output'] = colorized_output
                    item['image'] = colorized_output  # pass through for non-postprocess paths
                    del working_np

            elif stage == 'postprocess':
                # CPU-only stage: OCR detection + luminance transfer
                luminance_source = item.pop('_luminance_source', None)
                colorized_output = item.pop('_colorized_output', None)

                if luminance_source is not None and colorized_output is not None:
                    import copy as _copy_mod
                    local_transfer_config = _copy_mod.copy(transfer_config)
                    local_transfer_config.current_image_path = item.get('image_path')

                    item['image'] = transfer_luminance_from_source(
                        luminance_source,
                        colorized_output,
                        local_transfer_config,
                    )
                    del luminance_source, colorized_output

            elif stage == 'upscale' and 'upscale' in self.pools:
                with torch.inference_mode():
                    with self.pools['upscale'].checkout(stop_events=stop_events) as upscaler:
                        # Pass 2 owns global precision. Set it ONCE per worker to
                        # the upscaler's (filename-derived) dtype: all math matches
                        # the model -> min casts, min VRAM. TF32 stays off (no fp32
                        # math left to accelerate).
                        if not getattr(self, '_upscale_pass_precision_set', False):
                            try:
                                from precision_utils import set_active_pass_precision
                                set_active_pass_precision(
                                    getattr(upscaler, '_model_dtype', None),
                                    log=getattr(transfer_config, 'log_callback', None),
                                )
                            except Exception:
                                pass
                            self._upscale_pass_precision_set = True
                        upscale_factor = getattr(transfer_config, 'upscale_factor', 4)
                        item['image'] = upscaler.upscale(
                            item['image'],
                            upscale_factor,
                            image_name=item.get('image_name'),
                        )
                # No periodic cache flush. Weight sharing + single active pass
                # keeps VRAM stable. empty_cache() churns the allocator and
                # causes stalls on subsequent images. OOM-triggered flushes
                # in tile_process handle emergency cases.

            item.pop('_retried_memory', None)
            item['status'] = 'ready'
            item['duration'] = max(0.0, time.monotonic() - stage_compute_started)
            stage_compute_done = time.monotonic()
            timing[f'{stage}_compute_s'] = max(0.0, stage_compute_done - stage_compute_started)

            out_enqueue_started = time.monotonic()
            if not safe_put(out_q, item, (self.terminate_event,), diag=stage_out_diag):
                pass_stop_event.set()
            else:
                out_enqueued_at = time.monotonic()
                item['_last_enqueued_at'] = out_enqueued_at
                timing[f'{stage}_outbound_wait_s'] = max(0.0, out_enqueued_at - out_enqueue_started)

                if detailed_logs:
                    stage_intake_s = float(timing.get(f'{stage}_intake_wait_s', 0.0) or 0.0)
                    stage_compute_s = float(timing.get(f'{stage}_compute_s', 0.0) or 0.0)
                    stage_outbound_s = float(timing.get(f'{stage}_outbound_wait_s', 0.0) or 0.0)
                    if max(stage_intake_s, stage_compute_s, stage_outbound_s) >= QUEUE_DIAG_SLOW_STAGE_S:
                        self._emit(
                            {
                                'type': 'log',
                                'message': (
                                    '[*] Stage latency => '
                                    f'{stage} '
                                    f'item:{item.get("image_name", "?")} '
                                    f'intake:{stage_intake_s:.2f}s '
                                    f'compute:{stage_compute_s:.2f}s '
                                    f'outbound:{stage_outbound_s:.2f}s '
                                    f'in_q:{_queue_state_text(in_q)} '
                                    f'out_q:{_queue_state_text(out_q)}'
                                ),
                            },
                            level=LogLevel.DEBUG,
                        )

        except _PipelineAborted:
            item['image'] = None
            pass_stop_event.set()
            break
        except Exception as err:
            err_text = str(err)
            is_memory_error = isinstance(err, MemoryError) or ('Unable to allocate' in err_text)
            if is_memory_error and not bool(item.get('_retried_memory', False)):
                item['_retried_memory'] = True
                self._emit(
                    {
                        'type': 'log',
                        'message': f"[!] {stage} hit host-memory pressure for {item.get('image_name', '?')}; retrying once after GC.",
                    },
                    level=LogLevel.WARNING,
                )
                gc.collect()
                clear_torch_cache()
                # Cooldown after OOM to let other workers finish their current
                # items before this retry re-enters the GPU. Prevents concurrent
                # OOM → GC storms from multiple workers.
                time.sleep(0.5)
                if item.get('image') is not None:
                    stage_item_ref['item'] = str(item.get('image_name', '?'))
                    retry_started = time.monotonic()
                    if safe_put(in_q, item, stop_events, diag=stage_retry_diag):
                        retry_enqueued = time.monotonic()
                        item['_last_enqueued_at'] = retry_enqueued
                        timing[f'{stage}_retry_wait_s'] = max(0.0, retry_enqueued - retry_started)
                        continue
                    pass_stop_event.set()
                else:
                    self._emit(
                        {
                            'type': 'log',
                            'message': (
                                f"[!] {stage} retry skipped for {item.get('image_name', '?')}: "
                                "image data freed by GC."
                            ),
                        },
                        level=LogLevel.WARNING,
                    )

            self._emit(
                {
                    'type': 'log',
                    'message': f"[!!!] {stage} failed for {item.get('image_name', '?')}: {err_text}",
                },
                level=LogLevel.ERROR,
            )
            item['image'] = None
            item['status'] = 'failed'
            item['error'] = err_text
            item['duration'] = max(0.0, time.monotonic() - stage_compute_started)
            stage_item_ref['item'] = str(item.get('image_name', '?'))
            fail_enqueue_started = time.monotonic()
            if safe_put(out_q, item, (self.terminate_event,), diag=stage_out_diag):
                fail_enqueued = time.monotonic()
                item['_last_enqueued_at'] = fail_enqueued
                timing[f'{stage}_outbound_wait_s'] = max(0.0, fail_enqueued - fail_enqueue_started)
            if self.fail_fast:
                pass_stop_event.set()
        finally:
            in_q.task_done()


def _persistent_writer(
    self,
    write_queue,
    output_path,
    stop_events,
    stats,
    stats_lock,
    total_images,
    pass_name,
    detailed_logs,
    progress_start,
    progress_end,
    diag_ctx=None,
    transfer_config=None,
):
    def _writer_stop_requested():
        return bool(self.terminate_event.is_set())

    while True:
        while self.pause_event.is_set() and not _writer_stop_requested():
            time.sleep(0.2)

        if _writer_stop_requested() and write_queue.empty():
            break

        try:
            item = write_queue.get(timeout=0.5)
        except queue.Empty:
            if _writer_stop_requested():
                break
            continue

        if item is None:
            write_queue.task_done()
            break

        writer_recv_mono = time.monotonic()
        try:
            status = item.get('status', 'failed')
            image_name = item.get('image_name', 'unknown')
            rel_dir = item.get('rel_dir', '')
            duration = float(item.get('duration', 0.0))
            folder_name = item.get('folder_name', '')
            timing = item.get('_timing') if isinstance(item, dict) else None
            if not isinstance(timing, dict):
                timing = {}
                if isinstance(item, dict):
                    item['_timing'] = timing

            last_enqueued_at = float(item.get('_last_enqueued_at', writer_recv_mono))
            writer_queue_wait_s = max(0.0, writer_recv_mono - last_enqueued_at)
            timing['writer_queue_wait_s'] = writer_queue_wait_s
            write_io_s = 0.0

            if status == 'ready':
                # Apply external color transfer if enabled
                ext_enabled = bool(
                    getattr(transfer_config, 'external_color_source_enabled', False)
                ) if transfer_config else False
                if ext_enabled and item['image'] is not None:
                    ext_key = (rel_dir, image_name)
                    ext_path = self._external_color_map.get(ext_key)
                    if ext_path and os.path.exists(ext_path):
                        try:
                            ext_img = np.array(PIL.Image.open(ext_path).convert('RGB'))
                            ext_cfg = copy.copy(transfer_config)
                            ext_cfg.current_image_path = item.get('image_path')
                            ext_cfg.edge_chroma_protection = False
                            ext_cfg.line_ink_protection = False
                            ext_cfg.screentone_chroma_smoothing = False
                            item['image'] = transfer_luminance_from_source(
                                item['image'], ext_img, ext_cfg,
                            )
                            del ext_img
                        except Exception as ext_err:
                            self._emit(
                                {
                                    'type': 'log',
                                    'message': f"[!] External color transfer failed for {image_name}: {ext_err}",
                                },
                                level=LogLevel.WARNING,
                            )
                    elif ext_path is None:
                        self._emit(
                            {
                                'type': 'log',
                                'message': f"[!] No external color match for {image_name}, saving without color transfer",
                            },
                            level=LogLevel.WARNING,
                        )

                write_started_mono = time.monotonic()
                out_dir = os.path.join(output_path, rel_dir) if rel_dir else output_path
                os.makedirs(out_dir, exist_ok=True)
                save_image(item['image'], os.path.join(out_dir, image_name))
                write_io_s = max(0.0, time.monotonic() - write_started_mono)
                timing['writer_io_s'] = write_io_s
                item['image'] = None
                with stats_lock:
                    stats['completed'] += 1

                if detailed_logs:
                    self._emit(
                        {
                            'type': 'log',
                            'message': f"[+] {folder_name}/{image_name} done in {duration:.2f}s",
                        },
                        level=LogLevel.VERBOSE,
                    )

            elif status == 'failed':
                with stats_lock:
                    stats['failed'] += 1
                error_text = item.get('error', 'Unknown')
                self._emit(
                    {
                        'type': 'log',
                        'message': f"[!!!] FAILED {image_name}: {error_text}",
                    },
                    level=LogLevel.ERROR,
                )
            elif status == 'skipped':
                with stats_lock:
                    stats['skipped'] += 1
            else:
                with stats_lock:
                    stats['cancelled'] += 1

            with stats_lock:
                done_total = int(stats['completed'] + stats['skipped'] + stats['failed'] + stats['cancelled'])

            end_to_end_s = max(0.0, time.monotonic() - float(item.get('_created_at_mono', writer_recv_mono)))
            timing['end_to_end_s'] = end_to_end_s

            if detailed_logs and max(writer_queue_wait_s, write_io_s, end_to_end_s) >= QUEUE_DIAG_SLOW_STAGE_S:
                self._emit(
                    {
                        'type': 'log',
                        'message': (
                            '[*] Item latency => '
                            f'{folder_name}/{image_name} '
                            f'writer_q:{writer_queue_wait_s:.2f}s '
                            f'write_io:{write_io_s:.2f}s '
                            f'e2e:{end_to_end_s:.2f}s '
                            f'write_q:{_queue_state_text(write_queue)}'
                        ),
                    },
                    level=LogLevel.DEBUG,
                )

            if isinstance(diag_ctx, dict) and bool(diag_ctx.get('enabled', False)):
                inflight = diag_ctx.get('inflight')
                inflight_lock = diag_ctx.get('inflight_lock')
                seq = item.get('global_seq')
                if isinstance(inflight, dict) and hasattr(inflight_lock, 'acquire') and seq is not None:
                    with inflight_lock:
                        inflight.pop(int(seq), None)

                writer_lock = diag_ctx.get('writer_lock')
                if hasattr(writer_lock, 'acquire'):
                    with writer_lock:
                        now = time.monotonic()
                        diag_ctx['last_progress_at'] = now

                        if status == 'ready':
                            writer_samples_ms = diag_ctx.setdefault('writer_samples_ms', [])
                            writer_samples_ms.append(float(write_io_s * 1000.0))
                            if len(writer_samples_ms) > 256:
                                del writer_samples_ms[:-256]

                        last_report_at = float(diag_ctx.get('writer_last_report_at', now))
                        if detailed_logs and (now - last_report_at) >= QUEUE_DIAG_WRITER_REPORT_INTERVAL_S:
                            interval_s = max(0.001, now - last_report_at)
                            previous_done = int(diag_ctx.get('writer_last_done', 0) or 0)
                            delta_done = max(0, int(done_total) - previous_done)
                            done_rate = float(delta_done) / interval_s
                            avg_ms, p95_ms, max_ms = self._latency_summary_ms(diag_ctx.get('writer_samples_ms', []))

                            self._emit(
                                {
                                    'type': 'log',
                                    'message': (
                                        '[*] Writer stats => '
                                        f'rate:{done_rate:.2f}/s '
                                        f'avg:{avg_ms:.1f}ms '
                                        f'p95:{p95_ms:.1f}ms '
                                        f'max:{max_ms:.1f}ms '
                                        f'write_q:{_queue_state_text(write_queue)} '
                                        f'done:{done_total}/{total_images}'
                                    ),
                                },
                                level=LogLevel.DEBUG,
                            )
                            diag_ctx['writer_last_report_at'] = now
                            diag_ctx['writer_last_done'] = int(done_total)

            if total_images > 0:
                local_progress = min(100.0, (done_total / float(total_images)) * 100.0)
            else:
                local_progress = 100.0

            span = float(progress_end - progress_start)
            scaled_progress = float(progress_start) + (local_progress / 100.0) * span
            scaled_progress = max(0.0, min(100.0, scaled_progress))

            self._emit(
                {
                    'type': 'progress',
                    'value': float(scaled_progress),
                    'eta': f"ETA: {pass_name} {done_total}/{total_images}",
                    'pass': pass_name,
                    'done': done_total,
                    'total': int(total_images),
                },
                level=LogLevel.DEBUG,
            )

            self._emit(
                {
                    'type': 'log_update',
                    'message': f"[*] {pass_name}: {folder_name} | {done_total}/{total_images} done",
                },
                level=LogLevel.DEBUG,
            )

        except Exception as write_err:
            with stats_lock:
                stats['failed'] += 1

            if isinstance(diag_ctx, dict) and bool(diag_ctx.get('enabled', False)):
                inflight = diag_ctx.get('inflight')
                inflight_lock = diag_ctx.get('inflight_lock')
                seq = item.get('global_seq') if isinstance(item, dict) else None
                if isinstance(inflight, dict) and hasattr(inflight_lock, 'acquire') and seq is not None:
                    with inflight_lock:
                        inflight.pop(int(seq), None)

            self._emit(
                {
                    'type': 'log',
                    'message': f"[!!!] Writer error: {write_err}",
                },
                level=LogLevel.ERROR,
            )
        finally:
            write_queue.task_done()
