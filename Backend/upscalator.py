"""
MangaUpscaler — Multi-architecture upscaler backend.

Supports all architectures via spandrel (ESRGAN, SRVGGNet, DAT/DAT2, HAT, SPAN, SPSR)
plus custom FDAT support, and legacy GigaGAN.

Public API (unchanged from original):
    MangaUpscaler(config)
    .upscale(image, scale, image_name=None) -> np.ndarray (uint8 RGB)
    .precision_summary() -> str
"""

import contextlib
import warnings
import torch
import numpy as np
from torch.serialization import SourceChangeWarning

from model_router import ModelRouter
from networks.aura_sr import Upscaler as GigaUpscaler, upscale_4x_overlapped, upscale_4x
from utils.utils import tile_process
from precision_utils import (
    apply_tf32,
    dtype_from_model_name,
    has_invalid_output,
    precision_summary,
    resolve_precision,
    FALLBACK_ABORT,
    FALLBACK_DISABLE_AMP,
    FALLBACK_PER_IMAGE,
)


class MangaUpscaler:
    def __init__(self, config):
        if config.device == 'cuda' and not torch.cuda.is_available():
            print("[-] CUDA not available, using CPU.")
            self.device = 'cpu'
        else:
            self.device = config.device

        self.tile_size = config.upscaler_tile_size
        self.tile_pad = config.tile_pad

        self._precision = resolve_precision(config, self.device, stage='upscale')
        apply_tf32(self._precision.get('allow_tf32'))
        self._precision_summary = precision_summary(self._precision)
        self._fallback_mode = self._precision.get('fallback_mode', FALLBACK_PER_IMAGE)
        self._log_callback = getattr(config, 'log_callback', None)
        self._current_image_name = None
        self._fallback_count = 0

        # Detect architecture type
        upscaler_type = getattr(config, 'upscaler_type', 'Auto-Detect')
        self._is_gigagan = (upscaler_type == 'GigaGAN')
        self.arch_name = 'GigaGAN' if self._is_gigagan else 'Unknown'
        self._detected_scale = 4  # default

        if self._is_gigagan:
            # === GigaGAN path — preserved as-is ===
            self.model = GigaUpscaler().to(self.device)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=SourceChangeWarning)
                model_or_chkpt = torch.load(config.upscaler_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(model_or_chkpt, strict=True)
        else:
            # === Universal auto-detect path ===
            router = ModelRouter()
            model, self._detected_scale, self.arch_name = router.load_model(
                config.upscaler_path, device=self.device
            )
            self.model = model.to(self.device)

        self.model = self.model.eval()

        # --- Force precision FROM the model filename (no needless casts) ---
        # Model names encode their NATIVE dtype, e.g. "..._FDAT_M_..._fp16",
        # "..._FDAT_XL_..._bf16". When present, that tag is the LAW: weights,
        # autocast, and input tensor all match it -> zero fp16<->bf16<->fp32
        # round-trips (memory bandwidth is the bottleneck on this card).
        name_dtype = dtype_from_model_name(getattr(config, 'upscaler_path', None))
        if not self._is_gigagan and name_dtype is not None and self.device == 'cuda':
            self._precision['model_dtype'] = name_dtype
            if name_dtype in (torch.float16, torch.bfloat16):
                self._precision['autocast_dtype'] = name_dtype
                self._precision['autocast_enabled'] = True
                self._precision['resolved_policy'] = (
                    'fp16' if name_dtype == torch.float16 else 'bf16'
                )
            else:  # fp32 model -> no autocast, no casts
                self._precision['autocast_dtype'] = None
                self._precision['autocast_enabled'] = False
                self._precision['resolved_policy'] = 'fp32'
            self._precision_summary = precision_summary(self._precision)
            if callable(self._log_callback):
                tag = 'fp16' if name_dtype == torch.float16 else (
                    'bf16' if name_dtype == torch.bfloat16 else 'fp32')
                self._log_callback(
                    f"[*] upscaler dtype forced from filename => {tag}", level='D')

        # Precision dtype casting — store for single-cast input pipeline
        target_dtype = self._precision.get('model_dtype')
        self._model_dtype = target_dtype if target_dtype is not None else torch.float32
        if target_dtype is not None and target_dtype != torch.float32:
            self.model = self.model.to(dtype=target_dtype)

        # channels_last memory format optimization (Ada Lovelace Tensor Cores)
        if self.device == 'cuda':
            try:
                self.model = self.model.to(memory_format=torch.channels_last)
            except Exception:
                pass

        # PyTorch 2.x torch.compile — operator fusion reduces memory bandwidth pressure
        self._compiled = False
        if hasattr(torch, 'compile') and self.device == 'cuda' and getattr(config, 'enable_torch_compile', False):
            try:
                # max-autotune-no-cudagraphs: best kernel fusion without the
                # cudagraph memory pinning (which fights our shared-weight pool +
                # variable tile batches). dynamic=False since tiles are fixed-size.
                self.model = torch.compile(
                    self.model,
                    mode="max-autotune-no-cudagraphs",
                    dynamic=False,
                    fullgraph=False,
                )
                self._compiled = True
                # Warmup: trigger kernel compilation so first real image isn't slow
                if callable(self._log_callback):
                    self._log_callback("[*] torch.compile: Compiling optimized kernels (first run may take 1-2 min)...", level='D')
                ws = min(self.tile_size + 2 * self.tile_pad, 256) if self.tile_size > 0 else 128
                with torch.inference_mode():
                    with self._autocast_context():
                        dummy = torch.zeros(1, 3, ws, ws, device=self.device, dtype=self._model_dtype)
                        try:
                            dummy = dummy.contiguous(memory_format=torch.channels_last)
                        except Exception:
                            pass
                        _ = self.model(dummy)
                        del dummy
                torch.cuda.empty_cache()
                if callable(self._log_callback):
                    self._log_callback("[*] torch.compile: Kernel compilation complete.", level='D')
            except Exception as e:
                self._compiled = False
                if callable(self._log_callback):
                    self._log_callback(f"[!] torch.compile failed ({e}), continuing without compilation.", level='W')

    @staticmethod
    def _ensure_uint8_rgb(image):
        arr = image
        if isinstance(arr, torch.Tensor):
            arr = arr.detach().cpu().numpy()

        arr = np.asarray(arr)
        if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[-1] != 3:
            arr = np.transpose(arr, (1, 2, 0))

        if arr.ndim != 3:
            raise RuntimeError(f"Unexpected upscaler output shape: {arr.shape}")

        if arr.shape[2] > 3:
            arr = arr[:, :, :3]

        if arr.dtype == np.uint8:
            return arr

        arr = arr.astype(np.float32, copy=False)
        if arr.size > 0 and float(np.max(arr)) <= 1.5:
            arr *= np.float32(255.0)

        np.clip(arr, 0.0, 255.0, out=arr)
        return arr.round().astype(np.uint8)

    def _autocast_context(self):
        if self._precision.get('autocast_enabled') and self.device == 'cuda':
            return torch.autocast(device_type='cuda', dtype=self._precision.get('autocast_dtype'))
        return contextlib.nullcontext()

    def _set_model_dtype(self, dtype):
        if dtype is None:
            return
        self.model = self.model.to(dtype=dtype)

    def _should_fallback(self):
        return self._fallback_mode in {FALLBACK_PER_IMAGE, FALLBACK_DISABLE_AMP}

    def _log_fallback(self, err):
        self._fallback_count += 1
        name = self._current_image_name or 'image'
        message = f"[!] Upscaler precision fallback to FP32 for {name} ({err})."
        if callable(self._log_callback):
            self._log_callback(message, level='W')
        else:
            print(message)

    def _run_with_precision(self, run_fn):
        try:
            with torch.inference_mode():
                with self._autocast_context():
                    result = run_fn()
            if has_invalid_output(result):
                raise RuntimeError('NaN/Inf detected in upscaler output')
            return result
        except Exception as err:
            if not self._should_fallback():
                raise
            # Don't fall back to FP32 for OOM errors — if FP16 can't fit,
            # FP32 (2× VRAM) definitely can't. Only fall back for numerical
            # issues (NaN/Inf, dtype mismatch, etc.). Prevents 400s+ stalls
            # where the recovery ladder retries with even less VRAM headroom.
            err_text = str(err).lower()
            is_oom = isinstance(err, torch.cuda.OutOfMemoryError) or 'out of memory' in err_text
            if is_oom:
                raise
            self._log_fallback(err)
            restore_dtype = None
            target_dtype = self._precision.get('model_dtype')
            # Force the *entire* run (model weights AND input cast inside run_fn)
            # to FP32. run_fn reads self._model_dtype to cast the input tensor,
            # so we must update it here or the retry will hit the same dtype
            # mismatch that triggered the fallback.
            previous_model_dtype = self._model_dtype
            if target_dtype is not None and target_dtype != torch.float32:
                restore_dtype = target_dtype
                self._set_model_dtype(torch.float32)
                self._model_dtype = torch.float32
            try:
                with torch.inference_mode():
                    if self.device == 'cuda':
                        with torch.autocast(device_type='cuda', enabled=False):
                            result = run_fn()
                    else:
                        result = run_fn()
            finally:
                if restore_dtype is not None:
                    if self._fallback_mode == FALLBACK_DISABLE_AMP:
                        # Permanently disable AMP for this upscaler instance.
                        self._precision['autocast_enabled'] = False
                        self._precision['autocast_dtype'] = None
                        self._precision['model_dtype'] = torch.float32
                        self._precision_summary = precision_summary(self._precision)
                        # Keep model + input pipeline in FP32 going forward.
                        self._model_dtype = torch.float32
                    else:
                        # Per-image fallback: restore weights and input dtype
                        # so the next image can attempt mixed precision again.
                        self._set_model_dtype(restore_dtype)
                        self._model_dtype = previous_model_dtype
            if has_invalid_output(result):
                raise RuntimeError('NaN/Inf detected after FP32 fallback')
            return result

    def upscale(self, image, scale, image_name=None):
        self._current_image_name = image_name
        if image.shape[2] == 4:
            image = image[:, :, :3]  # Discard the alpha channel

        def run_upscale():
            if self._is_gigagan:
                return upscale_4x(image, self.model)

            # Universal path — single-cast pipeline
            # For uint8 input: keep uint8 on host (4× less H2D bandwidth than
            # FP32), pin_memory, async transfer, then convert + normalize on
            # the GPU in one fused step. For already-float input, keep the
            # original path.
            np_img = np.ascontiguousarray(image)
            img_tensor = torch.from_numpy(np_img)
            was_uint8 = (img_tensor.dtype == torch.uint8)

            if self.device == 'cuda':
                try:
                    img_tensor = img_tensor.pin_memory()
                except RuntimeError:
                    # Already pinned or pinning unsupported — fall through.
                    pass
                img_tensor = img_tensor.to(device=self.device, non_blocking=True)
                if was_uint8:
                    # uint8 -> model dtype on GPU, then /255 in-place.
                    img_tensor = img_tensor.to(dtype=self._model_dtype).div_(255.0)
                else:
                    img_tensor = img_tensor.to(dtype=self._model_dtype)
            else:
                if was_uint8:
                    img_tensor = img_tensor.to(dtype=self._model_dtype).div_(255.0)
                else:
                    img_tensor = img_tensor.to(dtype=self._model_dtype)

            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0)
            if self.device == 'cuda':
                img_tensor = img_tensor.contiguous(memory_format=torch.channels_last)

            if self.tile_size > 0:
                result = tile_process(self.model, img_tensor, scale, self.tile_size, self.tile_pad, self.arch_name)
            else:
                result = self.model(img_tensor)

            # Free the input tensor BEFORE allocating the FP32 staging copy.
            # On 8GB cards the input + output canvas + uint8 result alive
            # simultaneously is the most common OOM site.
            del img_tensor

            # Quantize to uint8 directly from the model dtype — no FP32 detour.
            # clamp/mul/round/cast all operate natively on BF16/FP16/FP32 and
            # the final uint8 tensor is 4× smaller than the FP32 detour, which
            # halves the peak GPU residency right before the .cpu() copy.
            result = result.data.squeeze(0)
            result = result.clamp_(0, 1).mul_(255.0).round_().to(torch.uint8)
            result = result.cpu().permute(1, 2, 0).contiguous().numpy()
            return result

        result = self._run_with_precision(run_upscale)
        return self._ensure_uint8_rgb(result)

    def precision_summary(self):
        return self._precision_summary
