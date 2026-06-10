import contextlib
import torch
from torchvision.transforms import ToTensor
import numpy as np
import os

from networks.colorizer import Colorizer
from denoising.denoiser import FFDNetDenoiser
from utils.utils import resize_pad, tile_process
from precision_utils import (
    apply_tf32,
    has_invalid_output,
    precision_summary,
    resolve_precision,
    FALLBACK_ABORT,
    FALLBACK_DISABLE_AMP,
    FALLBACK_PER_IMAGE,
)


class MangaColorizator:
    def __init__(self, config):
        if config.device == 'cuda' and not torch.cuda.is_available():
            print("[-] CUDA not available, using CPU.")
            self.device = 'cpu'
        else:
            self.device = config.device

        self.tile_size = config.colorizer_tile_size
        self.tile_pad = config.tile_pad

        self._precision = resolve_precision(config, self.device, stage='colorize')
        apply_tf32(self._precision.get('allow_tf32'))
        self._precision_summary = precision_summary(self._precision)
        self._fallback_mode = self._precision.get('fallback_mode', FALLBACK_PER_IMAGE)
        self._log_callback = getattr(config, 'log_callback', None)
        self._current_image_name = None
        self._fallback_count = 0

        self.model = Colorizer().to(self.device)
        state_dict = torch.load(config.colorizer_path, map_location=self.device)
        self.model.generator.load_state_dict(state_dict)
        self.model = self.model.eval()
        target_dtype = self._precision.get('model_dtype')
        if target_dtype is not None and target_dtype != torch.float32:
            self.model = self.model.to(dtype=target_dtype)

        # Apply channels_last memory format optimization
        if self.device == 'cuda':
            try:
                self.model = self.model.to(memory_format=torch.channels_last)
            except Exception:
                pass

        # PyTorch 2.0+ torch.compile optimization
        if hasattr(torch, 'compile') and self.device == 'cuda' and getattr(config, 'enable_torch_compile', False):
            try:
                self.model.generator = torch.compile(self.model.generator)
            except Exception:
                pass

        weights_dir = getattr(config, '_denoiser_weights_dir', None)
        if not weights_dir:
            weights_dir = getattr(config, 'denoiser_weights_dir', None)
        if not weights_dir:
            colorizer_path = getattr(config, 'colorizer_path', '')
            inferred_dir = os.path.dirname(os.path.abspath(colorizer_path)) if colorizer_path else ''
            if inferred_dir and os.path.isdir(inferred_dir):
                weights_dir = inferred_dir
            else:
                weights_dir = os.path.join(os.path.dirname(__file__), 'networks')

        self.denoiser = FFDNetDenoiser(
            self.device,
            _weights_dir=os.path.abspath(weights_dir),
            precision_config=config,
        )
        
        self.current_image = None
        self.current_hint = None
        self.current_pad = None

        self.scale = 1
        

    def set_image(self, image, size=576, transform=ToTensor(), image_name=None):
        if size % 32 != 0:
            raise RuntimeError("[-] Size is not divisible by 32")
        
        image, self.current_pad = resize_pad(image, size)
        self.current_image = transform(image).unsqueeze(0).to(self.device)
        # Cast input to model precision (FP16/BF16) — ToTensor() always returns FP32
        # but with cast_weights=True the model expects matching dtype.
        target_dtype = self._precision.get('model_dtype')
        if target_dtype is not None and target_dtype != torch.float32:
            self.current_image = self.current_image.to(dtype=target_dtype)
        self.current_hint = torch.zeros(1, 4, self.current_image.shape[2], self.current_image.shape[3],
                                        device=self.device, dtype=self.current_image.dtype)
        self._current_image_name = image_name
    
    def update_hint(self, hint, mask):
        if issubclass(hint.dtype.type, np.integer):
            hint = hint.astype('float32') / 255
            
        hint = (hint - 0.5) / 0.5
        hint = torch.FloatTensor(hint).permute(2, 0, 1)
        mask = torch.FloatTensor(np.expand_dims(mask, 0))

        self.current_hint = torch.cat([hint * mask, mask], 0).unsqueeze(0).to(self.device)

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
        message = f"[!] Colorizer precision fallback to FP32 for {name} ({err})."
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
                raise RuntimeError('NaN/Inf detected in colorizer output')
            return result
        except Exception as err:
            if not self._should_fallback():
                raise
            self._log_fallback(err)
            restore_dtype = None
            target_dtype = self._precision.get('model_dtype')
            if target_dtype is not None and target_dtype != torch.float32:
                restore_dtype = target_dtype
                self._set_model_dtype(torch.float32)
            with torch.inference_mode():
                if self.device == 'cuda':
                    with torch.autocast(device_type='cuda', enabled=False):
                        result = run_fn()
                else:
                    result = run_fn()
            if restore_dtype is not None:
                if self._fallback_mode == FALLBACK_DISABLE_AMP:
                    self._precision['autocast_enabled'] = False
                    self._precision['model_dtype'] = torch.float32
                    self._precision_summary = precision_summary(self._precision)
                else:
                    self._set_model_dtype(restore_dtype)
            if has_invalid_output(result):
                raise RuntimeError('NaN/Inf detected after FP32 fallback')
            return result

    def colorize(self):
        def run_colorize():
            img = torch.cat([self.current_image, self.current_hint], 1)

            if self.tile_size > 0:
                fake_color = tile_process(self.model, img, self.scale, self.tile_size, self.tile_pad, 'Colorizer')
            else:
                fake_color, _ = self.model(img)
            result = fake_color[0].detach().permute(1, 2, 0) * 0.5 + 0.5
            del fake_color  # free full model output before pad-trim + CPU transfer
            if self.current_pad[0] != 0:
                result = result[:-self.current_pad[0]]
            if self.current_pad[1] != 0:
                result = result[:, :-self.current_pad[1]]
            return result

        result = self._run_with_precision(run_colorize)
        # Free GPU input tensors — model weights stay, only per-image data freed
        self.current_image = None
        self.current_hint = None
        return (result.detach().cpu().numpy() * 255.0).round().astype(np.uint8)

    def precision_summary(self):
        return self._precision_summary