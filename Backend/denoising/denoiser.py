"""
Denoise an image with the FFDNet denoising method

Copyright (C) 2018, Matias Tassano <matias.tassano@parisdescartes.fr>

This program is free software: you can use, modify and/or
redistribute it under the terms of the GNU General Public
License as published by the Free Software Foundation, either
version 3 of the License, or (at your option) any later
version. You should have received a copy of this license along
this program. If not, see <http://www.gnu.org/licenses/>.
"""
import contextlib
import os
import argparse
import time


import numpy as np
import cv2
import torch
import torch.nn as nn

from types import SimpleNamespace
from .models import FFDNet
from .utils import normalize, variable_to_cv2_image, remove_dataparallel_wrapper, is_rgb
from precision_utils import (
    apply_tf32,
    has_invalid_output,
    precision_summary,
    resolve_precision,
    FALLBACK_ABORT,
    FALLBACK_DISABLE_AMP,
    FALLBACK_PER_IMAGE,
)
    
class FFDNetDenoiser:
    def __init__(self, _device, _sigma=25, _weights_dir='denoising/models/', _in_ch=3, max_side=None, precision_config=None):
        self.sigma = _sigma / 255
        self.weights_dir = _weights_dir
        self.channels = _in_ch
        self.device = _device
        self.max_side = max_side

        self.config = precision_config if precision_config is not None else SimpleNamespace(device=self.device)
        config_obj = self.config
        self._precision = resolve_precision(config_obj, self.device, stage='denoise')
        apply_tf32(self._precision.get('allow_tf32'))
        self._precision_summary = precision_summary(self._precision)
        self._fallback_mode = self._precision.get('fallback_mode', FALLBACK_PER_IMAGE)
        self._log_callback = getattr(config_obj, 'log_callback', None)
        self._current_image_name = None
        self._fallback_count = 0
        
        self.model = FFDNet(num_input_channels = _in_ch)
        self.load_weights()
        self.model.eval()
       
    
    def load_weights(self):
        weights_name = 'net_rgb.pth' if self.channels == 3 else 'net_gray.pth'
        weights_path = os.path.join(self.weights_dir, weights_name)
        if self.device == 'cuda':
            state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
            device_ids = [0]
            self.model = nn.DataParallel(self.model, device_ids=device_ids).cuda()
        else:
            state_dict = torch.load(weights_path, map_location='cpu')
            # CPU mode: remove the DataParallel wrapper
            state_dict = remove_dataparallel_wrapper(state_dict)
        self.model.load_state_dict(state_dict)
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
        if hasattr(torch, 'compile') and self.device == 'cuda' and getattr(self.config, 'enable_torch_compile', False):
            try:
                self.model = torch.compile(self.model)
            except Exception:
                pass
        
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
        message = f"[!] Denoiser precision fallback to FP32 for {name} ({err})."
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
                raise RuntimeError('NaN/Inf detected in denoiser output')
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

    def get_denoised_image(self, imorig, sigma=None, image_name=None):
        self._current_image_name = image_name
        
        if sigma is not None:
            cur_sigma = sigma / 255
        else:
            cur_sigma = self.sigma 
    
        if len(imorig.shape) < 3 or imorig.shape[2] == 1:
            imorig = np.repeat(np.expand_dims(imorig, 2), 3, 2)
            
        imorig = imorig[..., :3]

        original_shape = imorig.shape[:2]
        resized_for_limit = False
        if self.max_side and max(imorig.shape[0], imorig.shape[1]) > self.max_side:
            ratio = max(imorig.shape[0], imorig.shape[1]) / self.max_side
            imorig = cv2.resize(
                imorig,
                (int(imorig.shape[1] / ratio), int(imorig.shape[0] / ratio)),
                interpolation=cv2.INTER_AREA
            )
            resized_for_limit = True

        imorig = imorig.transpose(2, 0, 1)
 
        if (imorig.max() > 1.2):
            imorig = normalize(imorig)
        imorig = np.expand_dims(imorig, 0)

        # Handle odd sizes
        expanded_h = False
        expanded_w = False
        sh_im = imorig.shape
        if sh_im[2]%2 == 1:
            expanded_h = True
            imorig = np.concatenate((imorig, imorig[:, :, -1, :][:, :, np.newaxis, :]), axis=2)

        if sh_im[3]%2 == 1:
            expanded_w = True
            imorig = np.concatenate((imorig, imorig[:, :, :, -1][:, :, :, np.newaxis]), axis=3)


        imorig = torch.as_tensor(imorig, device=self.device, dtype=torch.float32)
        imnoisy = imorig  # alias — never mutated in-place, clone was pure VRAM waste
        nsigma = torch.tensor([cur_sigma], device=self.device, dtype=torch.float32)

        def run_denoise():
            im_noise_estim = self.model(imnoisy, nsigma)
            return torch.clamp(imnoisy - im_noise_estim, 0.0, 1.0)

        outim = self._run_with_precision(run_denoise)

        if expanded_h:
            imorig = imorig[:, :, :-1, :]
            outim = outim[:, :, :-1, :]

        if expanded_w:
            imorig = imorig[:, :, :, :-1]
            outim = outim[:, :, :, :-1]
        
        result = variable_to_cv2_image(outim)

        if resized_for_limit and result.shape[:2] != original_shape:
            result = cv2.resize(result, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_CUBIC)

        return result

    def precision_summary(self):
        return self._precision_summary
