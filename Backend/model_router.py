#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Model loading and architecture auto-detection.

Uses spandrel for 30+ architectures (ESRGAN, SRVGGNet, DAT, HAT, SPAN, SPSR, etc.)
with custom FDAT support via Backend/architectures/.

This replaces the hardcoded RRDBNet-only loading in the old upscalator.py.
"""

import warnings
import torch
import safetensors.torch
from pathlib import Path
from torch.serialization import SourceChangeWarning

try:
    from spandrel import ModelLoader, ImageModelDescriptor
    HAS_SPANDREL = True
except ImportError:
    HAS_SPANDREL = False
    print("[!] spandrel not installed. Only FDAT and legacy ESRGAN models will be supported.")
    print("[!] Install with: pip install spandrel>=0.4.0")

try:
    from architectures.fdat_support import try_load_fdat, is_fdat_architecture
except ImportError:
    from Backend.architectures.fdat_support import try_load_fdat, is_fdat_architecture


class ModelRouter:
    """Auto-detect model architecture and load from file.
    
    Loading priority:
    1. Try FDAT detection first (not in spandrel)
    2. Try spandrel (covers ESRGAN, SRVGGNet, DAT, HAT, SPAN, SPSR, etc.)
    3. Fallback: legacy .pt full-model files
    """

    def __init__(self):
        if HAS_SPANDREL:
            self._loader = ModelLoader()
        else:
            self._loader = None

    def load_model(self, model_path: str, device: str = 'cpu'):
        """Load any supported model file, auto-detecting architecture.
        
        Args:
            model_path: Path to .pth, .pt, or .safetensors model file
            device: Target device ('cpu' or 'cuda:N')
            
        Returns:
            tuple: (model: nn.Module, scale: int, arch_name: str)
            
        Raises:
            RuntimeError: If the model cannot be loaded by any method
        """
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Step 1: Load state_dict
        state_dict = self._load_state_dict(path)

        # Step 2: Try FDAT first (not in spandrel mainline)
        if state_dict is not None and is_fdat_architecture(state_dict):
            model, scale = try_load_fdat(state_dict)
            if model is not None:
                return model, scale, "FDAT"

        # Step 3: Try spandrel (covers 30+ architectures)
        if HAS_SPANDREL:
            try:
                descriptor = self._loader.load_from_file(str(path))
                if isinstance(descriptor, ImageModelDescriptor):
                    arch_name = descriptor.architecture.name
                    return descriptor.model, descriptor.scale, arch_name
            except Exception as spandrel_err:
                # spandrel couldn't handle it — fall through to legacy
                pass

        # Step 4: Legacy .pt full-model files (e.g. RealESRGAN_x4plus_anime_6B.pt)
        return self._load_legacy_pt(path, device)

    def _load_state_dict(self, path: Path):
        """Load state_dict from .pth, .pt, or .safetensors file.
        
        Returns the raw state_dict dict, or None if the file is a full serialized model.
        """
        try:
            if path.suffix == '.safetensors':
                return safetensors.torch.load_file(str(path))

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=SourceChangeWarning)
                data = torch.load(str(path), map_location='cpu', weights_only=False)

            if isinstance(data, dict):
                # Check for common wrapper keys
                for key in ('params_ema', 'params', 'state_dict', 'model'):
                    if key in data:
                        return data[key]
                # If it has tensor values, it's likely a raw state_dict
                first_val = next(iter(data.values()), None)
                if isinstance(first_val, torch.Tensor):
                    return data
            
            # Full serialized model — return None so we fall through to legacy loading
            return None
            
        except Exception:
            return None

    def _load_legacy_pt(self, path: Path, device: str):
        """Handle .pt files with full serialized models.
        
        These are legacy files like RealESRGAN_x4plus_anime_6B.pt where
        the entire nn.Module was serialized with torch.save().
        """
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=SourceChangeWarning)
            model = torch.load(str(path), map_location=device, weights_only=False)

        if hasattr(model, 'generator'):
            # Wrapper model with .generator attribute (our old Upscaler class)
            return model.generator, 4, "ESRGAN-Legacy"
        elif isinstance(model, torch.nn.Module):
            return model, 4, "Unknown-Legacy"
        else:
            raise RuntimeError(
                f"Cannot load model from {path}. "
                f"File contains {type(model).__name__}, expected nn.Module or state_dict."
            )

    @staticmethod
    def detect_content_type(image) -> str:
        """Classify image content as bw_manga or color_illust.
        
        Uses the existing distance_from_grayscale utility.
        """
        import PIL.Image
        try:
            from utils.utils import distance_from_grayscale
        except ImportError:
            from Backend.utils.utils import distance_from_grayscale

        if isinstance(image, PIL.Image.Image):
            pil_img = image
        else:
            pil_img = PIL.Image.fromarray(image)
        
        coloredness = distance_from_grayscale(pil_img)
        return 'bw_manga' if coloredness <= 15 else 'color_illust'

    @staticmethod
    def suggest_models(content_type: str) -> list:
        """Suggest optimal models for a given content type."""
        CATALOG = {
            'bw_manga': [
                'MangaJaNai (DAT2 — best quality)',
                '4x_eula_digimanga_bw_v2 (ESRGAN — fast)',
            ],
            'color_illust': [
                '4x_IllustrationJaNai_V3detail_DAT2 (best quality, 15s)',
                '4x_IllustrationJaNai_V3detail_FDAT_M (good quality, 1.8s)',
                '4x_IllustrationJaNai_V3denoise_DAT2 (halftone cleaning, 15s)',
                '2x_IllustrationJaNai_V3detail_SPAN_S (fastest 2x, 0.08s)',
            ],
            'general': [
                '4x-UltraSharp (photos)',
                '4x-AnimeSharp (anime)',
                'RealESRGAN_x4plus (general)',
            ],
        }
        return CATALOG.get(content_type, CATALOG['general'])
