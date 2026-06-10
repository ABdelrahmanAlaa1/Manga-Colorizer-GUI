#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FDAT architecture support for model_router.
Provides detection and loading for FDAT models that spandrel doesn't natively support.
"""

import torch
from pathlib import Path


def is_fdat_architecture(state_dict: dict) -> bool:
    """Detect if a state_dict belongs to an FDAT model.
    
    FDAT models have these signature keys:
    - 'conv_first.weight' (initial convolution)
    - 'groups.N.blocks.N.n1.weight' (transformer block layer norms)
    - 'groups.N.conv.weight' (group-level convolution)
    """
    keys = set(state_dict.keys())
    has_conv_first = 'conv_first.weight' in keys
    has_groups = any(k.startswith('groups.') and '.blocks.' in k for k in keys)
    has_conv_after = 'conv_after.weight' in keys
    
    # Distinguish from DAT: FDAT uses 'groups.N.blocks.N.n1.weight' (LayerNorm)
    # while DAT uses 'layers.N.blocks.N' pattern
    has_fdat_pattern = any(
        k.startswith('groups.') and '.blocks.' in k and '.n1.weight' in k 
        for k in keys
    )
    
    return has_conv_first and has_groups and has_conv_after and has_fdat_pattern


def try_load_fdat(state_dict: dict):
    """Try to load a state_dict as an FDAT model.
    
    Returns:
        (model, scale) if successful, (None, None) if not an FDAT model
    """
    if not is_fdat_architecture(state_dict):
        return None, None
    
    try:
        from .FDAT import FDATNet
        model = FDATNet(state_dict)
        scale = model.scale
        return model, scale
    except Exception as e:
        print(f"[!] FDAT loading failed: {e}")
        return None, None
