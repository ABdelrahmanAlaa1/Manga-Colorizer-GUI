#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Utility functions for FDAT architecture, adapted from spandrel utilities.
Ported from: thefirst632student/simple_upscaler
"""

import functools
import inspect
import torch
import torch.nn as nn


def store_hyperparameters():
    """
    Decorator that stores constructor hyperparameters in a class attribute.
    Adapted from spandrel.util.store_hyperparameters
    """
    def decorator(cls):
        original_init = cls.__init__

        @functools.wraps(original_init)
        def new_init(self, *args, **kwargs):
            # Get the signature of the original __init__ method
            sig = inspect.signature(original_init)
            # Bind the arguments to parameters
            bound_args = sig.bind(self, *args, **kwargs)
            bound_args.apply_defaults()

            # Store hyperparameters (excluding 'self')
            hyperparams = {k: v for k, v in bound_args.arguments.items() if k != 'self'}

            # Ensure the class has a hyperparameters attribute
            if not hasattr(cls, 'hyperparameters'):
                cls.hyperparameters = {}

            # Store the hyperparameters in the instance
            self.hyperparameters = hyperparams.copy()

            # Call the original __init__
            original_init(self, *args, **kwargs)

        cls.__init__ = new_init
        return cls

    return decorator


class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    Adapted from timm library.
    """
    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'
