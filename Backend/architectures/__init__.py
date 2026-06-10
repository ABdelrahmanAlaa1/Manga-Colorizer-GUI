# Backend/architectures — Custom architecture support
# Architectures not natively in spandrel are registered here.
# Currently: FDAT (Fast Dual Aggregation Transformer)

from .fdat_support import try_load_fdat, is_fdat_architecture

__all__ = ['try_load_fdat', 'is_fdat_architecture']
