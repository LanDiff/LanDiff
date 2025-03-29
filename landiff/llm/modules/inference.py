#!/usr/bin/env python
from __future__ import annotations

from torch.nn.parallel import DataParallel, DistributedDataParallel

from landiff.utils import maybe_assign_module_scope


def unwrap_data_parallel(model):
    if isinstance(model, (DistributedDataParallel, DataParallel)):
        model = model.module
    return model


class KVCacheManager:

    def __init__(self, model, prefix=""):
        self.prefix = prefix
        model = unwrap_data_parallel(model)
        maybe_assign_module_scope(model, prefix)

    def get_kvcache(self, module_name):
        return getattr(self, module_name, None)

    def set_kvcache(self, module_name, kvcache):
        setattr(self, module_name, kvcache)

    def __enter__(self):
        global _KVCACHE_MANAGER
        if _KVCACHE_MANAGER is None:
            _KVCACHE_MANAGER = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _KVCACHE_MANAGER
        assert _KVCACHE_MANAGER == self
        _KVCACHE_MANAGER = None


_KVCACHE_MANAGER = None


def get_kvcache_manager() -> KVCacheManager | None:
    """
    Returns:
        The :class:`KVCacheManager` object that's currently being used, or None.
    """
    if _KVCACHE_MANAGER is None:
        return None
    return _KVCACHE_MANAGER
