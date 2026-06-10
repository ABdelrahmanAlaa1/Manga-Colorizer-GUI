import queue
from contextlib import contextmanager

class _PipelineAborted(Exception):
    pass


# Architectures verified safe for weight/buffer sharing (eval + no_grad only)
SHARED_WEIGHT_SAFE_ARCHS = {
    'ESRGAN', 'SRVGGNet', 'SPSR', 'SPAN', 'SPAN-S',
    'DAT', 'DAT-2', 'HAT', 'HAT-L',
    'FDAT', 'FDAT-M', 'FDAT-XL',
}


class ModelInstancePool:
    def __init__(self, model_factory, count, stage_name='stage',
                 share_weights=True, arch_name=None):
        self.stage_name = stage_name
        self.count = int(count)
        self._pool = queue.Queue()
        self.precision_summary = ''

        master = model_factory()
        summary = getattr(master, 'precision_summary', None)
        if callable(summary):
            summary = summary()
        if isinstance(summary, str):
            self.precision_summary = summary
        self._pool.put(master)

        can_share = share_weights and (
            arch_name is None or arch_name in SHARED_WEIGHT_SAFE_ARCHS)

        for _ in range(1, self.count):
            instance = model_factory()
            if can_share:
                self._share_weights(master, instance)
            self._pool.put(instance)

    @staticmethod
    def _share_weights(master, clone):
        """Point clone's weight+buffer tensors to master's (zero-copy).
        
        Safe because inference uses eval() + no_grad() — read-only.
        Saves ~100% of weight VRAM per extra instance.
        """
        # Find the nn.Module — wrapper classes use .model or .net
        master_mod = getattr(master, 'model', getattr(master, 'net', master))
        clone_mod = getattr(clone, 'model', getattr(clone, 'net', clone))
        # Skip if not an nn.Module (e.g. FFDNetDenoiser wrapper)
        if not hasattr(master_mod, 'named_parameters') or not hasattr(clone_mod, 'named_parameters'):
            return
        # Share parameters (weights)
        master_params = dict(master_mod.named_parameters())
        for name, param in clone_mod.named_parameters():
            if name in master_params:
                param.data = master_params[name].data
        # Share buffers (RPB tables, etc.)
        if hasattr(master_mod, 'named_buffers') and hasattr(clone_mod, 'named_buffers'):
            master_bufs = dict(master_mod.named_buffers())
            for name, buf in clone_mod.named_buffers():
                if name in master_bufs:
                    buf.data = master_bufs[name].data

    @contextmanager
    def checkout(self, stop_events=(), timeout=0.5):
        instance = None
        while instance is None:
            for evt in stop_events:
                if evt.is_set():
                    raise _PipelineAborted(self.stage_name)
            try:
                instance = self._pool.get(timeout=timeout)
            except queue.Empty:
                continue

        try:
            yield instance
        finally:
            self._pool.put(instance)

    def drain(self):
        instances = []
        while not self._pool.empty():
            try:
                instances.append(self._pool.get_nowait())
            except queue.Empty:
                break
        return instances

