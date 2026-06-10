import torch


POLICY_AUTO = 'auto'
POLICY_FP16 = 'fp16'
POLICY_BF16 = 'bf16'
POLICY_FP32 = 'fp32'
POLICY_FP32_TF32 = 'fp32_tf32'
POLICY_FP8 = 'fp8'

FALLBACK_PER_IMAGE = 'per_image'
FALLBACK_ABORT = 'abort'
FALLBACK_DISABLE_AMP = 'disable_amp'


def normalize_precision_policy(value):
    if not value:
        return POLICY_AUTO
    text = str(value).strip().lower()
    if text in {'auto'}:
        return POLICY_AUTO
    if text in {'fp16', 'float16', 'half'}:
        return POLICY_FP16
    if text in {'bf16', 'bfloat16'}:
        return POLICY_BF16
    if text in {'fp32', 'float32'}:
        return POLICY_FP32
    if text in {'fp32_tf32', 'fp32+tf32', 'tf32'}:
        return POLICY_FP32_TF32
    if text in {'fp8', 'float8', 'float8_e4m3', 'float8_e4m3fn', 'float8_e5m2'}:
        return POLICY_FP8
    return POLICY_AUTO


def normalize_fallback_mode(value):
    if not value:
        return FALLBACK_PER_IMAGE
    text = str(value).strip().lower()
    if text in {FALLBACK_PER_IMAGE, 'per-image', 'perimage', 'retry'}:
        return FALLBACK_PER_IMAGE
    if text in {FALLBACK_ABORT, 'error', 'raise'}:
        return FALLBACK_ABORT
    if text in {FALLBACK_DISABLE_AMP, 'disable', 'fp32'}:
        return FALLBACK_DISABLE_AMP
    return FALLBACK_PER_IMAGE


def _cuda_bf16_supported():
    if not torch.cuda.is_available():
        return False
    if hasattr(torch.cuda, 'is_bf16_supported'):
        try:
            return bool(torch.cuda.is_bf16_supported())
        except Exception:
            return False
    try:
        props = torch.cuda.get_device_properties(0)
        return int(props.major) >= 8
    except Exception:
        return False


def _fp8_autocast_dtype():
    for name in ('float8_e4m3fn', 'float8_e4m3fnuz', 'float8_e5m2', 'float8_e5m2fnuz'):
        if hasattr(torch, name):
            return getattr(torch, name)
    return None


def _cuda_fp8_supported():
    # Temporarily disabled: PyTorch BatchNorm2d does not support Float8_e4m3fn weights
    return False


def _cuda_device_capability(device_index=0):
    if not torch.cuda.is_available():
        return None
    try:
        props = torch.cuda.get_device_properties(device_index)
        return int(props.major), int(props.minor)
    except Exception:
        return None


def supports_fp8():
    return bool(_cuda_fp8_supported())


def supports_bf16():
    return bool(_cuda_bf16_supported())


def resolve_precision(config, device_name=None, stage=None, override_policy=None):
    override = override_policy
    if override is None and config is not None:
        override = getattr(config, 'precision_policy_override', None)
        
    if stage in {'colorize', 'denoise'} and config is not None:
        default_policy = getattr(config, 'colorize_precision_policy', getattr(config, 'precision_policy', POLICY_AUTO))
    elif stage == 'upscale' and config is not None:
        default_policy = getattr(config, 'upscale_precision_policy', getattr(config, 'precision_policy', POLICY_AUTO))
    else:
        default_policy = getattr(config, 'precision_policy', POLICY_AUTO) if config is not None else POLICY_AUTO

    policy = normalize_precision_policy(override if override else default_policy)
    cast_weights = bool(getattr(config, 'precision_cast_weights', True))
    allow_tf32 = bool(getattr(config, 'precision_allow_tf32', True))
    fallback_mode = normalize_fallback_mode(getattr(config, 'precision_fallback', FALLBACK_PER_IMAGE))
    device = device_name or getattr(config, 'device', 'cpu')

    is_cuda = str(device).lower() == 'cuda' and torch.cuda.is_available()
    bf16_supported = _cuda_bf16_supported() if is_cuda else False
    fp8_supported = _cuda_fp8_supported() if is_cuda else False

    resolved = POLICY_FP32
    reason = ''
    if is_cuda:
        if policy == POLICY_AUTO:
            resolved = POLICY_BF16 if bf16_supported else POLICY_FP16
        elif policy == POLICY_FP8:
            if fp8_supported:
                resolved = POLICY_FP8
            else:
                resolved = POLICY_FP16
                reason = 'fp8_unsupported'
        elif policy == POLICY_BF16:
            if bf16_supported:
                resolved = POLICY_BF16
            else:
                resolved = POLICY_FP16
                reason = 'bf16_unsupported'
        elif policy == POLICY_FP16:
            resolved = POLICY_FP16
        elif policy == POLICY_FP32_TF32:
            resolved = POLICY_FP32_TF32
        else:
            resolved = POLICY_FP32


    autocast_enabled = False
    autocast_dtype = None
    if resolved == POLICY_FP16:
        autocast_dtype = torch.float16
    elif resolved == POLICY_BF16:
        autocast_dtype = torch.bfloat16
    elif resolved == POLICY_FP8:
        # Mixed FP8: PyTorch Conv2d natively rejects FP8 autocast.
        # Fallback compute to BF16 (or FP16), but load weights in FP8 later.
        autocast_dtype = torch.bfloat16 if _cuda_bf16_supported() else torch.float16

    autocast_enabled = bool(is_cuda and autocast_dtype is not None)

    model_dtype = autocast_dtype if (cast_weights and autocast_dtype is not None) else torch.float32
    if cast_weights and resolved == POLICY_FP8:
        model_dtype = _fp8_autocast_dtype() or model_dtype

    allow_tf32_effective = is_cuda and allow_tf32

    return {
        'device': device,
        'policy': policy,
        'resolved_policy': resolved,
        'reason': reason,
        'cast_weights': cast_weights,
        'fallback_mode': fallback_mode,
        'autocast_enabled': bool(autocast_enabled),
        'autocast_dtype': autocast_dtype,
        'model_dtype': model_dtype,
        'allow_tf32': bool(allow_tf32_effective),
        'bf16_supported': bool(bf16_supported),
        'fp8_supported': bool(fp8_supported),
    }


def apply_tf32(allow_tf32):
    enabled = bool(allow_tf32)
    try:
        torch.backends.cuda.matmul.allow_tf32 = enabled
    except Exception:
        pass
    try:
        torch.backends.cudnn.allow_tf32 = enabled
    except Exception:
        pass
    try:
        if enabled:
            torch.set_float32_matmul_precision('high')
        else:
            torch.set_float32_matmul_precision('highest')
    except Exception:
        pass


def dtype_from_model_name(path):
    """
    Parse the model's NATIVE precision from its filename.

    Model files encode their trained dtype, e.g.:
        ..._FDAT_M_..._fp16.safetensors  -> torch.float16
        ..._FDAT_XL_..._bf16.safetensors -> torch.bfloat16
        ..._DAT2_..._bf16.safetensors    -> torch.bfloat16
    Returns the torch dtype, or None if the name carries no precision tag
    (e.g. RealESRGAN_x4plus.pth -> fall back to policy).
    """
    if not path:
        return None
    n = str(path).lower()
    # bf16 first: 'fp16' substring can't appear in a bf16 name, but be explicit.
    if 'bf16' in n or 'bfloat16' in n:
        return torch.bfloat16
    if 'fp16' in n or 'float16' in n:
        return torch.float16
    if 'fp32' in n or 'float32' in n:
        return torch.float32
    return None


# --- Per-pass GLOBAL precision ---------------------------------------------
# The pipeline runs two passes that NEVER overlap (pass1 = denoise+colorize,
# pass2 = upscale). VRAM peak = max(pass1, pass2). So whichever pass is active
# should own the process-wide low-precision settings: match everything to the
# active pass's dtype to minimise casts AND VRAM, and skip TF32 (irrelevant once
# all math is fp16/bf16 — TF32 only touches stray fp32 ops).
_ACTIVE_PASS_DTYPE = None


def set_active_pass_precision(model_dtype, log=None):
    """
    Set process-wide precision for the pass that is now starting.

    - fp16/bf16 active  -> low-precision is global; TF32 is meaningless, so we
      leave matmul precision at the cheapest setting and DON'T burn cycles
      flipping TF32 flags (your "tf32 feels overkill" — correct here).
    - fp32 active        -> keep full precision (highest), no TF32 surprises.

    `model_dtype` is the dtype the active pass's model runs in (ideally derived
    from the model filename via dtype_from_model_name()).
    """
    if model_dtype is None:
        return

    global _ACTIVE_PASS_DTYPE
    _ACTIVE_PASS_DTYPE = model_dtype

    def _say(msg):
        if callable(log):
            try:
                log(msg, level='D')
            except TypeError:
                log(msg)

    try:
        if model_dtype in (torch.float16, torch.bfloat16):
            # All heavy math is already low precision -> TF32 has nothing to do.
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.set_float32_matmul_precision('highest')
            _say(f"[*] pass precision => global {('fp16' if model_dtype==torch.float16 else 'bf16')} (TF32 off, no needless casts)")
        else:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.set_float32_matmul_precision('highest')
            _say("[*] pass precision => global fp32 (full precision)")
    except Exception:
        pass


def active_pass_dtype():
    return _ACTIVE_PASS_DTYPE


def precision_summary(spec):
    if not isinstance(spec, dict):
        return ''
    resolved = spec.get('resolved_policy', spec.get('policy', 'fp32'))
    autocast = spec.get('autocast_dtype')
    autocast_name = 'none'
    if autocast == torch.float16:
        autocast_name = 'fp16'
    elif autocast == torch.bfloat16:
        autocast_name = 'bf16'
    elif autocast in {getattr(torch, 'float8_e4m3fn', object()), getattr(torch, 'float8_e4m3fnuz', object())}:
        autocast_name = 'fp8'
    elif autocast in {getattr(torch, 'float8_e5m2', object()), getattr(torch, 'float8_e5m2fnuz', object())}:
        autocast_name = 'fp8'
    weights = spec.get('model_dtype')
    weights_name = 'fp32'
    if weights == torch.float16:
        weights_name = 'fp16'
    elif weights == torch.bfloat16:
        weights_name = 'bf16'
    elif weights in {getattr(torch, 'float8_e4m3fn', object()), getattr(torch, 'float8_e4m3fnuz', object())}:
        weights_name = 'fp8'
    elif weights in {getattr(torch, 'float8_e5m2', object()), getattr(torch, 'float8_e5m2fnuz', object())}:
        weights_name = 'fp8'
    tf32 = 'on' if spec.get('allow_tf32') else 'off'
    reason = spec.get('reason')
    reason_text = f" ({reason})" if reason else ''
    return f"policy={spec.get('policy','auto')} resolved={resolved}{reason_text} autocast={autocast_name} weights={weights_name} tf32={tf32}"


def precision_vram_scale(spec):
    if not isinstance(spec, dict):
        return 1.0
    resolved = spec.get('resolved_policy', spec.get('policy', POLICY_FP32))
    cast_weights = bool(spec.get('cast_weights', False))
    if not cast_weights:
        return 1.0
    if resolved in {POLICY_FP16, POLICY_BF16}:
        return 0.65
    if resolved == POLICY_FP8:
        return 0.5
    return 1.0


def has_invalid_output(value):
    if isinstance(value, torch.Tensor):
        if not value.is_floating_point():
            return False
        return bool(torch.isnan(value).any().item() or torch.isinf(value).any().item())
    if isinstance(value, (list, tuple)):
        return any(has_invalid_output(item) for item in value)
    try:
        import numpy as np
        if isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.floating):
            return bool(np.isnan(value).any() or np.isinf(value).any())
    except Exception:
        return False
    return False


def get_cuda_device_summary(device_index=0):
    summary = {
        'available': False,
        'device_index': device_index,
        'name': 'CPU',
        'capability': None,
        'bf16_supported': False,
        'tf32_available': False,
        'fp8_supported': False,
    }

    if not torch.cuda.is_available():
        return summary

    try:
        props = torch.cuda.get_device_properties(device_index)
        summary['available'] = True
        summary['name'] = str(getattr(props, 'name', 'CUDA'))
        summary['capability'] = f"{int(props.major)}.{int(props.minor)}"
        summary['bf16_supported'] = bool(_cuda_bf16_supported())
        summary['tf32_available'] = int(props.major) >= 8
        summary['fp8_supported'] = bool(_cuda_fp8_supported())
    except Exception:
        return summary

    return summary


def format_cuda_device_summary(summary):
    if not summary:
        return 'CUDA info unavailable.'
    if not summary.get('available'):
        return 'CUDA not available. Using CPU (FP32).'

    name = summary.get('name', 'CUDA')
    capability = summary.get('capability') or 'unknown'
    bf16 = 'yes' if summary.get('bf16_supported') else 'no'
    tf32 = 'yes' if summary.get('tf32_available') else 'no'
    fp8 = 'yes' if summary.get('fp8_supported') else 'no'
    return f"Device: {name}\nCompute capability: {capability}\nBF16 supported: {bf16}\nFP8 supported: {fp8}\nTF32 available: {tf32}"
