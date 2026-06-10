"""
SageAttention integration for Manga Colorizer.

Provides runtime monkey-patching of torch.nn.functional.scaled_dot_product_attention
with SageAttention's faster CUDA kernel. Benefits transformer-based upscaler models
(DAT, DAT-2, HAT, HAT-L) running on RTX 30/40/50 series GPUs.

SageAttention 2.2.0 post4 supports torch.compile (no graph breaks).
Older versions required torch.compiler.disable.

SageAttention must be installed separately. This module handles:
- Compatibility detection (GPU, CUDA, PyTorch)
- Runtime monkey-patch apply/remove (with safe attn_mask fallback)
- Installation helper (downloads pre-built wheel from woct0rdho's releases)
"""

import sys
import subprocess
import platform

# Store original SDPA reference for clean removal
_original_sdpa = None
_patch_applied = False


def detect_compatibility():
    """Check if the system supports SageAttention.
    
    Returns:
        dict with keys:
            - compatible (bool): True if likely compatible
            - cuda_version (str or None): e.g. '12.8'
            - torch_version (str or None): e.g. '2.7.0'
            - python_version (str): e.g. '312'
            - gpu_name (str or None): GPU name
            - compute_capability (tuple or None): e.g. (8, 9)
            - message (str): Human-readable status
    """
    result = {
        'compatible': False,
        'cuda_version': None,
        'torch_version': None,
        'python_version': f"{sys.version_info.major}{sys.version_info.minor}",
        'gpu_name': None,
        'compute_capability': None,
        'message': 'Unknown'
    }
    
    try:
        import torch
        result['torch_version'] = torch.__version__.split('+')[0]
        
        if not torch.cuda.is_available():
            result['message'] = 'No CUDA GPU detected'
            return result
            
        result['gpu_name'] = torch.cuda.get_device_name(0)
        
        # Extract CUDA version
        cuda_ver = torch.version.cuda
        if cuda_ver:
            result['cuda_version'] = cuda_ver
        
        # Check for RTX 30/40/50 series (compute capability >= 8.0)
        cap = torch.cuda.get_device_capability(0)
        result['compute_capability'] = cap
        if cap[0] >= 8:
            result['compatible'] = True
            result['message'] = f"Compatible: {result['gpu_name']} (SM {cap[0]}.{cap[1]}, CUDA {cuda_ver})"
        else:
            result['message'] = f"GPU {result['gpu_name']} (SM {cap[0]}.{cap[1]}) needs SM >= 8.0 for SageAttention"
            
    except ImportError:
        result['message'] = 'PyTorch not installed'
    except Exception as e:
        result['message'] = f'Detection error: {e}'
    
    return result


def get_installed_version():
    """Get the currently installed SageAttention version.
    
    Returns:
        str: Version string (e.g. '2.2.0') or None if not installed.
    """
    try:
        import importlib.metadata
        return importlib.metadata.version('sageattention')
    except Exception:
        return None


_gpu_sm = None  # cached (major, minor) compute capability


def _detect_sm():
    """Detect and cache GPU compute capability once."""
    global _gpu_sm
    if _gpu_sm is None:
        try:
            import torch
            _gpu_sm = torch.cuda.get_device_capability(0)
        except Exception:
            _gpu_sm = (0, 0)
    return _gpu_sm


def _sage_sdpa_wrapper(query, key, value, attn_mask=None, dropout_p=0.0,
                       is_causal=False, scale=None, **kwargs):
    """Smart SageAttention router — Sage2++ for unmasked attention only.
    
    - attn_mask present or dropout → original PyTorch SDPA (FlashAttn2/mem-efficient)
    - sm89+ (RTX 40/50) no mask → sageattn_qk_int8_pv_fp8_cuda (Sage2++ 8+8)
    - sm86 (RTX 30) no mask → sageattn (auto-pick)
    - Any failure → original PyTorch SDPA
    """
    global _original_sdpa

    # Masked attention or dropout → native SDPA (already uses FlashAttn2/mem-efficient)
    if attn_mask is not None or dropout_p > 0.0:
        return _original_sdpa(query, key, value, attn_mask=attn_mask,
                              dropout_p=dropout_p, is_causal=is_causal, scale=scale)

    sm = _detect_sm()

    try:
        if sm[0] >= 8 and sm[1] >= 9:  # sm89+ (RTX 40/50, Ada/Blackwell)
            from sageattention import sageattn_qk_int8_pv_fp8_cuda
            return sageattn_qk_int8_pv_fp8_cuda(
                query, key, value, is_causal=is_causal, sm_scale=scale,
                tensor_layout="HND", pv_accum_dtype="fp32+fp16")
        else:  # sm80/sm86 (RTX 30, A100)
            from sageattention import sageattn
            return sageattn(query, key, value, is_causal=is_causal,
                            sm_scale=scale, tensor_layout="HND")
    except Exception:
        return _original_sdpa(query, key, value, attn_mask=attn_mask,
                              is_causal=is_causal, scale=scale)


def apply_monkey_patch():
    """Apply SageAttention monkey-patch to F.scaled_dot_product_attention.
    
    Uses a safe wrapper that falls back to original SDPA when attn_mask
    is provided (needed for FDAT's position bias, etc.).
    
    Returns:
        tuple: (success: bool, message: str)
    """
    global _original_sdpa, _patch_applied
    
    if _patch_applied:
        return True, "SageAttention already active"
    
    try:
        # Verify sageattention is importable
        from sageattention import sageattn  # noqa: F401
        import torch.nn.functional as F
        
        # Save original for clean removal and fallback
        if _original_sdpa is None:
            _original_sdpa = F.scaled_dot_product_attention
        
        # Use safe wrapper instead of raw sageattn
        F.scaled_dot_product_attention = _sage_sdpa_wrapper
        _patch_applied = True
        
        version = get_installed_version() or 'unknown'
        return True, f"SageAttention {version} active (safe wrapper with attn_mask fallback)"
        
    except ImportError as e:
        # Distinguish "package not found" vs "CUDA kernel load failure"
        installed_ver = get_installed_version()
        if installed_ver:
            return False, (
                f"SageAttention {installed_ver} installed but CUDA kernel failed to load: {e}. "
                f"This usually means a CUDA version mismatch between the wheel and your runtime."
            )
        else:
            return False, (
                "SageAttention package not installed. "
                "Use the 'Install SageAttention' button in Advanced Settings, "
                "or run: pip install sageattention"
            )
    except Exception as e:
        return False, f"Failed to apply patch: {e}"


def remove_monkey_patch():
    """Remove the SageAttention monkey-patch, restoring original SDPA."""
    global _original_sdpa, _patch_applied
    
    if not _patch_applied or _original_sdpa is None:
        return
    
    try:
        import torch.nn.functional as F
        F.scaled_dot_product_attention = _original_sdpa
        _patch_applied = False
    except Exception:
        pass


def install_sageattention(log_callback=None):
    """Install SageAttention, trying post4 pre-built wheel first then fallbacks.
    
    Wheel source: https://github.com/woct0rdho/SageAttention/releases
    
    Priority order:
    1. v2.2.0-windows.post4 (latest, supports torch.compile, no graph breaks)
    2. v2.2.0-windows.post3 (stable, requires torch.compiler.disable)
    3. v2.1.1-windows (older stable)
    4. PyPI sageattention==1.0.6 (Triton-based, ~2.1x speedup)
    
    Args:
        log_callback: Optional callable(message: str) for progress updates
        
    Returns:
        tuple: (success: bool, message: str)
    """
    def log(msg):
        if log_callback:
            log_callback(msg)
        print(f"[SageAttention] {msg}")
    
    compat = detect_compatibility()
    if not compat['cuda_version']:
        return False, "No CUDA detected. SageAttention requires an NVIDIA GPU."
    
    log(f"System: {compat['gpu_name']}, CUDA {compat['cuda_version']}, "
        f"PyTorch {compat['torch_version']}, Python {compat['python_version']}")
    
    # Only try pre-built wheels on Windows
    if platform.system() == 'Windows':
        success, msg = _try_install_from_wheel(compat, log)
        if success:
            return True, msg
        log(f"Pre-built wheel not available: {msg}")
    
    # Fallback: pip install from PyPI (SA 1.x, Triton-based)
    log("Trying PyPI fallback (SageAttention 1.0.6)...")
    try:
        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'install', 'sageattention==1.0.6'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        version = get_installed_version()
        return True, f"Installed SageAttention {version} from PyPI (~2.1x speedup)"
    except subprocess.CalledProcessError as e:
        return False, f"pip install failed: {e}"
    except Exception as e:
        return False, f"Installation error: {e}"


def _try_install_from_wheel(compat, log):
    """Try to install SA 2.x from pre-built wheel (Windows only).
    
    Wheel source: https://github.com/woct0rdho/SageAttention/releases
    
    post4 naming convention:
      - Uses "torch2.9.0andhigher" to mean PyTorch >= 2.9
      - Uses Python ABI3 (cp39-abi3) for all Python versions >= 3.9
      - Uses libtorch stable ABI
    
    Returns:
        tuple: (success: bool, message: str)
    """
    cuda_ver = compat['cuda_version']
    torch_ver = compat['torch_version']
    
    if not cuda_ver or not torch_ver:
        return False, "Missing CUDA or PyTorch version"
    
    # Normalize CUDA version to wheel format (e.g. '12.8' -> '128')
    cuda_parts = cuda_ver.split('.')
    cuda_whl = ''.join(cuda_parts[:2])
    
    # Extract torch major.minor for matching
    torch_parts = torch_ver.split('.')
    torch_major = int(torch_parts[0]) if torch_parts else 0
    torch_minor = int(torch_parts[1]) if len(torch_parts) > 1 else 0
    torch_mm = f"{torch_major}.{torch_minor}"
    
    # Build wheel configs ordered by preference (newest first)
    # Format: (wheel_url_suffix, description)
    wheel_attempts = []
    
    base_url = "https://github.com/woct0rdho/SageAttention/releases/download"
    
    # --- v2.2.0 post4 (latest stable, supports torch.compile) ---
    # post4 wheels use "torch2.9.0andhigher" → works for PyTorch >= 2.9
    if torch_major > 2 or (torch_major == 2 and torch_minor >= 9):
        if cuda_whl == '128':
            wheel_attempts.append((
                f"{base_url}/v2.2.0-windows.post4/"
                f"sageattention-2.2.0+cu128torch2.9.0andhigher.post4-cp39-abi3-win_amd64.whl",
                "SA 2.2.0 post4 (cu128, torch>=2.9, torch.compile support)"
            ))
        if cuda_whl == '130':
            wheel_attempts.append((
                f"{base_url}/v2.2.0-windows.post4/"
                f"sageattention-2.2.0+cu130torch2.9.0andhigher.post4-cp39-abi3-win_amd64.whl",
                "SA 2.2.0 post4 (cu130, torch>=2.9, torch.compile support)"
            ))
    
    # --- v2.2.0 post3 (stable fallback for older PyTorch) ---
    # post3 has ABI3 wheels for specific torch versions
    post3_configs = [
        ("128", "2.7", "2.7.1.post3"),
        ("126", "2.7", "2.7.1.post3"),
        ("128", "2.6", "2.6.0.post3"),
        ("126", "2.6", "2.6.0.post3"),
        ("124", "2.6", "2.6.0.post3"),
        ("124", "2.5", "2.5.1.post3"),
        ("118", "2.5", "2.5.1.post3"),
    ]
    for cfg_cuda, cfg_torch, torch_fn_ver in post3_configs:
        if cfg_cuda == cuda_whl and cfg_torch == torch_mm:
            wheel_attempts.append((
                f"{base_url}/v2.2.0-windows.post3/"
                f"sageattention-2.2.0+cu{cfg_cuda}torch{torch_fn_ver}-cp39-abi3-win_amd64.whl",
                f"SA 2.2.0 post3 (cu{cfg_cuda}, torch {cfg_torch})"
            ))
    
    # --- v2.1.1 (older stable, exact Python version wheels) ---
    py_ver = compat['python_version']
    v211_configs = [
        ("128", "2.7"),
        ("126", "2.6"),
        ("124", "2.6"),
        ("124", "2.5"),
        ("118", "2.5"),
    ]
    for cfg_cuda, cfg_torch in v211_configs:
        if cfg_cuda == cuda_whl and cfg_torch == torch_mm:
            wheel_attempts.append((
                f"{base_url}/v2.1.1-windows/"
                f"sageattention-2.1.1+cu{cfg_cuda}torch{cfg_torch}.0-cp{py_ver}-cp{py_ver}-win_amd64.whl",
                f"SA 2.1.1 (cu{cfg_cuda}, torch {cfg_torch}, py{py_ver})"
            ))
    
    if not wheel_attempts:
        return False, f"No compatible wheel for CUDA {cuda_ver}, PyTorch {torch_ver}"
    
    # Try each wheel in order
    for wheel_url, desc in wheel_attempts:
        log(f"Trying: {desc}")
        try:
            subprocess.check_call(
                [sys.executable, '-m', 'pip', 'install', wheel_url],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            version = get_installed_version()
            speedup = "~3x" if "post4" in wheel_url or "post3" in wheel_url else "~2.5x"
            compile_note = " (torch.compile compatible)" if "post4" in wheel_url else ""
            return True, f"Installed SageAttention {version} ({speedup} speedup){compile_note}"
        except subprocess.CalledProcessError:
            continue
        except Exception:
            continue
    
    return False, f"All wheels failed for CUDA {cuda_ver}, PyTorch {torch_ver}"
