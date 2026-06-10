"""
msvc_env — Bootstrap the MSVC C++ toolchain into the current Python process.

torch.compile (TorchInductor) and CUDA-compiled SageAttention kernels need a C++
compiler (cl.exe) plus the full INCLUDE/LIB environment that vcvars64.bat sets up.

When the app is NOT launched from an "x64 Native Tools" prompt, those vars are
missing and Inductor fails with:
    "Failed to find C compiler. Please specify via CC environment variable."
(often flashing dozens of empty cmd windows as it probes for compilers).

This module locates vcvars64.bat (via vswhere or well-known paths), runs it once,
and imports the resulting INCLUDE / LIB / LIBPATH / PATH into os.environ so the
compiler is usable in-process. Idempotent and safe to call multiple times.
"""

import os
import subprocess
import glob

_BOOTSTRAPPED = False
_RESULT = None  # (ok: bool, message: str)

# Env vars worth importing from the vcvars shell.
_IMPORT_KEYS = {
    'PATH', 'INCLUDE', 'LIB', 'LIBPATH',
    'VCINSTALLDIR', 'VCToolsInstallDir', 'VCToolsVersion',
    'WindowsSdkDir', 'WindowsSdkVerBinPath', 'WindowsSDKVersion',
    'WindowsSdkBinPath', 'UCRTVersion', 'UniversalCRTSdkDir',
    'VSINSTALLDIR', 'VSCMD_ARG_HOST_ARCH', 'VSCMD_ARG_TGT_ARCH',
}


def _vswhere_paths():
    pf86 = os.environ.get('ProgramFiles(x86)', r'C:\Program Files (x86)')
    vswhere = os.path.join(pf86, 'Microsoft Visual Studio', 'Installer', 'vswhere.exe')
    if not os.path.isfile(vswhere):
        return []
    try:
        out = subprocess.run(
            [vswhere, '-products', '*', '-latest',
             '-requires', 'Microsoft.VisualCpp.Tools.Host.x64',
             '-property', 'installationPath'],
            capture_output=True, text=True, timeout=30,
        )
        roots = [line.strip() for line in out.stdout.splitlines() if line.strip()]
        return roots
    except Exception:
        return []


def _candidate_vcvars():
    """Yield possible vcvars64.bat locations, best first."""
    seen = set()
    # 1) vswhere-discovered installs
    for root in _vswhere_paths():
        bat = os.path.join(root, 'VC', 'Auxiliary', 'Build', 'vcvars64.bat')
        if bat not in seen:
            seen.add(bat)
            if os.path.isfile(bat):
                yield bat
    # 2) well-known glob fallbacks (handles "2022", preview "18", BuildTools, full IDE)
    pf86 = os.environ.get('ProgramFiles(x86)', r'C:\Program Files (x86)')
    pf = os.environ.get('ProgramFiles', r'C:\Program Files')
    patterns = []
    for base in (pf86, pf):
        patterns += [
            os.path.join(base, 'Microsoft Visual Studio', '*', '*', 'VC', 'Auxiliary', 'Build', 'vcvars64.bat'),
        ]
    for pat in patterns:
        for bat in sorted(glob.glob(pat), reverse=True):  # newest-ish first
            if bat not in seen:
                seen.add(bat)
                if os.path.isfile(bat):
                    yield bat


def _read_vcvars_env(vcvars_bat):
    """Run vcvars64.bat in a cmd shell and return its env as a dict."""
    # `set` dumps the full env after vcvars configures it.
    cmd = f'cmd /s /c ""{vcvars_bat}" >nul 2>&1 && set"'
    proc = subprocess.run(
        cmd, capture_output=True, text=True, timeout=120,
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
    )
    if proc.returncode != 0:
        raise RuntimeError(f"vcvars exited {proc.returncode}: {proc.stderr.strip()[:200]}")

    env = {}
    for line in proc.stdout.splitlines():
        if '=' not in line:
            continue
        key, _, val = line.partition('=')
        env[key] = val
    return env


def _apply_env(env):
    imported = 0
    for key, val in env.items():
        ku = key.strip().upper()
        # Always take PATH (compiler + linker live here) and the known build keys.
        if ku == 'PATH' or key in _IMPORT_KEYS or ku in {k.upper() for k in _IMPORT_KEYS}:
            os.environ[key] = val
            if ku == 'PATH':
                # Defensive: keep both casings so lookups always succeed.
                os.environ['PATH'] = val
                os.environ['Path'] = val
            imported += 1
    return imported


def _cl_on_path(path_val=None):
    if path_val is None:
        path_val = os.environ.get('PATH') or os.environ.get('Path') or ''
    for d in path_val.split(os.pathsep):
        if d and os.path.isfile(os.path.join(d, 'cl.exe')):
            return os.path.join(d, 'cl.exe')
    return None


def ensure_msvc(verbose=True, log=None):
    """
    Make cl.exe + the MSVC build env available in this process.

    Returns (ok, message). Idempotent: subsequent calls return the cached result.
    """
    global _BOOTSTRAPPED, _RESULT

    def _say(msg, level='I'):
        if callable(log):
            try:
                log(msg, level=level)
                return
            except TypeError:
                log(msg)
                return
        if verbose:
            print(msg)

    if _BOOTSTRAPPED:
        return _RESULT

    if os.name != 'nt':
        _BOOTSTRAPPED = True
        _RESULT = (True, 'non-windows: no MSVC needed')
        return _RESULT

    # Already usable? (e.g. launched from a dev prompt)
    existing = _cl_on_path()
    if existing:
        _BOOTSTRAPPED = True
        _RESULT = (True, f'cl.exe already on PATH: {existing}')
        _say(f"[*] MSVC: compiler already available ({existing})")
        return _RESULT

    last_err = None
    for bat in _candidate_vcvars():
        try:
            env = _read_vcvars_env(bat)
            path_val = env.get('PATH') or env.get('Path') or ''
            cl = _cl_on_path(path_val)
            if cl:
                n = _apply_env(env)
                # Hint compilers for tools that read CC/CXX (e.g. some inductor paths).
                os.environ.setdefault('CC', cl)
                os.environ.setdefault('CXX', cl)
                _BOOTSTRAPPED = True
                _RESULT = (True, f'MSVC loaded from {bat} ({n} vars), cl={cl}')
                _say(f"[*] MSVC: toolchain loaded from {os.path.dirname(bat)} (cl.exe ready)")
                return _RESULT
            last_err = f'vcvars ran but cl.exe still not found ({bat})'
        except Exception as e:
            last_err = f'{bat}: {e}'
            continue

    _BOOTSTRAPPED = True
    _RESULT = (False, last_err or 'no vcvars64.bat found')
    _say(f"[!] MSVC: C++ compiler not found — torch.compile/CUDA-Sage will be disabled. ({_RESULT[1]})", level='W')
    return _RESULT
