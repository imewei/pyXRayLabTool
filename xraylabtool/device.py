"""GPU detection and diagnostics (System CUDA)."""

from __future__ import annotations

import logging
import shutil
import subprocess

_logger = logging.getLogger(__name__)


def get_system_cuda_version() -> tuple[str | None, int | None]:
    """Detect system CUDA version from nvcc.

    Returns (full_version, major_version) or (None, None).
    Example: ("13.1", 13)
    """
    nvcc_path = shutil.which("nvcc")
    if nvcc_path is None:
        return None, None
    try:
        result = subprocess.run(
            [nvcc_path, "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if "release" in line.lower():
                    parts = line.split("release")[-1].strip()
                    version = parts.split(",")[0].strip()
                    major = int(version.split(".")[0])
                    return version, major
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError, IndexError):
        pass
    return None, None


def get_gpu_info() -> tuple[str | None, float | None]:
    """Detect GPU name and SM version.

    Returns (gpu_name, sm_version) or (None, None).
    Example: ("NVIDIA GeForce RTX 4090", 8.9)
    """
    nvidia_smi_path = shutil.which("nvidia-smi")
    if nvidia_smi_path is None:
        return None, None
    try:
        result = subprocess.run(
            [nvidia_smi_path, "--query-gpu=name,compute_cap", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split("\n")[0].split(", ")
            if len(parts) >= 2:
                return parts[0], float(parts[1])
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError, IndexError):
        pass
    return None, None


def check_plugin_conflicts() -> list[str]:
    """Check for known JAX CUDA plugin conflicts.

    Returns list of issue descriptions (empty = no issues).
    """
    issues: list[str] = []
    try:
        import importlib.metadata as md

        jaxlib_v = md.version("jaxlib")

        cuda12 = cuda13 = None
        try:
            cuda12 = md.version("jax-cuda12-plugin")
        except md.PackageNotFoundError:
            pass
        try:
            cuda13 = md.version("jax-cuda13-plugin")
        except md.PackageNotFoundError:
            pass

        if cuda12 and cuda13:
            issues.append(
                f"Both cuda12 ({cuda12}) and cuda13 ({cuda13}) plugins installed. "
                "Only ONE can be active — this causes PJRT registration conflicts."
            )

        for name, version in [("cuda12", cuda12), ("cuda13", cuda13)]:
            if version and version != jaxlib_v:
                issues.append(
                    f"jax-{name}-plugin {version} != jaxlib {jaxlib_v}. "
                    "Plugin version must exactly match jaxlib."
                )

    except Exception as e:
        _logger.debug("Plugin conflict check failed: %s", e)

    return issues


def check_gpu_availability(warn: bool = True) -> bool:
    """Check if GPU is available and being used by JAX.

    If GPU hardware is detected but JAX is in CPU mode, prints
    a diagnostic warning with installation instructions.

    Returns True if GPU is being used by JAX, False otherwise.
    """
    try:
        gpu_name, sm_version = get_gpu_info()
        if gpu_name is None:
            return False

        import jax

        devices = jax.devices()
        using_gpu = any("cuda" in str(d).lower() for d in devices)

        if using_gpu:
            for issue in check_plugin_conflicts():
                _logger.warning("Plugin issue: %s", issue)
            return True

        if warn:
            cuda_version, _cuda_major = get_system_cuda_version()
            plugin_issues = check_plugin_conflicts()

            print("\nGPU AVAILABLE BUT NOT USED")
            print(f"  GPU: {gpu_name} (SM {sm_version})")
            print(f"  System CUDA: {cuda_version or 'Not found'}")
            print(f"  JAX backend: {jax.default_backend()}")

            if plugin_issues:
                print("\n  Issues detected:")
                for issue in plugin_issues:
                    print(f"    - {issue}")

            print("\n  Fix: make install-jax-gpu")
            pkg = get_recommended_package()
            if pkg:
                print(
                    "  Or:  pip uninstall -y "
                    "jax-cuda13-plugin jax-cuda13-pjrt "
                    "jax-cuda12-plugin jax-cuda12-pjrt"
                )
                print("       pip uninstall -y jax jaxlib")
                print(f'       pip install "{pkg}"')
            print()

        return False

    except ImportError:
        return False


def get_recommended_package() -> str | None:
    """Get recommended JAX package based on system CUDA and GPU.

    Returns "jax[cuda12-local]", "jax[cuda13-local]", or None.
    """
    _, cuda_major = get_system_cuda_version()
    _, sm_version = get_gpu_info()

    if cuda_major is None or sm_version is None:
        return None

    if cuda_major == 13 and sm_version >= 7.5:
        return "jax[cuda13-local]"
    elif cuda_major == 12 and sm_version >= 5.2:
        return "jax[cuda12-local]"
    return None


def get_device_info() -> dict:
    """Get comprehensive device information as a dictionary."""
    info: dict = {
        "jax_version": None,
        "jax_backend": None,
        "devices": [],
        "gpu_count": 0,
        "using_gpu": False,
        "gpu_hardware": None,
        "gpu_sm_version": None,
        "system_cuda_version": None,
        "system_cuda_major": None,
        "recommended_package": None,
        "plugin_issues": [],
    }

    try:
        import jax

        info["jax_version"] = jax.__version__
        info["jax_backend"] = jax.default_backend()
        devices = jax.devices()
        info["devices"] = [str(d) for d in devices]
        info["gpu_count"] = sum(1 for d in devices if "cuda" in str(d).lower())
        info["using_gpu"] = info["gpu_count"] > 0
    except ImportError:
        pass

    gpu_name, sm_version = get_gpu_info()
    info["gpu_hardware"] = gpu_name
    info["gpu_sm_version"] = sm_version

    cuda_version, cuda_major = get_system_cuda_version()
    info["system_cuda_version"] = cuda_version
    info["system_cuda_major"] = cuda_major
    info["recommended_package"] = get_recommended_package()
    info["plugin_issues"] = check_plugin_conflicts()

    return info
