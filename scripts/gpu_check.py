#!/usr/bin/env python3
"""Report the JAX backend and GPU/CUDA state (used by `make gpu-check` / `gpu-diagnose`).

Usage:
    python scripts/gpu_check.py              # backend, devices, SVD sanity check
    python scripts/gpu_check.py --diagnose   # also list CUDA packages and plugin conflicts

Exits non-zero if JAX is missing, the SVD check fails, or CUDA plugins conflict.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import sys

from xraylabtool.device import get_device_info


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--diagnose", action="store_true", help="list CUDA packages")
    args = parser.parse_args()

    info = get_device_info()
    if info["jax_version"] is None:
        print("error: jax is not installed", file=sys.stderr)
        return 1

    print(
        f"JAX {info['jax_version']}  backend={info['jax_backend']}  "
        f"devices={info['gpu_count']} GPU"
    )
    if not info["using_gpu"]:
        print("Note: running on CPU (no GPU visible to JAX)")

    import jax.numpy as jnp  # noqa: PLC0415  (only after the jax-installed check)

    print(f"SVD check: {jnp.linalg.svd(jnp.eye(3))[1]}")

    if args.diagnose:
        print(f"System CUDA: {info['system_cuda_version'] or 'nvcc not found'}")
        print(f"GPU: {info['gpu_hardware']} (SM {info['gpu_sm_version']})")
        print(f"Recommended: {info['recommended_package'] or 'none'}")
        for dist in sorted(
            importlib.metadata.distributions(), key=lambda d: d.metadata["Name"]
        ):
            name = dist.metadata["Name"].lower()
            if name.startswith(("jax", "nvidia-cuda")):
                print(f"  {name} {dist.version}")

    for issue in info["plugin_issues"]:
        print(f"CONFLICT: {issue}", file=sys.stderr)
    if info["plugin_issues"]:
        return 1
    print("All checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
