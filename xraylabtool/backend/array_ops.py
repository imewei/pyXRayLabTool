from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    pass

Array = Any


@runtime_checkable
class ArrayBackend(Protocol):
    @property
    def float64(self) -> Any: ...

    def zeros(self, shape: Any, dtype: Any = np.float64) -> Array: ...
    def ones(self, shape: Any, dtype: Any = np.float64) -> Array: ...
    def asarray(self, x: Any, dtype: Any = None) -> Array: ...
    def square(self, x: Any) -> Array: ...
    def sqrt(self, x: Any) -> Array: ...
    def exp(self, x: Any) -> Array: ...
    def sum(self, x: Any, axis: Any = None) -> Array: ...
    def where(self, condition: Any, x: Any, y: Any) -> Array: ...
    def maximum(self, x: Any, y: Any) -> Array: ...
    def einsum(self, subscripts: str, *operands: Any) -> Array: ...
    def isnan(self, x: Any) -> Array: ...
    def isinf(self, x: Any) -> Array: ...
    def isfinite(self, x: Any) -> Array: ...
    def any(self, x: Any) -> bool: ...
    def all(self, x: Any) -> bool: ...
    def linspace(self, start: Any, stop: Any, num: int) -> Array: ...
    def logspace(self, start: Any, stop: Any, num: int) -> Array: ...
    def argsort(self, x: Any) -> Array: ...
    def ascontiguousarray(self, x: Any) -> Array: ...


class NumpyBackend:
    @property
    def float64(self) -> Any:
        return np.float64

    def zeros(self, shape: Any, dtype: Any = np.float64) -> Array:
        return np.zeros(shape, dtype=dtype)

    def ones(self, shape: Any, dtype: Any = np.float64) -> Array:
        return np.ones(shape, dtype=dtype)

    def asarray(self, x: Any, dtype: Any = None) -> Array:
        return np.asarray(x, dtype=dtype)

    def square(self, x: Any) -> Array:
        return np.square(x)

    def sqrt(self, x: Any) -> Array:
        return np.sqrt(x)

    def exp(self, x: Any) -> Array:
        return np.exp(x)

    def sum(self, x: Any, axis: Any = None) -> Array:
        return np.sum(x, axis=axis)

    def where(self, condition: Any, x: Any, y: Any) -> Array:
        return np.where(condition, x, y)

    def maximum(self, x: Any, y: Any) -> Array:
        return np.maximum(x, y)

    def einsum(self, subscripts: str, *operands: Any) -> Array:
        return np.einsum(subscripts, *operands)

    def isnan(self, x: Any) -> Array:
        return np.isnan(x)

    def isinf(self, x: Any) -> Array:
        return np.isinf(x)

    def isfinite(self, x: Any) -> Array:
        return np.isfinite(x)

    def any(self, x: Any) -> bool:
        return bool(np.any(x))

    def all(self, x: Any) -> bool:
        return bool(np.all(x))

    def linspace(self, start: Any, stop: Any, num: int) -> Array:
        return np.linspace(start, stop, num)

    def logspace(self, start: Any, stop: Any, num: int) -> Array:
        return np.logspace(start, stop, num)

    def argsort(self, x: Any) -> Array:
        return np.argsort(x)

    def ascontiguousarray(self, x: Any) -> Array:
        return np.ascontiguousarray(x)


class JaxBackend:
    def __init__(self) -> None:
        import jax  # type: ignore[import-not-found]

        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp  # type: ignore[import-not-found]

        self._jnp = jnp

    @property
    def float64(self) -> Any:
        return self._jnp.float64

    def zeros(self, shape: Any, dtype: Any = None) -> Array:
        dtype = dtype if dtype is not None else self._jnp.float64
        return self._jnp.zeros(shape, dtype=dtype)

    def ones(self, shape: Any, dtype: Any = None) -> Array:
        dtype = dtype if dtype is not None else self._jnp.float64
        return self._jnp.ones(shape, dtype=dtype)

    def asarray(self, x: Any, dtype: Any = None) -> Array:
        return self._jnp.asarray(x, dtype=dtype)

    def square(self, x: Any) -> Array:
        return self._jnp.square(x)

    def sqrt(self, x: Any) -> Array:
        return self._jnp.sqrt(x)

    def exp(self, x: Any) -> Array:
        return self._jnp.exp(x)

    def sum(self, x: Any, axis: Any = None) -> Array:
        return self._jnp.sum(x, axis=axis)

    def where(self, condition: Any, x: Any, y: Any) -> Array:
        return self._jnp.where(condition, x, y)

    def maximum(self, x: Any, y: Any) -> Array:
        return self._jnp.maximum(x, y)

    def einsum(self, subscripts: str, *operands: Any) -> Array:
        return self._jnp.einsum(subscripts, *operands)

    def isnan(self, x: Any) -> Array:
        return self._jnp.isnan(x)

    def isinf(self, x: Any) -> Array:
        return self._jnp.isinf(x)

    def isfinite(self, x: Any) -> Array:
        return self._jnp.isfinite(x)

    def any(self, x: Any) -> bool:
        return bool(self._jnp.any(x))

    def all(self, x: Any) -> bool:
        return bool(self._jnp.all(x))

    def linspace(self, start: Any, stop: Any, num: int) -> Array:
        return self._jnp.linspace(start, stop, num)

    def logspace(self, start: Any, stop: Any, num: int) -> Array:
        return self._jnp.logspace(start, stop, num)

    def argsort(self, x: Any) -> Array:
        return self._jnp.argsort(x)

    def ascontiguousarray(self, x: Any) -> Array:
        return self._jnp.asarray(x)


def _has_nvidia_gpu() -> bool:
    """Fast check for NVIDIA GPU without importing JAX (~1ms)."""
    import shutil

    return shutil.which("nvidia-smi") is not None


def _auto_select_backend() -> ArrayBackend:
    """Select the best available backend: JAX GPU > JAX CPU > NumPy.

    JAX is preferred when a GPU is available (10-50x speedup on large arrays).
    On CPU-only systems, NumPy is faster for typical workloads (<5000 points)
    due to lower per-call dispatch overhead vs JAX's XLA runtime.

    Avoids importing JAX on CPU-only systems to keep cold start fast (~300ms
    vs ~900ms with JAX import). Only pays the JAX import cost when a GPU is
    detected and JAX is installed.
    """
    import importlib.util

    if not importlib.util.find_spec("jax"):
        return NumpyBackend()

    if not _has_nvidia_gpu():
        return NumpyBackend()

    # GPU detected and JAX installed — pay the import cost
    try:
        import jax  # type: ignore[import-not-found]

        if jax.default_backend() == "gpu":
            return JaxBackend()
    except (ImportError, RuntimeError):
        pass

    return NumpyBackend()


_backend: ArrayBackend = _auto_select_backend()


def get_backend() -> ArrayBackend:
    return _backend


def set_backend(name: str) -> None:
    global _backend
    if name == "jax":
        _backend = JaxBackend()
    elif name == "numpy":
        _backend = NumpyBackend()
    else:
        raise ValueError(f"Unknown backend: {name!r}")
    ops._invalidate_cache()
    # Clear JIT cache so kernels recompile for the new backend
    try:
        from xraylabtool.calculators.core import _jit_cache

        _jit_cache.clear()
    except ImportError:
        pass


class _OpsProxy:
    """Proxy that delegates to the active backend with method caching."""

    def __getattr__(self, name: str) -> Any:
        attr = getattr(_backend, name)
        # Cache on the proxy instance for subsequent calls
        object.__setattr__(self, name, attr)
        return attr

    def is_jax(self) -> bool:
        """Check whether the active backend is JAX."""
        return isinstance(_backend, JaxBackend)

    def _invalidate_cache(self) -> None:
        """Clear cached methods when backend changes."""
        for key in list(self.__dict__):
            object.__delattr__(self, key)


ops = _OpsProxy()
