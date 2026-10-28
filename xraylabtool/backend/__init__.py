"""Backend helpers: JAX runtime, shape buckets and the in-tree PCHIP interpolator.

The NumPy backend and the ``ops`` dispatch layer were removed (ADR-006); this
package keeps ``set_backend``/``get_backend`` as explicit stubs so existing
callers fail loudly instead of silently running a different code path. It
imports nothing heavy: ``runtime``, ``shapes`` and ``interpolation`` are loaded
by their users.
"""

from __future__ import annotations

__all__ = ["get_backend", "set_backend"]


def get_backend() -> str:
    """Return the name of the only backend, ``"jax"``."""
    return "jax"


def set_backend(name: str) -> None:
    """Accept ``"jax"`` (a no-op); reject every other name.

    Parameters
    ----------
    name : str
        Backend name. Only ``"jax"`` is valid.

    Raises
    ------
    ValueError
        For any name other than ``"jax"``, including ``"numpy"``.
    """
    if name != "jax":
        raise ValueError("the NumPy backend was removed; see ADR-006")
