"""JAX runtime policy: x64, host/device transfers, compile cache, worker guard.

Everything that crosses the host/device boundary goes through :func:`upload`,
:func:`fetch_status` or :func:`fetch`; each opens a narrow
``jax.transfer_guard("allow")`` scope and bumps a counter, which is what the
transfer-budget tests assert.

Importing this module has no side effect: it does not call :func:`ensure_x64`,
enable the compile cache or query a device. Callers opt in.

References
----------
Design: ``docs/superpowers/specs/2026-10-26-package-jax-first-design.md`` §3;
``docs/architecture/005-adr-kk-jax-only.md``.
"""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
import os
from pathlib import Path
import sys
import threading
from typing import Any

import jax

__all__ = [
    "enable_persistent_cache",
    "ensure_x64",
    "fetch",
    "fetch_status",
    "reset_transfer_counts",
    "thread_transfer_counts",
    "transfer_counts",
    "upload",
    "worker_guard",
]

_counts: dict[str, int] = {"upload": 0, "fetch_status": 0, "fetch": 0}
_counts_lock = threading.Lock()
_thread_counts = threading.local()


_cache_opt_in_checked = False


def ensure_x64() -> None:
    """Enable 64-bit JAX arrays (a no-op when already on).

    Called at every concrete public entry before any upload or trace: with x64
    off a float64 buffer would be canonicalised to float32 inside a later trace.
    The first call also enables the persistent compile cache when the process
    sets ``XRAYLABTOOL_JAX_CACHE=1`` (see :func:`enable_persistent_cache`).

    References
    ----------
    Design spec §4 (``2026-10-25-kk-jax-first-design.md``).

    Examples
    --------
    >>> from xraylabtool.backend.runtime import ensure_x64
    >>> ensure_x64()
    """
    global _cache_opt_in_checked  # noqa: PLW0603
    jax.config.update("jax_enable_x64", True)
    if not _cache_opt_in_checked:
        # Opt-in only: the library never changes global JAX config unasked.
        if os.environ.get("XRAYLABTOOL_JAX_CACHE") == "1":
            enable_persistent_cache()
        _cache_opt_in_checked = True  # after success: a failure is retried, not hidden


_cache_lock = threading.Lock()


def _user_cache_dir(platform: str | None = None) -> Path:
    """OS user-cache directory for the JAX compile cache (``.../xraylabtool/jax``)."""
    platform = sys.platform if platform is None else platform
    home = Path.home()
    if platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or home / "AppData" / "Local")
    elif platform == "darwin":
        base = home / "Library" / "Caches"
    else:
        xdg = Path(os.environ.get("XDG_CACHE_HOME", ""))
        # The XDG spec says to ignore a relative value: it would put a cache that
        # JAX loads compiled programs from under whatever directory we run in.
        base = xdg if xdg.is_absolute() else home / ".cache"
    return base / "xraylabtool" / "jax"


def enable_persistent_cache() -> bool:
    """Turn on JAX's persistent compile cache in the OS user-cache directory.

    Called by the ``calc``, ``batch``, ``compare`` and ``kk`` CLI commands and
    the GUI launcher; the library calls it only when ``XRAYLABTOOL_JAX_CACHE=1``
    is set (a library must not change the host application's global JAX
    configuration unasked). Does nothing when ``XRAYLABTOOL_NO_JAX_CACHE=1`` or when the user
    already set ``jax_compilation_cache_dir``. When it enables the cache it
    also sets ``jax_persistent_cache_min_compile_time_secs`` to 0 (JAX's
    default of 1 s would skip most kk programs) unless the user configured
    that threshold; the lock makes concurrent calls single-flight.

    Returns
    -------
    bool
        Whether this call enabled the cache.

    References
    ----------
    Design spec §10 (``2026-10-25-kk-jax-first-design.md``).

    Examples
    --------
    >>> from xraylabtool.backend.runtime import enable_persistent_cache
    >>> isinstance(enable_persistent_cache.__doc__, str)
    True
    """
    if os.environ.get("XRAYLABTOOL_NO_JAX_CACHE") == "1":
        return False
    with _cache_lock:
        if jax.config.jax_compilation_cache_dir:
            return False
        jax.config.update("jax_compilation_cache_dir", str(_user_cache_dir()))
        # shortcut: JAX's default (1.0) is indistinguishable from an explicit 1.0.
        if (
            "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS" not in os.environ
            and jax.config.jax_persistent_cache_min_compile_time_secs == 1.0
        ):
            jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
        return True


def worker_guard() -> AbstractContextManager[Any]:
    """The transfer-guard scope a worker entry opens around a compute call under tests.

    The guard is thread-local, so every worker thread opens its own. It is on
    when ``XRAYLABTOOL_TRANSFER_GUARD`` is set (the test suite sets it in
    ``tests/conftest.py``; subprocesses inherit it) and a no-op otherwise.

    References
    ----------
    Design spec §1 and §10 (``2026-10-25-kk-jax-first-design.md``).

    Examples
    --------
    >>> from xraylabtool.backend.runtime import worker_guard
    >>> with worker_guard():
    ...     pass
    """
    if os.environ.get("XRAYLABTOOL_TRANSFER_GUARD"):
        return jax.transfer_guard("disallow_explicit")
    return nullcontext()


def _bump(kind: str) -> None:
    with _counts_lock:
        _counts[kind] += 1
    mine = _thread_counts.__dict__.setdefault("counts", dict.fromkeys(_counts, 0))
    mine[kind] += 1


def transfer_counts() -> dict[str, int]:
    """Number of boundary calls so far, by kind (``upload``, ``fetch_status``, ``fetch``)."""
    with _counts_lock:
        return dict(_counts)


def thread_transfer_counts() -> dict[str, int]:
    """Boundary calls made by the calling thread so far, by kind.

    Unlike :func:`transfer_counts` it is not disturbed by other threads (GUI
    workers), so a wrapper can assert its own per-call budget from the difference
    of two readings. :func:`reset_transfer_counts` does not reset it.
    """
    return dict(_thread_counts.__dict__.setdefault("counts", dict.fromkeys(_counts, 0)))


def reset_transfer_counts() -> None:
    """Set all boundary counters to zero."""
    with _counts_lock:
        for kind in _counts:
            _counts[kind] = 0


def upload[T](tree: T) -> T:
    """``jax.device_put`` of a whole pytree: one counted host-to-device transfer."""
    _bump("upload")
    with jax.transfer_guard("allow"):
        return jax.device_put(tree)  # type: ignore[no-any-return]


def fetch_status[T](tree: T) -> T:
    """``jax.device_get`` of a small status pytree: one counted transfer."""
    _bump("fetch_status")
    with jax.transfer_guard("allow"):
        return jax.device_get(tree)  # type: ignore[no-any-return]


def fetch[T](tree: T) -> T:
    """``jax.device_get`` of the final output pytree: one counted transfer."""
    _bump("fetch")
    with jax.transfer_guard("allow"):
        return jax.device_get(tree)  # type: ignore[no-any-return]
