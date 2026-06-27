"""Backend detection and dispatch for OpenBoost."""

from __future__ import annotations

import os
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Literal

# Backend state
_BACKEND: Literal["cuda", "cpu"] | None = None
_BACKEND_LOCK = threading.Lock()


def get_backend() -> Literal["cuda", "cpu"]:
    """Get the current compute backend.

    Returns:
        "cuda" if NVIDIA GPU is available, "cpu" otherwise.
    """
    global _BACKEND

    with _BACKEND_LOCK:
        if _BACKEND is not None:
            return _BACKEND

        # Allow override via environment variable
        env_backend = os.environ.get("OPENBOOST_BACKEND", "").lower()
        if env_backend in ("cuda", "cpu"):
            _BACKEND = env_backend
            return _BACKEND

        # Auto-detect CUDA
        _BACKEND = "cuda" if _cuda_available() else "cpu"
        return _BACKEND


def _cuda_available() -> bool:
    """Check if CUDA is available via Numba."""
    try:
        from numba import cuda
        return cuda.is_available()
    except Exception:
        return False


def set_backend(backend: Literal["cuda", "cpu"]) -> None:
    """Force a specific backend.

    Thread-safe: uses a lock to prevent concurrent modification.

    Args:
        backend: "cuda" or "cpu"

    Raises:
        ValueError: If backend is not "cuda" or "cpu"
        RuntimeError: If CUDA is requested but not available
    """
    global _BACKEND

    if backend not in ("cuda", "cpu"):
        raise ValueError(f"backend must be 'cuda' or 'cpu', got {backend!r}")

    if backend == "cuda" and not _cuda_available():
        raise RuntimeError("CUDA backend requested but CUDA is not available")

    with _BACKEND_LOCK:
        _BACKEND = backend


def is_cuda() -> bool:
    """Check if using CUDA backend."""
    return get_backend() == "cuda"


def is_cpu() -> bool:
    """Check if using CPU backend."""
    return get_backend() == "cpu"


def clear_tree_workspace_cache() -> None:
    """Free cached GPU tree-building workspace arrays (no-op on CPU).

    Call this between training runs that use different data shapes to avoid
    holding onto stale GPU allocations. Safe to call on CPU-only installs,
    where it does nothing.
    """
    if get_backend() != "cuda":
        return
    from ._cuda import clear_tree_workspace_cache as _clear
    _clear()


class backend_context:
    """Context manager for temporarily switching the compute backend.

    Restores the previous backend on exit, even if an exception occurs. If no
    backend had been resolved yet on entry, exit restores the unresolved state
    so the backend is auto-detected again on next use.

    Note:
        This mutates the **process-global** backend, not a thread-local one.
        It is therefore not safe to use concurrently from multiple threads —
        one thread's context will affect every other thread.

    Example::

        with backend_context('cpu'):
            model.fit(X, y)  # Forces CPU
        # Original backend is restored here
    """

    def __init__(self, backend: Literal["cuda", "cpu"]) -> None:
        self._backend = backend
        self._previous: Literal["cuda", "cpu"] | None = None

    def __enter__(self) -> None:
        with _BACKEND_LOCK:
            self._previous = _BACKEND
        set_backend(self._backend)

    def __exit__(self, *exc: object) -> None:
        global _BACKEND
        # Restore by direct assignment (set_backend only writes the global and
        # rejects None; direct assignment also lets us restore the
        # "unresolved -> auto-detect" state when _previous is None).
        with _BACKEND_LOCK:
            _BACKEND = self._previous

