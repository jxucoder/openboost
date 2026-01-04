"""Backend detection and dispatch for OpenBoost."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Literal

# Backend state
_BACKEND: Literal["cuda", "cpu"] | None = None


def get_backend() -> Literal["cuda", "cpu"]:
    """Get the current compute backend.
    
    Returns:
        "cuda" if NVIDIA GPU is available, "cpu" otherwise.
    """
    global _BACKEND
    
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
    
    _BACKEND = backend


def is_cuda() -> bool:
    """Check if using CUDA backend."""
    return get_backend() == "cuda"


def is_cpu() -> bool:
    """Check if using CPU backend."""
    return get_backend() == "cpu"

