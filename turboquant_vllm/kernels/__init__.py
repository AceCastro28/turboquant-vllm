"""
Kernel dispatch: use Triton if available and GPU is present, else PyTorch fallback.
"""
from __future__ import annotations

import torch

_TRITON_AVAILABLE = False
try:
    import triton  # noqa: F401
    _TRITON_AVAILABLE = True
except ImportError:
    pass


def use_triton() -> bool:
    return _TRITON_AVAILABLE and torch.cuda.is_available()
