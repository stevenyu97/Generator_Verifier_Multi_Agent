"""Pick a CUDA device: ``TAX_DEVICE`` overrides; else use the GPU with the most free VRAM."""
from __future__ import annotations

import os
from typing import Tuple

try:
    import torch
except ImportError:
    torch = None  # type: ignore


def pick_freest_cuda_device_string() -> str:
    """Return ``cuda:<index>`` for the GPU with the largest free memory, or ``cpu``."""
    if torch is None or not torch.cuda.is_available():
        return "cpu"
    n = torch.cuda.device_count()
    if n <= 0:
        return "cpu"
    best_i = 0
    best_free = -1
    for i in range(n):
        free, _total = _mem_get_info_safe(i)
        if free > best_free:
            best_free = free
            best_i = i
    return f"cuda:{best_i}"


def _mem_get_info_safe(device_index: int) -> Tuple[int, int]:
    assert torch is not None
    try:
        return torch.cuda.mem_get_info(device_index)  # type: ignore[arg-type]
    except Exception:
        try:
            with torch.cuda.device(device_index):
                return torch.cuda.mem_get_info()
        except Exception:
            return (0, 0)


def resolve_torch_device() -> str:
    """
    - If ``TAX_DEVICE`` is set: use it. The special value ``cuda`` (no index) picks the
      freest GPU.
    - If unset and CUDA is available: pick the freest GPU.
    - Otherwise ``cpu``.
    """
    if torch is None:
        return "cpu"
    explicit = os.environ.get("TAX_DEVICE", "").strip()
    if explicit:
        low = explicit.lower()
        if low == "cuda":
            return pick_freest_cuda_device_string()
        if low == "cpu":
            return "cpu"
        if explicit.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "TAX_DEVICE requests CUDA but CUDA is not available. "
                "Install a CUDA build of PyTorch or set TAX_DEVICE=cpu."
            )
        return explicit
    if torch.cuda.is_available():
        return pick_freest_cuda_device_string()
    return "cpu"
