"""Auto-detect the best available compute device."""

from __future__ import annotations

import torch


def get_device(preference: str = "auto") -> torch.device:
    """Return a torch device based on *preference*.

    Parameters
    ----------
    preference:
        ``"auto"`` tries CUDA > MPS > CPU.
        Any other value (``"cuda"``, ``"mps"``, ``"cpu"``) is used directly.
    """
    if preference != "auto":
        return torch.device(preference)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
