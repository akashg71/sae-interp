"""Device selection.

Everything in this repo is device-agnostic. We prefer CUDA (cloud GPU, e.g. Colab),
then Apple-Silicon MPS (the author's Mac), then CPU. Never hardcode "cuda" elsewhere —
pass the result of ``get_device()`` through to models and tensors.
"""

from __future__ import annotations

import torch


def get_device(prefer: str | None = None) -> torch.device:
    """Return the best available torch device.

    Order of preference: CUDA -> MPS (Apple Silicon) -> CPU.

    Parameters
    ----------
    prefer:
        Optional explicit override, e.g. "cpu" to force CPU even when a GPU exists
        (useful for debugging numerical issues or when MPS lacks an op). If the
        preferred device isn't available we fall back to auto-detection.
    """
    if prefer is not None:
        prefer = prefer.lower()
        if prefer == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if prefer == "mps" and _mps_available():
            return torch.device("mps")
        if prefer == "cpu":
            return torch.device("cpu")
        # Fall through to auto-detect if the preference can't be honoured.

    if torch.cuda.is_available():
        return torch.device("cuda")
    if _mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def _mps_available() -> bool:
    """True if Apple-Silicon MPS backend is built and available."""
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def device_str() -> str:
    """Convenience: the chosen device as a plain string (e.g. for SAELens, which
    sometimes wants a string rather than a torch.device)."""
    return str(get_device())
