from __future__ import annotations

from typing import Literal

import torch

DeviceName = Literal["auto", "cpu", "cuda", "mps"]


def resolve_device(preferred: DeviceName, *, allow_mps: bool = True) -> torch.device:
    if preferred == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if allow_mps and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if preferred == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("[device] Requested cuda but it is unavailable, fallback to cpu.")
        return torch.device("cpu")

    if preferred == "mps":
        if allow_mps and torch.backends.mps.is_available():
            return torch.device("mps")
        print("[device] Requested mps but it is unavailable, fallback to cpu.")
        return torch.device("cpu")

    return torch.device("cpu")


def device_name(device: torch.device) -> str:
    return str(device)
