"""Physical GPU resolution and the forbidden-GPU guard.

A torch device 'cuda:i' is logical: CUDA renumbers the devices listed in CUDA_VISIBLE_DEVICES from 0.
physical_gpu resolves it to the nvidia-smi index (CUDA_DEVICE_ORDER=PCI_BUS_ID, the nvidia-smi
order); check_device refuses a device whose physical index is in settings.forbidden_gpus (a shared
server lists other users' GPUs in the gitignored configs/local.yaml) or cannot be resolved (index not
visible, a UUID / MIG entry, another device order). 'cpu' needs no resolution and is always allowed.
"""

from __future__ import annotations

import os

ORDER_ENV, ORDER = "CUDA_DEVICE_ORDER", "PCI_BUS_ID"


class ForbiddenDevice(RuntimeError):
    """A device that is a forbidden physical GPU, or whose physical index cannot be resolved."""


def _index(device):
    """Logical CUDA index of `device` (int, 'cuda', 'cuda:i' or a torch.device); None for the CPU."""
    if isinstance(device, int):
        return device
    dev = str(device).strip()
    if dev == "cpu":
        return None
    if dev == "cuda":
        return 0
    if dev.startswith("cuda:") and dev[5:].isdigit():
        return int(dev[5:])
    raise ValueError(f"device {device!r}: expected 'cpu', 'cuda' or 'cuda:<index>'")


def physical_gpu(device):
    """nvidia-smi index behind a logical CUDA device; None for 'cpu' and for an index that cannot be
    told (not visible, or a UUID / MIG entry of CUDA_VISIBLE_DEVICES)."""
    i = _index(device)
    if i is None:
        return None
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is None:
        return i
    vis = [x.strip() for x in cvd.split(",") if x.strip()]
    return int(vis[i]) if 0 <= i < len(vis) and vis[i].isdigit() else None


def forbidden_gpus(settings=None):
    if settings is None:
        from habitus.config import settings as _settings
        settings = _settings()
    return {int(x) for x in (getattr(settings, "forbidden_gpus", None) or ())}


def check_device(device, settings=None):
    """Return `device` if it may be used; raise ForbiddenDevice for a forbidden or unresolvable GPU.
    For a CUDA device CUDA_DEVICE_ORDER must be PCI_BUS_ID (set here when unset, before CUDA starts)."""
    i = _index(device)
    if i is None:
        return device
    order = os.environ.get(ORDER_ENV)
    if order is None:
        os.environ[ORDER_ENV] = ORDER
    elif order != ORDER:
        raise ForbiddenDevice(f"device {device}: {ORDER_ENV}={order!r}, physical indices need {ORDER}")
    phys = physical_gpu(i)
    if phys is None:
        raise ForbiddenDevice(f"device {device}: no physical index under CUDA_VISIBLE_DEVICES="
                              f"{os.environ.get('CUDA_VISIBLE_DEVICES')!r}")
    bad = forbidden_gpus(settings)
    if phys in bad:
        raise ForbiddenDevice(f"device {device} is physical GPU {phys}, which is forbidden on this server "
                              f"(forbidden_gpus {sorted(bad)})")
    return device
