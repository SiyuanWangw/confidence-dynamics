"""Process env defaults for SGLang + layer-margin decode hooks."""

from __future__ import annotations

import os


def pin_torch_nccl_libs() -> None:
    """Prefer system NCCL over a mismatched pip nvidia-nccl when present."""
    distro = "/usr/lib/x86_64-linux-gnu"
    if os.path.isdir(distro):
        cur = os.environ.get("LD_LIBRARY_PATH", "")
        parts = [p for p in cur.split(":") if p]
        if distro not in parts:
            os.environ["LD_LIBRARY_PATH"] = ":".join([distro, *parts]) if parts else distro
        elif parts and parts[0] != distro:
            parts = [p for p in parts if p != distro]
            os.environ["LD_LIBRARY_PATH"] = ":".join([distro, *parts])
    os.environ.setdefault("NCCL_NET_PLUGIN", "none")
    os.environ.setdefault("NCCL_CUMEM_ENABLE", "0")


def apply_default_env() -> None:
    pin_torch_nccl_libs()
    os.environ.setdefault("OGF_LAYER_MARGINS", "1")
    os.environ.setdefault("DUMP_STEP_STATS", "1")
    os.environ.setdefault("LOCAL_TAIL_WINDOW", "2048")
    os.environ.setdefault("DEEPCONF_K", "20")
    # Scheduler is a spawned subprocess: it inherits env, not the parent logger.
    os.environ.setdefault("SGLANG_LOG_LEVEL", "error")
