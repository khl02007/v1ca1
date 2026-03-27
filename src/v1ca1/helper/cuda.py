from __future__ import annotations

"""Helpers for early CUDA visibility configuration in CLI scripts."""

import argparse
import os
import sys
from typing import Sequence


def extract_cuda_visible_devices_argument(
    argv: Sequence[str],
) -> tuple[str | None, list[str]]:
    """Return `--cuda-visible-devices` plus the remaining CLI arguments."""
    argv_list = list(argv)
    if not argv_list:
        return None, []

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--cuda-visible-devices", dest="cuda_visible_devices")
    args, remaining = parser.parse_known_args(argv_list[1:])
    return args.cuda_visible_devices, [argv_list[0], *remaining]


def pop_cuda_visible_devices_argument(argv: Sequence[str] | None = None) -> str | None:
    """Remove `--cuda-visible-devices` from `sys.argv` and return its value."""
    current_argv = list(sys.argv if argv is None else argv)
    cuda_visible_devices, remaining = extract_cuda_visible_devices_argument(current_argv)
    if argv is None:
        sys.argv = remaining
    return cuda_visible_devices


def configure_cuda_visible_devices(cuda_visible_devices: str | None) -> None:
    """Apply `CUDA_VISIBLE_DEVICES` when one explicit value was requested."""
    if cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)
