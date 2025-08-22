"""Simple file I/O utilities with optional HDFS support (stub implementation)."""

import os
import shutil
from typing import Optional


def exists(path: str, **kwargs) -> bool:
    """Check if a path exists. For now, only supports local filesystem."""
    return os.path.exists(path)


def makedirs(name: str, mode: int = 0o777, exist_ok: bool = False, **kwargs) -> None:
    """Create directory and all intermediate ones. For now, only supports local filesystem."""
    os.makedirs(name, mode=mode, exist_ok=exist_ok)


def copy(src: str, dst: str, **kwargs):
    """Copy files or directories. For now, only supports local filesystem."""
    if os.path.isdir(src):
        return shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        return shutil.copy2(src, dst)


