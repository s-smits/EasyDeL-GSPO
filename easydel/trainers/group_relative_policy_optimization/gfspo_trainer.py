"""GFSPO unified wrapper (deprecated), redirects to SPOTrainer implementation."""

from __future__ import annotations

from .spo_trainer import SPOTrainer
from .gfspo_config import GFSPOConfig


class GFSPOTrainer(SPOTrainer):
    """Deprecated wrapper: use SPOTrainer. Kept for backward compatibility."""
    arguments: GFSPOConfig  # type hinting


def trainer(**kwargs) -> GFSPOTrainer:
    return GFSPOTrainer(**kwargs)

