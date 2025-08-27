"""GFSPO stable wrapper (deprecated), redirects to unified SPOTrainer."""

from __future__ import annotations

from .spo_trainer import SPOTrainer


class GFSPOStableTrainer(SPOTrainer):
    """Deprecated wrapper: use SPOTrainer. Kept for backward compatibility."""


def trainer(**kwargs) -> GFSPOStableTrainer:
    return GFSPOStableTrainer(**kwargs)

# Backward-compat alias
GFSPOWShrinkTrainer = GFSPOStableTrainer
