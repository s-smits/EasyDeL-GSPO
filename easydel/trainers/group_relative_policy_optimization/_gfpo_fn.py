# _gfpo_fn.py

# GFPO uses the same step function as GRPO. Filtering and per-prompt normalization
# are handled in the trainer preprocessing by modifying the advantages.
from ._fn import grpo_step as gfpo_step


