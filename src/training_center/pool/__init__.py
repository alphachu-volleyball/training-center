"""Opponent pool modules for self-play and curriculum training."""

from training_center.pool.curriculum import CurriculumPool
from training_center.pool.opponent import OpponentPool, make_opponent_policy

__all__ = ["CurriculumPool", "OpponentPool", "make_opponent_policy"]
