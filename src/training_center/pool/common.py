"""Shared PFSP win-rate tracking for opponent pools."""

from __future__ import annotations

import random
from collections import deque

PFSP_WINDOW = 30


class PFSPMixin:
    """Sliding-window win-rate tracking and PFSP-weighted sampling."""

    win_stats: dict[str, deque]

    def update_stats(self, opponent_name: str, won: bool) -> None:
        """Record a win/loss result."""
        if opponent_name not in self.win_stats:
            self.win_stats[opponent_name] = deque(maxlen=PFSP_WINDOW)
        self.win_stats[opponent_name].append(bool(won))

    def get_win_rate(self, opponent_name: str) -> float:
        """Return sliding-window win rate for an opponent."""
        history = self.win_stats.get(opponent_name)
        if not history:
            return 0.5
        return sum(history) / len(history)

    @staticmethod
    def pfsp_weight(win_rate: float) -> float:
        """PFSP weight: lower win rate → higher sampling probability."""
        return 1.0 - win_rate + 0.1

    def _pfsp_sample(self, candidates: list[str]) -> str:
        """PFSP-weighted sampling from a list of opponent names."""
        weights = [self.pfsp_weight(self.get_win_rate(name)) for name in candidates]
        return random.choices(candidates, weights=weights, k=1)[0]
