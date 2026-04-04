"""Shared PFP win-rate tracking for opponent pools."""

from __future__ import annotations

import random
from collections import deque

PFP_WINDOW = 30


class PFPMixin:
    """Sliding-window win-rate tracking and PFP-weighted sampling."""

    win_stats: dict[str, deque]

    def update_stats(self, opponent_name: str, won: bool) -> None:
        """Record a win/loss result."""
        if opponent_name not in self.win_stats:
            self.win_stats[opponent_name] = deque(maxlen=PFP_WINDOW)
        self.win_stats[opponent_name].append(bool(won))

    def get_win_rate(self, opponent_name: str) -> float:
        """Return sliding-window win rate for an opponent."""
        history = self.win_stats.get(opponent_name)
        if not history:
            return 0.5
        return sum(history) / len(history)

    @staticmethod
    def pfp_weight(win_rate: float) -> float:
        """PFP weight: lower win rate → higher sampling probability."""
        return 1.0 - win_rate + 0.1

    def _pfp_sample(self, candidates: list[str]) -> str:
        """PFP-weighted sampling from a list of opponent names."""
        weights = [self.pfp_weight(self.get_win_rate(name)) for name in candidates]
        return random.choices(candidates, weights=weights, k=1)[0]
