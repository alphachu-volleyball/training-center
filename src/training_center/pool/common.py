"""Shared PFP win-rate tracking for opponent pools.

Win rates are stored as a single combined number per opponent — set once per
evaluation batch (e.g., 100 games) rather than as a rolling per-game deque.
The previous design (deque maxlen=30) made sense when eval batches were
smaller than the window (~10 games, accumulating across multiple evals),
but with the current 100-game eval batches each evaluation overflowed the
window, leaving only a noisy 30-game tail. That produced spurious unlock
decisions on lucky end-of-batch streaks (see iter 40 of experiment 027,
where builtin's combined 49% was masked by a sliding 93% on the last 30
games).
"""

from __future__ import annotations

import random


class PFPMixin:
    """Combined win-rate tracking and PFP-weighted sampling.

    ``win_rates`` stores the most recent eval-batch combined win rate per
    opponent (float in [0, 1]). Subclasses must initialize it as an empty
    dict in ``__init__``.
    """

    win_rates: dict[str, float]

    def set_win_rate(self, opponent_name: str, win_rate: float) -> None:
        """Record the latest combined win rate from an evaluation batch."""
        self.win_rates[opponent_name] = float(win_rate)

    def get_win_rate(self, opponent_name: str) -> float:
        """Return the latest combined win rate, or 0.5 if never evaluated."""
        return self.win_rates.get(opponent_name, 0.5)

    @staticmethod
    def pfp_weight(win_rate: float) -> float:
        """PFP weight: lower win rate → higher sampling probability."""
        return 1.0 - win_rate + 0.1

    def _pfp_sample(self, candidates: list[str]) -> str:
        """PFP-weighted sampling from a list of opponent names."""
        weights = [self.pfp_weight(self.get_win_rate(name)) for name in candidates]
        return random.choices(candidates, weights=weights, k=1)[0]
