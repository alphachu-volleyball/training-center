"""Curriculum-based opponent pool for progressive difficulty training.

Unlike OpponentPool (which manages model checkpoint files), CurriculumPool
manages a ladder of named AI specs (e.g. "builtin", "duckll:5") that are
progressively unlocked as the learner improves.
"""

from __future__ import annotations

from collections import deque

from training_center.pool.common import PFSP_WINDOW, PFSPMixin


class CurriculumPool(PFSPMixin):
    """PFSP opponent pool with unlock-gated difficulty ladder.

    Opponents are unlocked in order when the minimum win rate across
    all currently unlocked opponents exceeds the unlock threshold.
    """

    def __init__(self, ladder: list[str], unlock_threshold: float = 0.7) -> None:
        self.ladder = ladder
        self.unlock_threshold = unlock_threshold
        self.unlocked: list[str] = []
        self.win_stats: dict[str, deque] = {}

    def force_unlock(self, index: int) -> str:
        """Unlock the opponent at the given ladder index."""
        name = self.ladder[index]
        if name not in self.unlocked:
            self.unlocked.append(name)
            self.win_stats[name] = deque(maxlen=PFSP_WINDOW)
        return name

    def try_unlock(self) -> str | None:
        """Unlock the next opponent if all current ones are mastered.

        Returns the newly unlocked opponent name, or None.
        """
        if len(self.unlocked) >= len(self.ladder):
            return None

        for name in self.unlocked:
            stats = self.win_stats.get(name)
            if not stats:
                return None
            if self.get_win_rate(name) < self.unlock_threshold:
                return None

        next_idx = len(self.unlocked)
        return self.force_unlock(next_idx)

    def sample_opponent(self) -> str:
        """PFSP-weighted sampling from unlocked pool."""
        if not self.unlocked:
            return self.ladder[0]
        return self._pfsp_sample(self.unlocked)

    def status(self) -> dict:
        """Return pool status for logging."""
        win_rates = {name: self.get_win_rate(name) for name in self.unlocked}
        return {
            "pool_size": len(self.unlocked),
            "highest_unlocked": self.unlocked[-1] if self.unlocked else None,
            "min_win_rate": min(win_rates.values()) if win_rates else 0,
            "avg_win_rate": sum(win_rates.values()) / len(win_rates) if win_rates else 0,
            "per_opponent": win_rates,
        }
