"""Curriculum-based opponent pool for progressive difficulty training.

Unlike OpponentPool (which manages model checkpoint files), CurriculumPool
manages a ladder of named AI specs (e.g. "builtin", "duckll:5") that are
progressively unlocked as the learner improves.

The special spec ``"self"`` represents the learner's own past checkpoints
(self-play). It only makes sense for universal models (--side both) and
is excluded from unlock-readiness checks since its win rate hovers
around 50% by definition.
"""

from __future__ import annotations

from training_center.pool.common import PFPMixin

SELF_ENTRY = "self"


class CurriculumPool(PFPMixin):
    """PFP opponent pool with unlock-gated difficulty ladder.

    Opponents are unlocked in order when the minimum win rate across
    all currently unlocked opponents (excluding "self") exceeds the
    unlock threshold. Win rates come from ``set_win_rate`` (called per
    evaluation batch).
    """

    def __init__(self, ladder: list[str], unlock_threshold: float = 0.7) -> None:
        self.ladder = ladder
        self.unlock_threshold = unlock_threshold
        self.unlocked: list[str] = []
        self.win_rates: dict[str, float] = {}

    def force_unlock(self, index: int) -> str:
        """Unlock the opponent at the given ladder index."""
        name = self.ladder[index]
        if name not in self.unlocked:
            self.unlocked.append(name)
        return name

    def try_unlock(self) -> str | None:
        """Unlock the next opponent if all current ones are mastered.

        ``"self"`` entries are skipped in the readiness check (self vs
        self ~= 50% by definition, so including it would block all
        further unlocks).

        Returns the newly unlocked opponent name, or None.
        """
        if len(self.unlocked) >= len(self.ladder):
            return None

        for name in self.unlocked:
            if name == SELF_ENTRY:
                continue
            if name not in self.win_rates:
                return None  # not yet evaluated this round
            if self.get_win_rate(name) < self.unlock_threshold:
                return None

        next_idx = len(self.unlocked)
        return self.force_unlock(next_idx)

    def sample_opponent(self) -> str:
        """PFP-weighted sampling from unlocked pool."""
        if not self.unlocked:
            return self.ladder[0]
        return self._pfp_sample(self.unlocked)

    def status(self) -> dict:
        """Return pool status for logging.

        Win-rate aggregates exclude the ``"self"`` entry since it is
        always around 50% and would skew progress signals.
        """
        win_rates = {name: self.get_win_rate(name) for name in self.unlocked if name != SELF_ENTRY}
        return {
            "pool_size": len(self.unlocked),
            "highest_unlocked": self.unlocked[-1] if self.unlocked else None,
            "min_win_rate": min(win_rates.values()) if win_rates else 0,
            "avg_win_rate": sum(win_rates.values()) / len(win_rates) if win_rates else 0,
            "per_opponent": win_rates,
        }
