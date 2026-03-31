"""PFSP-based opponent pool management for self-play."""

from __future__ import annotations

import os
import random
from collections import deque
from collections.abc import Callable

import numpy as np
from pika_zoo.ai.protocol import AIPolicy
from stable_baselines3 import PPO

PFSP_WINDOW = 30


def make_opponent_policy(model: PPO) -> Callable[[np.ndarray], int]:
    """Wrap an SB3 model as a callable (obs) -> action."""

    def policy(obs: np.ndarray) -> int:
        action, _ = model.predict(obs, deterministic=True)
        return int(action)

    return policy


class OpponentPool:
    """PFSP opponent pool with sliding-window win-rate tracking.

    Opponent selection:
    - anchor_prob: probability of choosing the anchor AI (e.g., builtin, duckll)
    - remaining: PFSP-weighted sampling from pool (lower win-rate = higher weight)
    """

    def __init__(self, pool_dir: str, side: str, anchor: AIPolicy | None = None) -> None:
        self.pool_dir = pool_dir
        self.side = side
        self.anchor = anchor
        self.anchor_name = type(anchor).__name__ if anchor else "none"
        self.checkpoints: list[str] = []
        self.win_stats: dict[str, deque] = {}
        os.makedirs(pool_dir, exist_ok=True)

    def add_checkpoint(self, model: PPO, iteration: int) -> str:
        path = os.path.join(self.pool_dir, f"{self.side}_iter{iteration:06d}")
        model.save(path)
        name = os.path.basename(path)
        self.checkpoints.append(path)
        self.win_stats[name] = deque(maxlen=PFSP_WINDOW)
        return path

    def sample_opponent(
        self,
        latest_model: PPO,
        anchor_prob: float = 0.2,
    ) -> tuple[AIPolicy | PPO, str, bool]:
        """Sample an opponent. Returns (policy_or_model, name, is_anchor)."""
        r = random.random()

        if r < anchor_prob and self.anchor is not None:
            return self.anchor, self.anchor_name, True

        if not self.checkpoints:
            return latest_model, "latest", False

        weights = self._pfsp_weights()
        idx = random.choices(range(len(self.checkpoints)), weights=weights, k=1)[0]
        path = self.checkpoints[idx]
        name = os.path.basename(path)
        model = PPO.load(path, device="cpu")
        return model, name, False

    def update_stats(self, opponent_name: str, won: bool) -> None:
        if opponent_name not in self.win_stats:
            self.win_stats[opponent_name] = deque(maxlen=PFSP_WINDOW)
        self.win_stats[opponent_name].append(bool(won))

    def get_win_rate(self, opponent_name: str) -> float:
        history = self.win_stats.get(opponent_name)
        if not history:
            return 0.5
        return sum(history) / len(history)

    def _pfsp_weights(self) -> list[float]:
        weights = []
        for path in self.checkpoints:
            name = os.path.basename(path)
            win_rate = self.get_win_rate(name)
            weights.append(1.0 - win_rate + 0.1)
        return weights
