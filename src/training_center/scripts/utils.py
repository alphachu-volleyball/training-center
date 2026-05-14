"""Shared utilities for training and evaluation scripts."""

from __future__ import annotations

import os
import signal
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from pika_zoo.env.pikachu_volleyball import NoiseConfig
from pika_zoo.records.types import GameRecord, GamesRecord

from training_center.metrics import compute_eval_metrics

NOISE_LEVELS: dict[int, tuple[int, int, int]] = {
    0: (0, 0, 0),
    1: (5, 3, 1),
    2: (10, 5, 2),
    3: (20, 10, 3),
    4: (35, 15, 4),
    5: (50, 20, 5),
}


def parse_noise(
    noise_level: int | None,
    noise_x: int | None = None,
    noise_x_vel: int | None = None,
    noise_y_vel: int | None = None,
) -> NoiseConfig | None:
    """Parse noise arguments into a NoiseConfig."""
    if noise_level is not None and noise_level > 0:
        x, xv, yv = NOISE_LEVELS[noise_level]
        return NoiseConfig(x_range=x, x_velocity_range=xv, y_velocity_range=yv)
    if noise_x is not None or noise_x_vel is not None or noise_y_vel is not None:
        return NoiseConfig(
            x_range=noise_x or 0,
            x_velocity_range=noise_x_vel or 0,
            y_velocity_range=noise_y_vel or 0,
        )
    return None


def worker_init() -> None:
    """Ignore SIGINT in worker processes so only the main process handles it."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def shutdown_executor(executor: ProcessPoolExecutor) -> None:
    """Shut down executor and terminate all worker processes."""
    if executor._processes:
        for proc in executor._processes.values():
            if proc.is_alive():
                proc.terminate()
    executor.shutdown(wait=False, cancel_futures=True)


def setup_graceful_shutdown() -> None:
    """Set up signal handlers for graceful shutdown.

    SIGTERM → SystemExit (triggers try/finally cleanup).
    SIGINT → SIGTERM (prevents SIGINT from reaching worker processes via
    process group broadcast, which corrupts executor internal state).
    """
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))
    signal.signal(signal.SIGINT, lambda *_: os.kill(os.getpid(), signal.SIGTERM))


EVAL_METRIC_KEYS = [
    "win_rate",
    "avg_score",
    "avg_opp_score",
    "avg_p1_score",
    "var_p1_score",
    "avg_p2_score",
    "var_p2_score",
    "avg_game_frames",
    "var_game_frames",
    "serve_win_rate",
    "receive_win_rate",
    "avg_round_frames",
    "std_round_frames",
    "action_entropy",
    "power_hit_rate",
    "ball_own_side_ratio",
    "serve_avg_round_frames",
    "receive_avg_round_frames",
]


def _mean_var(values: list[int | float]) -> tuple[float, float]:
    """Return population mean and variance for a metric sample."""
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return float(mean), float(variance)


@dataclass
class EvalSummary:
    """Structured result from an evaluation batch.

    Keeps per-game samples for recomputing aggregate stats, while exposing
    dict-like reads for the older training code paths that still index by key.
    """

    wins: int
    losses: int
    game_winners: list[str]
    p1_scores: list[int | float] = field(default_factory=list)
    p2_scores: list[int | float] = field(default_factory=list)
    game_frames: list[int | float] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_episodes(cls, episodes: list[GameRecord], model_side: str) -> EvalSummary:
        """Build an eval summary from recorded games and the model's side."""
        model_idx = 0 if model_side == "player_1" else 1
        opp_idx = 1 - model_idx
        p1_scores = [e.scores[0] for e in episodes]
        p2_scores = [e.scores[1] for e in episodes]
        game_frames = [e.num_frames for e in episodes]
        wins = sum(1 for e in episodes if e.winner == model_side)

        model_scores = p1_scores if model_idx == 0 else p2_scores
        opp_scores = p1_scores if opp_idx == 0 else p2_scores
        avg_score, _ = _mean_var(model_scores)
        avg_opp_score, _ = _mean_var(opp_scores)

        metrics = {
            "avg_score": avg_score,
            "avg_opp_score": avg_opp_score,
            **compute_eval_metrics(GamesRecord(games=episodes), model_side),
        }
        return cls(
            wins=wins,
            losses=len(episodes) - wins,
            game_winners=[e.winner for e in episodes],
            p1_scores=p1_scores,
            p2_scores=p2_scores,
            game_frames=game_frames,
            metrics=metrics,
        )

    @classmethod
    def from_mapping(cls, result: EvalSummary | dict[str, Any]) -> EvalSummary:
        """Coerce an existing result mapping into EvalSummary."""
        if isinstance(result, EvalSummary):
            return result
        game_winners = list(result["game_winners"])
        wins = int(result.get("wins", 0))
        losses = int(result.get("losses", max(len(game_winners) - wins, 0)))
        metrics = {
            k: float(v)
            for k, v in result.items()
            if k in EVAL_METRIC_KEYS and k not in {"win_rate"} and isinstance(v, int | float)
        }
        return cls(
            wins=wins,
            losses=losses,
            game_winners=game_winners,
            p1_scores=list(result.get("p1_scores", [])),
            p2_scores=list(result.get("p2_scores", [])),
            game_frames=list(result.get("game_frames", [])),
            metrics=metrics,
        )

    @property
    def games(self) -> int:
        return self.wins + self.losses

    @property
    def win_rate(self) -> float:
        return self.wins / self.games if self.games else 0.0

    def to_log_dict(self) -> dict[str, Any]:
        """Return the legacy flat mapping used by W&B logging and ELO code."""
        avg_p1_score, var_p1_score = _mean_var(self.p1_scores)
        avg_p2_score, var_p2_score = _mean_var(self.p2_scores)
        avg_game_frames, var_game_frames = _mean_var(self.game_frames)
        return {
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": self.win_rate,
            "avg_p1_score": avg_p1_score,
            "var_p1_score": var_p1_score,
            "avg_p2_score": avg_p2_score,
            "var_p2_score": var_p2_score,
            "avg_game_frames": avg_game_frames,
            "var_game_frames": var_game_frames,
            "p1_scores": self.p1_scores,
            "p2_scores": self.p2_scores,
            "game_frames": self.game_frames,
            "game_winners": self.game_winners,
            **self.metrics,
        }

    def format_score_frame_line(self, label: str, *, indent: str = "    ", include_vs: bool = True) -> str:
        """Format wins, scores, and game length for console eval output."""
        data = self.to_log_dict()
        label_text = f"vs {label}" if include_vs else label
        return (
            f"{indent}{label_text}: {self.wins}W {self.losses}L "
            f"({data['avg_p1_score']:.1f} ± {data['var_p1_score']:.1f} "
            f"vs {data['avg_p2_score']:.1f} ± {data['var_p2_score']:.1f}, "
            f"frames: {data['avg_game_frames']:.0f} ± {data['var_game_frames']:.0f})"
        )

    def __getitem__(self, key: str) -> Any:
        return self.to_log_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_log_dict().get(key, default)

    def __contains__(self, key: str) -> bool:
        return key in self.to_log_dict()


def build_eval_log_data(
    results: dict[str, EvalSummary | dict],
    prefix: str,
) -> dict[str, float]:
    """Build a wandb log_data dict from eval results.

    Args:
        results: {opponent_name: {metric: value, ...}, ...}
        prefix: Key prefix (e.g. "eval", "p1/eval", "curriculum").

    Returns:
        Flat dict like {"eval/vs_builtin/win_rate": 0.8, ...}
    """
    log_data: dict[str, float] = {}
    for opp_name, r in results.items():
        data = r.to_log_dict() if isinstance(r, EvalSummary) else r
        for k in EVAL_METRIC_KEYS:
            if k in data:
                log_data[f"{prefix}/vs_{opp_name}/{k}"] = data[k]
    return log_data


def combine_per_side_results(p1_result: EvalSummary | dict, p2_result: EvalSummary | dict) -> EvalSummary:
    """Combine per-side eval results from a universal model into an aggregate.

    Win counts come from re-interpreting each side's game_winners (model wins
    when winner == its side). Numeric metrics are averaged across the two
    sides — exact when both ran the same number of games, which is how callers
    invoke this. The combined dict mirrors the schema produced by
    `_eval_matchup_worker` so it plugs into existing logging and pool-update
    paths unchanged.
    """
    p1_summary = EvalSummary.from_mapping(p1_result)
    p2_summary = EvalSummary.from_mapping(p2_result)
    p1_won = [w == "player_1" for w in p1_summary.game_winners]
    p2_won = [w == "player_2" for w in p2_summary.game_winners]
    won_list = p1_won + p2_won
    total = len(won_list)
    wins = sum(won_list)

    metrics: dict[str, float] = {}
    p1_data = p1_summary.to_log_dict()
    p2_data = p2_summary.to_log_dict()
    for k in EVAL_METRIC_KEYS:
        if k in {
            "win_rate",
            "avg_p1_score",
            "var_p1_score",
            "avg_p2_score",
            "var_p2_score",
            "avg_game_frames",
            "var_game_frames",
        }:
            continue  # computed from counts or per-game samples
        v1 = p1_data.get(k)
        v2 = p2_data.get(k)
        if isinstance(v1, int | float) and isinstance(v2, int | float):
            metrics[k] = (v1 + v2) / 2

    return EvalSummary(
        wins=wins,
        losses=total - wins,
        game_winners=p1_summary.game_winners + p2_summary.game_winners,
        p1_scores=p1_summary.p1_scores + p2_summary.p1_scores,
        p2_scores=p1_summary.p2_scores + p2_summary.p2_scores,
        game_frames=p1_summary.game_frames + p2_summary.game_frames,
        metrics=metrics,
    )


def model_won_per_game(result: EvalSummary | dict, model_side: str) -> list[bool]:
    """Convert an eval result's game_winners into a list of model-victory bools."""
    summary = EvalSummary.from_mapping(result)
    return [w == model_side for w in summary.game_winners]


def record_video(model_path: str, side: str, opponent: str, output_path: str) -> None:
    """Record a sample game video using pika-zoo's play script.

    If ``model_path`` points to a ``.zip`` file, the parent directory is
    passed to ``play`` instead, so ``model.json`` (side, observation_simplified,
    ...) is honored. Passing the bare ``.zip`` makes pika-zoo's loader skip the
    metadata, which silently breaks universal models on the side opposite their
    training side.
    """
    from pathlib import Path

    from pika_zoo.scripts.play import play

    mp = Path(model_path)
    if mp.is_file() and mp.suffix == ".zip":
        model_path = str(mp.parent)

    p1 = model_path if side == "player_1" else opponent
    p2 = opponent if side == "player_1" else model_path
    play(p1=p1, p2=p2, winning_score=5, render=False, record=output_path, seed=0)
