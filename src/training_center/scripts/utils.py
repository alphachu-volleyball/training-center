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
    """Pure statistics from an evaluation batch."""

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

    @property
    def games(self) -> int:
        return self.wins + self.losses

    @property
    def win_rate(self) -> float:
        return self.wins / self.games if self.games else 0.0

    @property
    def avg_p1_score(self) -> float:
        return _mean_var(self.p1_scores)[0]

    @property
    def var_p1_score(self) -> float:
        return _mean_var(self.p1_scores)[1]

    @property
    def avg_p2_score(self) -> float:
        return _mean_var(self.p2_scores)[0]

    @property
    def var_p2_score(self) -> float:
        return _mean_var(self.p2_scores)[1]

    @property
    def avg_game_frames(self) -> float:
        return _mean_var(self.game_frames)[0]

    @property
    def var_game_frames(self) -> float:
        return _mean_var(self.game_frames)[1]

    def metric(self, key: str, default: float = 0.0) -> float:
        """Return a named derived metric."""
        return self.metrics.get(key, default)

    def to_log_dict(self) -> dict[str, Any]:
        """Return the flat mapping used by W&B logging and table exports."""
        return {
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": self.win_rate,
            "avg_p1_score": self.avg_p1_score,
            "var_p1_score": self.var_p1_score,
            "avg_p2_score": self.avg_p2_score,
            "var_p2_score": self.var_p2_score,
            "avg_game_frames": self.avg_game_frames,
            "var_game_frames": self.var_game_frames,
            "p1_scores": self.p1_scores,
            "p2_scores": self.p2_scores,
            "game_frames": self.game_frames,
            "game_winners": self.game_winners,
            **self.metrics,
        }

    def format_score_frame_line(self, label: str, *, indent: str = "    ", include_vs: bool = True) -> str:
        """Format wins, scores, and game length for console eval output."""
        label_text = f"vs {label}" if include_vs else label
        return (
            f"{indent}{label_text}: {self.wins}W {self.losses}L "
            f"({self.avg_p1_score:.1f} ± {self.var_p1_score:.1f} "
            f"vs {self.avg_p2_score:.1f} ± {self.var_p2_score:.1f}, "
            f"frames: {self.avg_game_frames:.0f} ± {self.var_game_frames:.0f})"
        )


@dataclass
class EvalResult:
    """One model-vs-opponent evaluation record with identity and summary stats."""

    model_name: str
    opponent_name: str
    model_side: str
    opponent_side: str
    summary: EvalSummary
    matchup_name: str | None = None
    model_path: str | None = None
    opponent_spec: str | None = None
    winning_score: int | None = None
    iteration: int | None = None
    step: int | None = None
    seed: int | None = None

    @classmethod
    def from_episodes(
        cls,
        episodes: list[GameRecord],
        *,
        model_name: str,
        opponent_name: str,
        model_side: str,
        opponent_side: str,
        matchup_name: str | None = None,
        model_path: str | None = None,
        opponent_spec: str | None = None,
        winning_score: int | None = None,
        iteration: int | None = None,
        step: int | None = None,
        seed: int | None = None,
    ) -> EvalResult:
        """Build an identified eval result from recorded games."""
        return cls(
            model_name=model_name,
            opponent_name=opponent_name,
            model_side=model_side,
            opponent_side=opponent_side,
            summary=EvalSummary.from_episodes(episodes, model_side),
            matchup_name=matchup_name,
            model_path=model_path,
            opponent_spec=opponent_spec,
            winning_score=winning_score,
            iteration=iteration,
            step=step,
            seed=seed,
        )

    @property
    def games(self) -> int:
        return self.summary.games

    @property
    def wins(self) -> int:
        return self.summary.wins

    @property
    def losses(self) -> int:
        return self.summary.losses

    @property
    def win_rate(self) -> float:
        return self.summary.win_rate

    @property
    def label(self) -> str:
        return self.matchup_name or self.opponent_name

    def to_log_dict(self) -> dict[str, Any]:
        """Return numeric summary fields for existing scalar logging."""
        return self.summary.to_log_dict()

    def to_record(self) -> dict[str, Any]:
        """Return identity plus summary fields for tables, JSON, or artifacts."""
        return {
            "model_name": self.model_name,
            "opponent_name": self.opponent_name,
            "model_side": self.model_side,
            "opponent_side": self.opponent_side,
            "matchup_name": self.matchup_name,
            "model_path": self.model_path,
            "opponent_spec": self.opponent_spec,
            "winning_score": self.winning_score,
            "iteration": self.iteration,
            "step": self.step,
            "seed": self.seed,
            **self.summary.to_log_dict(),
        }

    def format_score_frame_line(
        self,
        label: str | None = None,
        *,
        indent: str = "    ",
        include_vs: bool = True,
    ) -> str:
        """Format wins, scores, and game length for console eval output."""
        return self.summary.format_score_frame_line(label or self.label, indent=indent, include_vs=include_vs)


@dataclass
class EvalBatch:
    """A set of eval results produced at the same training/evaluation point."""

    results: list[EvalResult]
    iteration: int | None = None
    step: int | None = None

    def __post_init__(self) -> None:
        for result in self.results:
            if result.iteration is None:
                result.iteration = self.iteration
            if result.step is None:
                result.step = self.step

    def by_opponent(self) -> dict[str, EvalResult]:
        """Return results keyed by opponent name."""
        return {result.opponent_name: result for result in self.results}

    def by_matchup(self) -> dict[str, EvalResult]:
        """Return results keyed by matchup name when available, otherwise opponent name."""
        return {result.label: result for result in self.results}

    def to_records(self) -> list[dict[str, Any]]:
        """Return table/JSON-friendly records."""
        return [result.to_record() for result in self.results]

    def format_score_frame_lines(self, *, indent: str = "    ", include_vs: bool = True) -> list[str]:
        """Format all results for console output."""
        return [result.format_score_frame_line(indent=indent, include_vs=include_vs) for result in self.results]


def build_eval_log_data(
    results: EvalBatch,
    prefix: str,
) -> dict[str, float]:
    """Build a wandb log_data dict from eval results.

    Args:
        results: Batch of eval results.
        prefix: Key prefix (e.g. "eval", "p1/eval", "curriculum").

    Returns:
        Flat dict like {"eval/vs_builtin/win_rate": 0.8, ...}
    """
    log_data: dict[str, float] = {}
    for opp_name, result in results.by_opponent().items():
        data = result.to_log_dict()
        for k in EVAL_METRIC_KEYS:
            if k in data:
                log_data[f"{prefix}/vs_{opp_name}/{k}"] = data[k]
    return log_data


def combine_per_side_summaries(p1_summary: EvalSummary, p2_summary: EvalSummary) -> EvalSummary:
    """Combine player_1 and player_2 perspective summaries."""
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
            continue
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


def combine_per_side_results(p1_result: EvalResult, p2_result: EvalResult) -> EvalResult:
    """Combine per-side eval results from a universal model into an aggregate.

    Win counts come from re-interpreting each side's game_winners (model wins
    when winner == its side). Numeric metrics are averaged across the two sides.
    """
    model_name = p1_result.model_name if p1_result.model_name == p2_result.model_name else "combined"
    opponent_name = p1_result.opponent_name if p1_result.opponent_name == p2_result.opponent_name else "combined"

    return EvalResult(
        model_name=model_name,
        opponent_name=opponent_name,
        model_side="both",
        opponent_side="both",
        summary=combine_per_side_summaries(p1_result.summary, p2_result.summary),
        matchup_name=p1_result.matchup_name or p2_result.matchup_name,
        model_path=p1_result.model_path if p1_result.model_path == p2_result.model_path else None,
        opponent_spec=p1_result.opponent_spec if p1_result.opponent_spec == p2_result.opponent_spec else None,
        winning_score=p1_result.winning_score if p1_result.winning_score == p2_result.winning_score else None,
        iteration=p1_result.iteration if p1_result.iteration == p2_result.iteration else None,
        step=p1_result.step if p1_result.step == p2_result.step else None,
        seed=None,
    )


def model_won_per_game(result: EvalResult | EvalSummary, model_side: str) -> list[bool]:
    """Convert an eval result's game_winners into a list of model-victory bools."""
    summary = result.summary if isinstance(result, EvalResult) else result
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
