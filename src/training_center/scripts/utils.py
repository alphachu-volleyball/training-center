"""Shared utilities for training and evaluation scripts."""

from __future__ import annotations

import os
import signal
import sys
from concurrent.futures import ProcessPoolExecutor

from pika_zoo.env.pikachu_volleyball import NoiseConfig

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


def build_eval_log_data(
    results: dict[str, dict],
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
        for k in EVAL_METRIC_KEYS:
            if k in r:
                log_data[f"{prefix}/vs_{opp_name}/{k}"] = r[k]
    return log_data


def combine_per_side_results(p1_result: dict, p2_result: dict) -> dict:
    """Combine per-side eval results from a universal model into an aggregate.

    Win counts come from re-interpreting each side's game_winners (model wins
    when winner == its side). Numeric metrics are averaged across the two
    sides — exact when both ran the same number of games, which is how callers
    invoke this. The combined dict mirrors the schema produced by
    `_eval_matchup_worker` so it plugs into existing logging and pool-update
    paths unchanged.
    """
    p1_won = [w == "player_1" for w in p1_result["game_winners"]]
    p2_won = [w == "player_2" for w in p2_result["game_winners"]]
    won_list = p1_won + p2_won
    total = len(won_list)
    wins = sum(won_list)

    combined: dict = {
        "wins": wins,
        "losses": total - wins,
        "win_rate": wins / total if total else 0.0,
        # Concatenated for downstream callers that iterate game_winners; values
        # alternate between "player_1"/<other> and "player_2"/<other> without a
        # single shared "model_side", so prefer wins/losses/win_rate above.
        "game_winners": list(p1_result["game_winners"]) + list(p2_result["game_winners"]),
    }

    for sample_key, avg_key, var_key in [
        ("p1_scores", "avg_p1_score", "var_p1_score"),
        ("p2_scores", "avg_p2_score", "var_p2_score"),
        ("game_frames", "avg_game_frames", "var_game_frames"),
    ]:
        v1 = p1_result.get(sample_key)
        v2 = p2_result.get(sample_key)
        if isinstance(v1, list) and isinstance(v2, list):
            samples = v1 + v2
            combined[sample_key] = samples
            combined[avg_key], combined[var_key] = _mean_var(samples)

    for k in EVAL_METRIC_KEYS:
        if k == "win_rate" or k in combined:
            continue  # already set above from total wins
        v1 = p1_result.get(k)
        v2 = p2_result.get(k)
        if isinstance(v1, int | float) and isinstance(v2, int | float):
            combined[k] = (v1 + v2) / 2

    return combined


def model_won_per_game(result: dict, model_side: str) -> list[bool]:
    """Convert an eval result's game_winners into a list of model-victory bools."""
    return [w == model_side for w in result["game_winners"]]


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
