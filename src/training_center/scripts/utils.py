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


def record_video(model_path: str, side: str, opponent: str, output_path: str) -> None:
    """Record a sample game video using pika-zoo's play script."""
    from pika_zoo.scripts.play import play

    p1 = model_path if side == "player_1" else opponent
    p2 = opponent if side == "player_1" else model_path
    play(p1=p1, p2=p2, winning_score=5, render=False, record=output_path, seed=0)
