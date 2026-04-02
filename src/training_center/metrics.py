"""Eval metrics computed from GameRecord frame data.

Requires games recorded with ``record_frames=True``.
"""

from __future__ import annotations

import math
from collections import Counter

from pika_zoo.records.types import GamesRecord

NET_X = 216


def compute_eval_metrics(record: GamesRecord, model_side: str) -> dict[str, float]:
    """Compute detailed eval metrics over multiple games.

    Args:
        record: GamesRecord with frame data (recorded with ``record_frames=True``).
        model_side: "player_1" or "player_2".

    Returns:
        Dict of metric name → value.
    """
    is_p1 = model_side == "player_1"
    action_key = "player1_action" if is_p1 else "player2_action"
    touch_key = "p1_touch_ball" if is_p1 else "p2_touch_ball"
    power_key = "p1_power_hit" if is_p1 else "p2_power_hit"

    events = record.event_counts

    # Frame-level iteration for metrics GamesRecord doesn't aggregate
    action_counts: Counter[int] = Counter()
    ball_own_side_frames = 0
    recorded_frames = 0
    for game in record.games:
        for frame in game.frames:
            recorded_frames += 1
            action_counts[getattr(frame, action_key)] += 1
            if is_p1:
                if frame.ball_x < NET_X:
                    ball_own_side_frames += 1
            else:
                if frame.ball_x >= NET_X:
                    ball_own_side_frames += 1

    # Round-level: serve/receive split
    round_durations: list[int] = []
    serve_round_durations: list[int] = []
    receive_round_durations: list[int] = []
    for game in record.games:
        for rnd in game.rounds:
            round_durations.append(rnd.num_frames)
            if rnd.server == model_side:
                serve_round_durations.append(rnd.num_frames)
            else:
                receive_round_durations.append(rnd.num_frames)

    metrics: dict[str, float] = {}

    # Action entropy
    if recorded_frames > 0:
        probs = [count / recorded_frames for count in action_counts.values()]
        metrics["action_entropy"] = -sum(p * math.log2(p) for p in probs if p > 0)
    else:
        metrics["action_entropy"] = 0.0

    # Power hit rate (from pre-aggregated event_counts)
    metrics["power_hit_rate"] = events[power_key] / max(events[touch_key], 1)

    # Ball own side ratio
    metrics["ball_own_side_ratio"] = ball_own_side_frames / max(recorded_frames, 1)

    # Round frame stats
    if round_durations:
        mean = sum(round_durations) / len(round_durations)
        variance = sum((d - mean) ** 2 for d in round_durations) / len(round_durations)
        metrics["avg_round_frames"] = mean
        metrics["std_round_frames"] = math.sqrt(variance)
    else:
        metrics["avg_round_frames"] = 0.0
        metrics["std_round_frames"] = 0.0

    if serve_round_durations:
        metrics["serve_avg_round_frames"] = sum(serve_round_durations) / len(serve_round_durations)
    else:
        metrics["serve_avg_round_frames"] = 0.0

    if receive_round_durations:
        metrics["receive_avg_round_frames"] = sum(receive_round_durations) / len(receive_round_durations)
    else:
        metrics["receive_avg_round_frames"] = 0.0

    # Serve/receive win rates
    all_rounds = [r for g in record.games for r in g.rounds]
    model_serve = [r for r in all_rounds if r.server == model_side]
    opp_serve = [r for r in all_rounds if r.server != model_side]
    metrics["serve_win_rate"] = (
        sum(1 for r in model_serve if r.scorer == model_side) / len(model_serve) if model_serve else 0.0
    )
    metrics["receive_win_rate"] = (
        sum(1 for r in opp_serve if r.scorer == model_side) / len(opp_serve) if opp_serve else 0.0
    )

    return metrics


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
