"""Shared utilities for training and evaluation scripts."""

from __future__ import annotations

import os
import re
import signal
import sys
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from io import StringIO
from math import sqrt
from numbers import Real
from typing import Any

import plotly.graph_objects as go
import wandb
from pika_zoo.env.pikachu_volleyball import NoiseConfig
from pika_zoo.records.types import GameRecord, GamesRecord
from plotly.subplots import make_subplots

from training_center.metrics import compute_eval_metrics

NOISE_LEVELS: dict[int, tuple[int, int, int]] = {
    0: (0, 0, 0),
    1: (5, 3, 1),
    2: (10, 5, 2),
    3: (20, 10, 3),
    4: (35, 15, 4),
    5: (50, 20, 5),
}

SERVE_RULES = ("winner", "loser", "alternate", "random")

TRAIN_DASHBOARD_METRICS: dict[str, str] = {
    "train/loss": "Loss",
    "train/entropy_loss": "Entropy loss",
    "train/explained_variance": "Explained variance",
    "train/approx_kl": "Approx KL",
}

CURRICULUM_TRAIN_DASHBOARD_METRICS: dict[str, dict[str, str]] = {
    "pool_size": {
        "title": "Curriculum pool size",
        "label": "unlocked pool",
        "color": "#4c78a8",
    },
    "selfplay_pool_size": {
        "title": "Self-play pool size",
        "label": "self-play pool",
        "color": "#b279a2",
    },
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


def _transparent_plotly_layout(fig: go.Figure) -> None:
    fig.update_layout(
        template="plotly_white",
        autosize=True,
        paper_bgcolor="rgba(255, 255, 255, 0)",
        plot_bgcolor="rgba(255, 255, 255, 0)",
    )
    grid_style = {"showgrid": True, "gridcolor": "rgba(31, 45, 61, 0.14)", "zerolinecolor": "rgba(31, 45, 61, 0.18)"}
    fig.update_xaxes(**grid_style)
    fig.update_yaxes(**grid_style)


def extend_train_chart_history(
    history: list[dict[str, Any]],
    metrics: dict[str, Any],
    *,
    step: int,
) -> list[dict[str, Any]]:
    """Append selected SB3 metrics to a compact train dashboard history."""
    for key in TRAIN_DASHBOARD_METRICS:
        value = metrics.get(key)
        if isinstance(value, Real):
            history.append({"step": step, "metric": key, "value": float(value)})
    return history


def build_train_chart_log_data(
    history: list[dict[str, Any]],
    *,
    curriculum_history: list[dict[str, Any]] | None = None,
    prefix: str = "train",
) -> dict[str, Any]:
    """Build one compact dashboard for training diagnostics.

    Curriculum-specific pool traces are appended as extra subplots when
    curriculum_history is provided, so curriculum runs still live under the
    train section instead of creating a separate W&B section.
    """
    curriculum_metrics = _curriculum_train_subplot_metrics(curriculum_history)
    include_curriculum = bool(curriculum_metrics)
    row_count = len(TRAIN_DASHBOARD_METRICS) + len(curriculum_metrics)
    subplot_titles = list(TRAIN_DASHBOARD_METRICS.values())
    subplot_titles.extend(CURRICULUM_TRAIN_DASHBOARD_METRICS[metric]["title"] for metric in curriculum_metrics)
    fig = make_subplots(
        rows=row_count,
        cols=1,
        shared_xaxes=True,
        subplot_titles=subplot_titles,
        vertical_spacing=0.06 if include_curriculum else 0.08,
    )
    for row, (metric, label) in enumerate(TRAIN_DASHBOARD_METRICS.items(), start=1):
        metric_rows = [item for item in history if item["metric"] == metric]
        if not metric_rows:
            continue
        fig.add_trace(
            go.Scatter(
                x=[item["step"] for item in metric_rows],
                y=[item["value"] for item in metric_rows],
                mode="lines+markers",
                name=label,
                showlegend=False,
            ),
            row=row,
            col=1,
        )
        fig.update_yaxes(title_text=label, row=row, col=1)
    if include_curriculum:
        _add_curriculum_train_subplots(
            fig,
            curriculum_history or [],
            len(TRAIN_DASHBOARD_METRICS) + 1,
            curriculum_metrics,
        )
    fig.update_layout(title="Training diagnostics", hovermode="x unified")
    fig.update_xaxes(title_text="Step", row=row_count, col=1)
    _transparent_plotly_layout(fig)
    return {f"{prefix}/dashboard": fig}


def _curriculum_train_subplot_metrics(history: list[dict[str, Any]] | None) -> list[str]:
    if history is None:
        return []
    return [
        metric
        for metric in CURRICULUM_TRAIN_DASHBOARD_METRICS
        if any(item["metric"] == metric for item in history)
    ]


def extend_curriculum_chart_history(
    history: list[dict[str, Any]],
    status: dict[str, Any],
    *,
    iteration: int,
    step: int,
    selfplay_pool_size: int | None = None,
) -> list[dict[str, Any]]:
    """Append curriculum pool status to a compact dashboard history."""
    history.append({"step": step, "iteration": iteration, "metric": "pool_size", "value": float(status["pool_size"])})
    if selfplay_pool_size is not None:
        history.append(
            {
                "step": step,
                "iteration": iteration,
                "metric": "selfplay_pool_size",
                "value": float(selfplay_pool_size),
            }
        )
    return history


def _add_curriculum_train_subplots(
    fig: go.Figure,
    history: list[dict[str, Any]],
    start_row: int,
    metrics: list[str],
) -> None:
    """Add curriculum pool traces to the shared train dashboard."""
    for offset, metric in enumerate(metrics):
        rows = [item for item in history if item["metric"] == metric]
        if rows:
            row = start_row + offset
            config = CURRICULUM_TRAIN_DASHBOARD_METRICS[metric]
            fig.add_trace(
                go.Scatter(
                    x=[item["step"] for item in rows],
                    y=[item["value"] for item in rows],
                    mode="lines+markers",
                    name=config["label"],
                    line={"color": config["color"]},
                    showlegend=False,
                ),
                row=row,
                col=1,
            )
            fig.update_yaxes(title_text="Count", row=row, col=1)


def build_video_log_data(samples: list[dict[str, Any]], *, prefix: str = "video") -> dict[str, Any]:
    """Build one W&B media table for sample videos."""
    table = wandb.Table(columns=["opponent", "model_side", "serve", "winner", "score", "frames", "video"])
    for sample in samples:
        table.add_data(
            sample["opponent"],
            sample["model_side"],
            sample["serve"],
            sample["winner"],
            sample["score"],
            sample["frames"],
            sample["video"],
        )
    return {f"{prefix}/samples": table}


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
    "std_score",
    "var_score",
    "avg_opp_score",
    "std_opp_score",
    "var_opp_score",
    "avg_round_frames",
    "avg_p1_score",
    "std_p1_score",
    "var_p1_score",
    "avg_p2_score",
    "std_p2_score",
    "var_p2_score",
    "avg_game_frames",
    "std_game_frames",
    "var_game_frames",
    "serve_win_rate",
    "receive_win_rate",
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


def _mean_std(values: list[int | float]) -> tuple[float, float]:
    """Return population mean and standard deviation for a metric sample."""
    mean, variance = _mean_var(values)
    return mean, sqrt(variance)


def _normal_ci95(mean: float, std: float, n: int) -> tuple[float, float]:
    """Return a normal-approximation 95% confidence interval for a mean."""
    if n <= 0:
        return mean, mean
    half_width = 1.96 * std / sqrt(n)
    return mean - half_width, mean + half_width


def _binomial_ci95(successes: int, n: int) -> tuple[float, float]:
    """Return a Wilson 95% confidence interval for a proportion."""
    if n <= 0:
        return 0.0, 0.0
    p = successes / n
    z = 1.96
    denominator = 1.0 + z**2 / n
    center = (p + z**2 / (2.0 * n)) / denominator
    half_width = z * sqrt((p * (1.0 - p) + z**2 / (4.0 * n)) / n) / denominator
    return max(0.0, center - half_width), min(1.0, center + half_width)


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
        avg_score, std_score = _mean_std(model_scores)
        avg_opp_score, std_opp_score = _mean_std(opp_scores)

        metrics = {
            "avg_score": avg_score,
            "std_score": std_score,
            "var_score": std_score**2,
            "avg_opp_score": avg_opp_score,
            "std_opp_score": std_opp_score,
            "var_opp_score": std_opp_score**2,
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
    def std_p1_score(self) -> float:
        return sqrt(self.var_p1_score)

    @property
    def avg_p2_score(self) -> float:
        return _mean_var(self.p2_scores)[0]

    @property
    def var_p2_score(self) -> float:
        return _mean_var(self.p2_scores)[1]

    @property
    def std_p2_score(self) -> float:
        return sqrt(self.var_p2_score)

    @property
    def avg_game_frames(self) -> float:
        return _mean_var(self.game_frames)[0]

    @property
    def var_game_frames(self) -> float:
        return _mean_var(self.game_frames)[1]

    @property
    def std_game_frames(self) -> float:
        return sqrt(self.var_game_frames)

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
            "std_p1_score": self.std_p1_score,
            "var_p1_score": self.var_p1_score,
            "avg_p2_score": self.avg_p2_score,
            "std_p2_score": self.std_p2_score,
            "var_p2_score": self.var_p2_score,
            "avg_game_frames": self.avg_game_frames,
            "std_game_frames": self.std_game_frames,
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
        avg_score = self.metric("avg_score", self.avg_p1_score)
        std_score = self.metric("std_score", self.std_p1_score)
        avg_opp_score = self.metric("avg_opp_score", self.avg_p2_score)
        std_opp_score = self.metric("std_opp_score", self.std_p2_score)
        return (
            f"{indent}{label_text}: {self.wins}W {self.losses}L "
            f"({avg_score:.1f}±{std_score:.1f} "
            f"vs {avg_opp_score:.1f}±{std_opp_score:.1f}, "
            f"frames: {self.avg_game_frames:.0f}±{self.std_game_frames:.0f})"
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


def _eval_side_label(label: str, result: EvalResult) -> str:
    """Return a compact side label for chart legends."""
    if label in {"p1", "p2"}:
        return label
    if result.model_side == "player_1":
        return "p1"
    if result.model_side == "player_2":
        return "p2"
    return "combined" if label == "combined" else label


def _eval_side_labels(label: str, result: EvalResult, *, has_combined_batch: bool) -> list[str]:
    """Return chart side labels for a result.

    The dashboard always has a representative "combined" series. If callers
    provide an explicit combined batch, use it as the only combined source. If
    they only provide side-specific batches, each side also gets a combined
    alias so single-side dashboards remain structurally consistent.
    """
    side = _eval_side_label(label, result)
    if label == "combined":
        return ["combined", side] if side != "combined" else ["combined"]
    if has_combined_batch:
        return [side]
    if side in {"p1", "p2"}:
        return ["combined", side]
    return [side]


def build_eval_chart_table(batches: dict[str, EvalBatch]) -> wandb.Table:
    """Build one long-form eval table used by all W&B eval charts."""
    columns = [
        "step",
        "iteration",
        "opponent",
        "eval_side",
        "metric",
        "value",
        "std",
        "ci95_low",
        "ci95_high",
        "n",
        "wins",
        "losses",
    ]
    rows: list[list[Any]] = []
    has_combined_batch = "combined" in batches
    for label, batch in batches.items():
        for result in batch.results:
            data = result.to_log_dict()
            side_labels = _eval_side_labels(label, result, has_combined_batch=has_combined_batch)

            ci_low, ci_high = _binomial_ci95(result.wins, result.games)
            for side_label in side_labels:
                rows.append(
                    [
                        result.step,
                        result.iteration,
                        result.opponent_name,
                        side_label,
                        "win_rate",
                        result.win_rate,
                        None,
                        ci_low,
                        ci_high,
                        result.games,
                        result.wins,
                        result.losses,
                    ]
                )

                for metric, mean_key, std_key in [
                    ("model_score", "avg_score", "std_score"),
                    ("opponent_score", "avg_opp_score", "std_opp_score"),
                    ("round_frames", "avg_round_frames", "std_round_frames"),
                    ("game_frames", "avg_game_frames", "std_game_frames"),
                ]:
                    value = float(data.get(mean_key, 0.0))
                    std = float(data.get(std_key, 0.0))
                    metric_ci_low, metric_ci_high = _normal_ci95(value, std, result.games)
                    rows.append(
                        [
                            result.step,
                            result.iteration,
                            result.opponent_name,
                            side_label,
                            metric,
                            value,
                            std,
                            metric_ci_low,
                            metric_ci_high,
                            result.games,
                            None,
                            None,
                        ]
                    )
    return wandb.Table(data=rows, columns=columns)


def extend_eval_chart_history(
    history: dict[str, list[EvalResult]],
    batches: dict[str, EvalBatch],
) -> dict[str, EvalBatch]:
    """Append current eval batches and return cumulative batches for chart logging."""
    for label, batch in batches.items():
        history.setdefault(label, []).extend(batch.results)
    return {label: EvalBatch(list(results)) for label, results in history.items()}


def _rows_to_dicts(rows: list[list[Any]], columns: list[str]) -> list[dict[str, Any]]:
    return [dict(zip(columns, row, strict=True)) for row in rows]


def _add_plotly_ci_series(
    fig: go.Figure,
    *,
    row: int,
    col: int,
    x: list[Any],
    y: list[float],
    y_low: list[float],
    y_high: list[float],
    side: str,
    color: str,
    fillcolor: str,
    visible: bool | str,
    showlegend: bool,
    trace_specs: list[tuple[str, str | None]],
    opponent: str,
    customdata: list[list[Any]],
    value_column: str,
) -> None:
    fig.add_trace(
        go.Scatter(
            x=x + x[::-1],
            y=y_high + y_low[::-1],
            mode="lines",
            line={"width": 0, "color": "rgba(255, 255, 255, 0)"},
            fill="toself",
            fillcolor=fillcolor,
            name=f"{side} 95% CI",
            showlegend=False,
            hoverinfo="skip",
            legendgroup=side,
            visible=visible,
        ),
        row=row,
        col=col,
    )
    trace_specs.append((opponent, side))
    for bound_name, bound_y in [("upper", y_high), ("lower", y_low)]:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=bound_y,
                mode="lines",
                line={"width": 1, "color": color, "dash": "dot"},
                opacity=0.45,
                name=f"{side} 95% CI {bound_name}",
                showlegend=False,
                hoverinfo="skip",
                legendgroup=side,
                visible=visible,
            ),
            row=row,
            col=col,
        )
        trace_specs.append((opponent, side))
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines+markers",
            line={"color": color},
            marker={"color": color},
            name=side,
            showlegend=showlegend,
            legendgroup=side,
            customdata=customdata,
            hovertemplate=(
                "step=%{x}<br>"
                f"{value_column}=%{{y:.3f}}<br>"
                "ci95=[%{customdata[4]:.3f}, %{customdata[5]:.3f}]<br>"
                "std=%{customdata[0]}<br>"
                "n=%{customdata[1]}<br>"
                "wins=%{customdata[2]} losses=%{customdata[3]}<extra></extra>"
            ),
            visible=visible,
        ),
        row=row,
        col=col,
    )
    trace_specs.append((opponent, side))


def _default_visible_sides(data: list[dict[str, Any]]) -> dict[str, set[str]]:
    sides_by_opponent: dict[str, set[str]] = {}
    for row in data:
        sides_by_opponent.setdefault(str(row["opponent"]), set()).add(str(row["eval_side"]))
    return {opponent: {"combined"} if "combined" in sides else sides for opponent, sides in sides_by_opponent.items()}


def _eval_trace_visibility(
    trace_opponent: str,
    side: str | None,
    selected_opponent: str,
    visible_sides_by_opponent: dict[str, set[str]],
) -> bool | str:
    if trace_opponent != selected_opponent:
        return False
    if side is None or side in visible_sides_by_opponent.get(selected_opponent, set()):
        return True
    return "legendonly"


def _plotly_eval_dashboard(
    rows: list[list[Any]],
    columns: list[str],
    *,
    opponent: str,
    unlock_threshold: float | None = None,
) -> go.Figure:
    """Build one opponent-specific dashboard with core eval curves together."""
    data = _rows_to_dicts(rows, columns)
    selected_opponent = opponent
    visible_sides_by_opponent = _default_visible_sides(data)

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        subplot_titles=["Win rate", "Model score", "Opponent score", "Round frames"],
        vertical_spacing=0.08,
    )
    color_by_side = {"combined": "#4c78a8", "p1": "#f58518", "p2": "#54a24b"}
    fill_by_side = {
        "combined": "rgba(76, 120, 168, 0.36)",
        "p1": "rgba(245, 133, 24, 0.28)",
        "p2": "rgba(84, 162, 75, 0.28)",
    }
    trace_specs: list[tuple[str, str | None]] = []

    for opponent in [selected_opponent]:
        for side in ["combined", "p1", "p2"]:
            visible = _eval_trace_visibility(opponent, side, selected_opponent, visible_sides_by_opponent)
            win_side_rows = [
                row
                for row in data
                if row["opponent"] == opponent and row["eval_side"] == side and row["metric"] == "win_rate"
            ]
            if win_side_rows:
                win_side_rows.sort(key=lambda row: row["step"])
                x = [row["step"] for row in win_side_rows]
                _add_plotly_ci_series(
                    fig,
                    row=1,
                    col=1,
                    x=x,
                    y=[row["value"] for row in win_side_rows],
                    y_low=[row["ci95_low"] for row in win_side_rows],
                    y_high=[row["ci95_high"] for row in win_side_rows],
                    side=side,
                    color=color_by_side.get(side, "#777"),
                    fillcolor=fill_by_side.get(side, "rgba(119, 119, 119, 0.18)"),
                    visible=visible,
                    showlegend=True,
                    trace_specs=trace_specs,
                    opponent=opponent,
                    customdata=[
                        [
                            row.get("std"),
                            row.get("n"),
                            row.get("wins"),
                            row.get("losses"),
                            row["ci95_low"],
                            row["ci95_high"],
                        ]
                        for row in win_side_rows
                    ],
                    value_column="win_rate",
                )

            for subplot_row, metric, value_column in [
                (2, "model_score", "score"),
                (3, "opponent_score", "opponent_score"),
            ]:
                score_side_rows = [
                    row
                    for row in data
                    if row["opponent"] == opponent and row["eval_side"] == side and row["metric"] == metric
                ]
                if score_side_rows:
                    score_side_rows.sort(key=lambda row: row["step"])
                    x = [row["step"] for row in score_side_rows]
                    _add_plotly_ci_series(
                        fig,
                        row=subplot_row,
                        col=1,
                        x=x,
                        y=[row["value"] for row in score_side_rows],
                        y_low=[row["ci95_low"] for row in score_side_rows],
                        y_high=[row["ci95_high"] for row in score_side_rows],
                        side=side,
                        color=color_by_side.get(side, "#777"),
                        fillcolor=fill_by_side.get(side, "rgba(119, 119, 119, 0.18)"),
                        visible=visible,
                        showlegend=False,
                        trace_specs=trace_specs,
                        opponent=opponent,
                        customdata=[
                            [
                                row.get("std"),
                                row.get("n"),
                                None,
                                None,
                                row["ci95_low"],
                                row["ci95_high"],
                            ]
                            for row in score_side_rows
                        ],
                        value_column=value_column,
                    )

            frame_side_rows = [
                row
                for row in data
                if row["opponent"] == opponent and row["eval_side"] == side and row["metric"] == "round_frames"
            ]
            if frame_side_rows:
                frame_side_rows.sort(key=lambda row: row["step"])
                x = [row["step"] for row in frame_side_rows]
                _add_plotly_ci_series(
                    fig,
                    row=4,
                    col=1,
                    x=x,
                    y=[row["value"] for row in frame_side_rows],
                    y_low=[row["ci95_low"] for row in frame_side_rows],
                    y_high=[row["ci95_high"] for row in frame_side_rows],
                    side=side,
                    color=color_by_side.get(side, "#777"),
                    fillcolor=fill_by_side.get(side, "rgba(119, 119, 119, 0.18)"),
                    visible=visible,
                    showlegend=False,
                    trace_specs=trace_specs,
                    opponent=opponent,
                    customdata=[
                        [
                            row.get("std"),
                            row.get("n"),
                            None,
                            None,
                            row["ci95_low"],
                            row["ci95_high"],
                        ]
                        for row in frame_side_rows
                    ],
                    value_column="round_frames",
                )

        steps = sorted({row["step"] for row in data if row["opponent"] == opponent and row["metric"] == "win_rate"})
        if steps and unlock_threshold is not None:
            fig.add_trace(
                go.Scatter(
                    x=[min(steps), max(steps)],
                    y=[unlock_threshold, unlock_threshold],
                    mode="lines",
                    line={"color": "#777", "dash": "dash"},
                    name=f"unlock threshold ({unlock_threshold:.2f})",
                    showlegend=True,
                    hoverinfo="skip",
                    visible=_eval_trace_visibility(opponent, None, selected_opponent, visible_sides_by_opponent),
                ),
                row=1,
                col=1,
            )
            trace_specs.append((opponent, None))

    fig.update_layout(
        template="plotly_white",
        autosize=True,
        title=f"Eval vs {selected_opponent}",
        hovermode="x unified",
        paper_bgcolor="rgba(255, 255, 255, 0)",
        plot_bgcolor="rgba(255, 255, 255, 0)",
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.16,
            "xanchor": "left",
            "x": 0,
            "bgcolor": "rgba(255, 255, 255, 0.75)",
        },
        margin={"l": 60, "r": 30, "t": 110, "b": 50},
    )
    grid_style = {"showgrid": True, "gridcolor": "rgba(31, 45, 61, 0.14)", "zerolinecolor": "rgba(31, 45, 61, 0.18)"}
    fig.update_yaxes(title_text="Win rate", range=[0, 1], row=1, col=1, **grid_style)
    fig.update_yaxes(title_text="Mean score", range=[0, 5], row=2, col=1, **grid_style)
    fig.update_yaxes(title_text="Mean score", range=[0, 5], row=3, col=1, **grid_style)
    fig.update_yaxes(title_text="Frames", row=4, col=1, **grid_style)
    fig.update_xaxes(**grid_style)
    fig.update_xaxes(title_text="Step", row=4, col=1)
    return fig


def _eval_dashboard_key(opponent: str) -> str:
    return opponent.replace("/", "_")


def _eval_table_opponents(rows: list[list[Any]], columns: list[str]) -> list[str]:
    opponent_index = columns.index("opponent")
    return sorted({str(row[opponent_index]) for row in rows})


def build_eval_chart_log_data(
    batches: dict[str, EvalBatch],
    *,
    prefix: str = "eval",
    unlock_threshold: float | None = None,
) -> dict[str, Any]:
    """Build W&B chart data from a single source of eval table rows.

    The Plotly dashboard is derived only from the tables returned here. Scalar
    summaries remain separate quick-look metrics and are not chart inputs.
    """
    table = build_eval_chart_table(batches)
    log_data: dict[str, Any] = {
        f"{prefix}/table": table,
    }
    for opponent in _eval_table_opponents(table.data, table.columns):
        log_data[f"{prefix}/dashboard/{_eval_dashboard_key(opponent)}"] = _plotly_eval_dashboard(
            table.data,
            table.columns,
            opponent=opponent,
            unlock_threshold=unlock_threshold,
        )
    return log_data


def combine_per_side_summaries(p1_summary: EvalSummary, p2_summary: EvalSummary) -> EvalSummary:
    """Combine player_1 and player_2 perspective summaries."""
    p1_won = [w == "player_1" for w in p1_summary.game_winners]
    p2_won = [w == "player_2" for w in p2_summary.game_winners]
    won_list = p1_won + p2_won
    total = len(won_list)
    wins = sum(won_list)

    metrics: dict[str, float] = {}
    model_scores = p1_summary.p1_scores + p2_summary.p2_scores
    opp_scores = p1_summary.p2_scores + p2_summary.p1_scores
    avg_score, std_score = _mean_std(model_scores)
    avg_opp_score, std_opp_score = _mean_std(opp_scores)
    metrics.update(
        {
            "avg_score": avg_score,
            "std_score": std_score,
            "var_score": std_score**2,
            "avg_opp_score": avg_opp_score,
            "std_opp_score": std_opp_score,
            "var_opp_score": std_opp_score**2,
        }
    )
    p1_data = p1_summary.to_log_dict()
    p2_data = p2_summary.to_log_dict()
    for k in EVAL_METRIC_KEYS:
        if k in {
            "win_rate",
            "avg_score",
            "std_score",
            "var_score",
            "avg_opp_score",
            "std_opp_score",
            "var_opp_score",
            "avg_p1_score",
            "std_p1_score",
            "var_p1_score",
            "avg_p2_score",
            "std_p2_score",
            "var_p2_score",
            "avg_game_frames",
            "std_game_frames",
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


def record_video(
    model_path: str,
    side: str,
    opponent: str,
    output_path: str,
    *,
    serve: str = "winner",
) -> dict[str, Any]:
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
    stdout = StringIO()
    with redirect_stdout(stdout):
        play(p1=p1, p2=p2, winning_score=5, serve=serve, render=False, record=output_path, seed=0)
    return _parse_video_result(stdout.getvalue(), model_side=side)


def _parse_video_result(output: str, *, model_side: str) -> dict[str, Any]:
    match = re.search(r"Game over! Player ([12]) wins (\d+)-(\d+) \((\d+) frames\)", output)
    if match is None:
        return {"winner": None, "score": None, "frames": None}
    winner_side = "player_1" if match.group(1) == "1" else "player_2"
    return {
        "winner": "model" if winner_side == model_side else "opponent",
        "score": f"{match.group(2)}-{match.group(3)}",
        "frames": int(match.group(4)),
    }
