"""Tests for evaluation result records and batch helpers."""

from training_center.scripts.utils import (
    EvalBatch,
    EvalResult,
    EvalSummary,
    combine_per_side_results,
    combine_per_side_summaries,
    model_won_per_game,
)


def _result(
    side: str,
    *,
    wins: int,
    losses: int,
    winners: list[str],
    p1_scores: list[int] | None = None,
    p2_scores: list[int] | None = None,
    game_frames: list[int] | None = None,
    metrics: dict[str, float] | None = None,
) -> EvalResult:
    return EvalResult(
        model_name="model",
        opponent_name="builtin",
        model_side=side,
        opponent_side="player_2" if side == "player_1" else "player_1",
        summary=EvalSummary(
            wins=wins,
            losses=losses,
            game_winners=winners,
            p1_scores=p1_scores or [],
            p2_scores=p2_scores or [],
            game_frames=game_frames or [],
            metrics=metrics or {},
        ),
    )


def test_combine_per_side_results_basic_counts():
    p1 = _result("player_1", wins=8, losses=2, winners=["player_1"] * 8 + ["player_2"] * 2)
    p2 = _result("player_2", wins=4, losses=6, winners=["player_2"] * 4 + ["player_1"] * 6)
    combined = combine_per_side_results(p1, p2)
    assert combined.wins == 12
    assert combined.losses == 8
    assert abs(combined.win_rate - 0.6) < 1e-9
    assert combined.model_side == "both"
    assert len(combined.summary.game_winners) == 20


def test_combine_per_side_summaries_averages_metric_keys():
    p1 = EvalSummary(
        wins=5,
        losses=5,
        game_winners=["player_1"] * 5 + ["player_2"] * 5,
        metrics={"action_entropy": 2.0, "power_hit_rate": 0.8},
    )
    p2 = EvalSummary(
        wins=5,
        losses=5,
        game_winners=["player_2"] * 5 + ["player_1"] * 5,
        metrics={"action_entropy": 3.0, "power_hit_rate": 0.6},
    )
    combined = combine_per_side_summaries(p1, p2)
    assert abs(combined.metric("action_entropy") - 2.5) < 1e-9
    assert abs(combined.metric("power_hit_rate") - 0.7) < 1e-9


def test_combine_per_side_results_combines_score_and_frame_samples():
    p1 = _result(
        "player_1",
        wins=1,
        losses=1,
        winners=["player_1", "player_2"],
        p1_scores=[5, 5],
        p2_scores=[1, 3],
        game_frames=[100, 200],
    )
    p2 = _result(
        "player_2",
        wins=1,
        losses=1,
        winners=["player_2", "player_1"],
        p1_scores=[2, 4],
        p2_scores=[5, 5],
        game_frames=[300, 400],
    )
    combined = combine_per_side_results(p1, p2)
    assert combined.summary.p1_scores == [5, 5, 2, 4]
    assert combined.summary.p2_scores == [1, 3, 5, 5]
    assert abs(combined.summary.avg_p1_score - 4.0) < 1e-9
    assert abs(combined.summary.var_p1_score - 1.5) < 1e-9
    assert abs(combined.summary.std_p1_score - 1.224744871391589) < 1e-9
    assert abs(combined.summary.avg_p2_score - 3.5) < 1e-9
    assert abs(combined.summary.var_p2_score - 2.75) < 1e-9
    assert abs(combined.summary.std_p2_score - 1.6583123951777) < 1e-9
    assert abs(combined.summary.avg_game_frames - 250.0) < 1e-9
    assert abs(combined.summary.var_game_frames - 12500.0) < 1e-9
    assert abs(combined.summary.std_game_frames - 111.80339887498948) < 1e-9


def test_eval_summary_formats_score_frame_line():
    summary = EvalSummary(
        wins=3,
        losses=1,
        game_winners=["player_1", "player_1", "player_2", "player_1"],
        p1_scores=[5, 5, 3, 5],
        p2_scores=[1, 2, 5, 0],
        game_frames=[100, 120, 140, 160],
    )
    assert summary.format_score_frame_line("random") == (
        "    vs random: 3W 1L (4.5 ± 0.9 vs 2.0 ± 1.9, frames: 130 ± 22)"
    )


def test_eval_result_carries_identity_and_formats_without_external_label():
    summary = EvalSummary(
        wins=2,
        losses=0,
        game_winners=["player_1", "player_1"],
        p1_scores=[5, 5],
        p2_scores=[1, 2],
        game_frames=[100, 120],
    )
    result = EvalResult(
        model_name="checkpoint_000010",
        opponent_name="builtin",
        model_side="player_1",
        opponent_side="player_2",
        summary=summary,
        model_path="experiments/run/checkpoint_000010/model.zip",
        opponent_spec="builtin",
        winning_score=5,
        step=10_000,
    )
    assert result.opponent_name == "builtin"
    assert result.wins == 2
    assert result.to_record()["step"] == 10_000
    assert result.to_record()["winning_score"] == 5
    assert result.to_record()["opponent_spec"] == "builtin"
    assert result.format_score_frame_line() == "    vs builtin: 2W 0L (5.0 ± 0.0 vs 1.5 ± 0.5, frames: 110 ± 10)"


def test_eval_batch_indexes_results():
    first = EvalResult(
        model_name="model",
        opponent_name="random",
        model_side="player_1",
        opponent_side="player_2",
        summary=EvalSummary(wins=1, losses=0, game_winners=["player_1"]),
    )
    second = EvalResult(
        model_name="model",
        opponent_name="builtin",
        model_side="player_1",
        opponent_side="player_2",
        summary=EvalSummary(wins=0, losses=1, game_winners=["player_2"]),
    )
    batch = EvalBatch([first, second], iteration=3, step=100)
    assert batch.by_opponent()["random"] is first
    assert first.iteration == 3
    assert second.step == 100
    assert [record["opponent_name"] for record in batch.to_records()] == ["random", "builtin"]


def test_model_won_per_game():
    summary = EvalSummary(
        wins=2,
        losses=2,
        game_winners=["player_1", "player_2", "player_1", "player_2"],
    )
    assert model_won_per_game(summary, "player_1") == [True, False, True, False]
    assert model_won_per_game(summary, "player_2") == [False, True, False, True]
