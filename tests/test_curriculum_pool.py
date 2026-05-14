"""Tests for CurriculumPool, including self-play entry handling."""

from training_center.pool.curriculum import SELF_ENTRY, CurriculumPool
from training_center.scripts.utils import (
    EvalBatch,
    EvalResult,
    EvalSummary,
    combine_per_side_results,
    model_won_per_game,
)


def test_initial_state():
    pool = CurriculumPool(["a", "b", "c"], unlock_threshold=0.75)
    assert pool.unlocked == []
    assert pool.sample_opponent() == "a"  # falls back to first ladder entry


def test_force_unlock_and_sample():
    pool = CurriculumPool(["a", "b"], unlock_threshold=0.75)
    pool.force_unlock(0)
    assert pool.unlocked == ["a"]
    assert pool.sample_opponent() == "a"


def test_try_unlock_blocks_until_threshold_met():
    pool = CurriculumPool(["a", "b"], unlock_threshold=0.75)
    pool.force_unlock(0)
    pool.set_win_rate("a", 0.6)  # below threshold
    assert pool.try_unlock() is None

    pool.set_win_rate("a", 0.8)  # above threshold
    assert pool.try_unlock() == "b"


def test_try_unlock_skips_self_entry():
    """`self` always sits at ~50% win rate; including it would block all unlocks."""
    pool = CurriculumPool(["a", SELF_ENTRY, "b"], unlock_threshold=0.75)
    pool.force_unlock(0)
    pool.force_unlock(1)  # unlock self

    pool.set_win_rate("a", 0.8)  # master "a"
    pool.set_win_rate(SELF_ENTRY, 0.5)  # self sits at ~50%

    # Should still unlock "b" despite self being at 50%.
    assert pool.try_unlock() == "b"


def test_try_unlock_blocks_when_opponent_not_yet_evaluated():
    pool = CurriculumPool(["a", "b"], unlock_threshold=0.75)
    pool.force_unlock(0)
    # No win rate set yet.
    assert pool.try_unlock() is None


def test_status_excludes_self_from_aggregates():
    pool = CurriculumPool(["a", SELF_ENTRY], unlock_threshold=0.75)
    pool.force_unlock(0)
    pool.force_unlock(1)

    pool.set_win_rate("a", 0.9)
    pool.set_win_rate(SELF_ENTRY, 0.5)

    status = pool.status()
    assert status["pool_size"] == 2  # both entries are unlocked
    assert SELF_ENTRY not in status["per_opponent"]
    assert abs(status["min_win_rate"] - 0.9) < 1e-9
    assert abs(status["avg_win_rate"] - 0.9) < 1e-9


def test_self_can_be_sampled():
    pool = CurriculumPool([SELF_ENTRY], unlock_threshold=0.75)
    pool.force_unlock(0)
    assert pool.sample_opponent() == SELF_ENTRY


def test_combine_per_side_results_basic_counts():
    p1 = {
        "wins": 8,
        "losses": 2,
        "win_rate": 0.8,
        "avg_score": 4.0,
        "game_winners": ["player_1"] * 8 + ["player_2"] * 2,
    }
    p2 = {
        "wins": 4,
        "losses": 6,
        "win_rate": 0.4,
        "avg_score": 3.0,
        "game_winners": ["player_2"] * 4 + ["player_1"] * 6,
    }
    combined = combine_per_side_results(p1, p2)
    assert combined["wins"] == 12  # 8 (P1) + 4 (P2 won when winner == "player_2")
    assert combined["losses"] == 8
    assert abs(combined["win_rate"] - 0.6) < 1e-9
    assert abs(combined["avg_score"] - 3.5) < 1e-9
    # Concatenated for callers that still iterate; length only check.
    assert len(combined["game_winners"]) == 20


def test_combine_per_side_results_averages_metric_keys():
    p1 = {
        "wins": 5,
        "losses": 5,
        "win_rate": 0.5,
        "avg_score": 4.0,
        "game_winners": ["player_1"] * 5 + ["player_2"] * 5,
        "action_entropy": 2.0,
        "power_hit_rate": 0.8,
    }
    p2 = {
        "wins": 5,
        "losses": 5,
        "win_rate": 0.5,
        "avg_score": 2.0,
        "game_winners": ["player_2"] * 5 + ["player_1"] * 5,
        "action_entropy": 3.0,
        "power_hit_rate": 0.6,
    }
    combined = combine_per_side_results(p1, p2)
    assert abs(combined["action_entropy"] - 2.5) < 1e-9
    assert abs(combined["power_hit_rate"] - 0.7) < 1e-9


def test_combine_per_side_results_combines_score_and_frame_samples():
    p1 = {
        "wins": 1,
        "losses": 1,
        "win_rate": 0.5,
        "game_winners": ["player_1", "player_2"],
        "p1_scores": [5, 5],
        "p2_scores": [1, 3],
        "game_frames": [100, 200],
    }
    p2 = {
        "wins": 1,
        "losses": 1,
        "win_rate": 0.5,
        "game_winners": ["player_2", "player_1"],
        "p1_scores": [2, 4],
        "p2_scores": [5, 5],
        "game_frames": [300, 400],
    }
    combined = combine_per_side_results(p1, p2)
    assert combined["p1_scores"] == [5, 5, 2, 4]
    assert combined["p2_scores"] == [1, 3, 5, 5]
    assert abs(combined["avg_p1_score"] - 4.0) < 1e-9
    assert abs(combined["var_p1_score"] - 1.5) < 1e-9
    assert abs(combined["avg_p2_score"] - 3.5) < 1e-9
    assert abs(combined["var_p2_score"] - 2.75) < 1e-9
    assert abs(combined["avg_game_frames"] - 250.0) < 1e-9
    assert abs(combined["var_game_frames"] - 12500.0) < 1e-9


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
        "    vs random: 3W 1L (4.5 ± 0.8 vs 2.0 ± 3.5, frames: 130 ± 500)"
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
        step=10_000,
    )
    assert result["opponent_name"] == "builtin"
    assert result["wins"] == 2
    assert result.to_record()["step"] == 10_000
    assert result.format_score_frame_line() == "    vs builtin: 2W 0L (5.0 ± 0.0 vs 1.5 ± 0.2, frames: 110 ± 100)"


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
    result = {"game_winners": ["player_1", "player_2", "player_1", "player_2"]}
    assert model_won_per_game(result, "player_1") == [True, False, True, False]
    assert model_won_per_game(result, "player_2") == [False, True, False, True]
