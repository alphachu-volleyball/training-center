"""Tests for CurriculumPool, including self-play entry handling."""

from training_center.pool.curriculum import SELF_ENTRY, CurriculumPool
from training_center.scripts.utils import combine_per_side_results, model_won_per_game


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
    # 60% win rate — below threshold
    for _ in range(6):
        pool.update_stats("a", True)
    for _ in range(4):
        pool.update_stats("a", False)
    assert pool.try_unlock() is None

    # 16W / 4L = 80% — should unlock
    for _ in range(10):
        pool.update_stats("a", True)
    assert pool.try_unlock() == "b"


def test_try_unlock_skips_self_entry():
    """`self` always sits at ~50% win rate; including it would block all unlocks."""
    pool = CurriculumPool(["a", SELF_ENTRY, "b"], unlock_threshold=0.75)
    pool.force_unlock(0)
    pool.force_unlock(1)  # unlock self

    # Master "a" at 80%, leave self at 50%.
    for _ in range(8):
        pool.update_stats("a", True)
    for _ in range(2):
        pool.update_stats("a", False)
    for _ in range(5):
        pool.update_stats(SELF_ENTRY, True)
    for _ in range(5):
        pool.update_stats(SELF_ENTRY, False)

    # Should still unlock "b" despite self being at 50%.
    assert pool.try_unlock() == "b"


def test_status_excludes_self_from_aggregates():
    pool = CurriculumPool(["a", SELF_ENTRY], unlock_threshold=0.75)
    pool.force_unlock(0)
    pool.force_unlock(1)

    for _ in range(9):
        pool.update_stats("a", True)
    pool.update_stats("a", False)
    for _ in range(5):
        pool.update_stats(SELF_ENTRY, True)
    for _ in range(5):
        pool.update_stats(SELF_ENTRY, False)

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


def test_model_won_per_game():
    result = {"game_winners": ["player_1", "player_2", "player_1", "player_2"]}
    assert model_won_per_game(result, "player_1") == [True, False, True, False]
    assert model_won_per_game(result, "player_2") == [False, True, False, True]
