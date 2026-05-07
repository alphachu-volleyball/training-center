"""Tests for CurriculumPool, including self-play entry handling."""

from training_center.pool.curriculum import SELF_ENTRY, CurriculumPool


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
