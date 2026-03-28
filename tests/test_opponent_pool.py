"""Tests for PFSP opponent pool."""

from training_center.eval.opponent_pool import OpponentPool


def test_empty_pool_win_rate():
    pool = OpponentPool("/tmp/test_pool", "p1")
    assert pool.get_win_rate("nonexistent") == 0.5


def test_update_stats():
    pool = OpponentPool("/tmp/test_pool", "p1")
    pool.update_stats("opp_a", True)
    pool.update_stats("opp_a", True)
    pool.update_stats("opp_a", False)
    assert abs(pool.get_win_rate("opp_a") - 2 / 3) < 1e-9


def test_pfsp_weights_favor_low_winrate():
    pool = OpponentPool("/tmp/test_pool", "p1")
    pool.checkpoints = ["/tmp/a", "/tmp/b"]
    pool.win_stats = {}

    # a: 80% win rate, b: 20% win rate
    for _ in range(8):
        pool.update_stats("a", True)
    for _ in range(2):
        pool.update_stats("a", False)
    for _ in range(2):
        pool.update_stats("b", True)
    for _ in range(8):
        pool.update_stats("b", False)

    weights = pool._pfsp_weights()
    # b should have higher weight (lower win rate -> more practice needed)
    assert weights[1] > weights[0]
