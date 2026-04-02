"""Tests for PFSP opponent pool."""

from training_center.pool import OpponentPool
from training_center.pool.common import PFSPMixin


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

    # b should have higher weight (lower win rate -> more practice needed)
    weight_a = PFSPMixin.pfsp_weight(pool.get_win_rate("a"))
    weight_b = PFSPMixin.pfsp_weight(pool.get_win_rate("b"))
    assert weight_b > weight_a
