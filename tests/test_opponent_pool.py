"""Tests for PFP opponent pool."""

from training_center.pool import OpponentPool
from training_center.pool.common import PFPMixin


def test_empty_pool_win_rate():
    pool = OpponentPool("/tmp/test_pool", "p1")
    assert pool.get_win_rate("nonexistent") == 0.5


def test_set_win_rate():
    pool = OpponentPool("/tmp/test_pool", "p1")
    pool.set_win_rate("opp_a", 0.66)
    assert abs(pool.get_win_rate("opp_a") - 0.66) < 1e-9


def test_pfp_weights_favor_low_winrate():
    pool = OpponentPool("/tmp/test_pool", "p1")
    pool.set_win_rate("a", 0.8)
    pool.set_win_rate("b", 0.2)

    # b should have higher weight (lower win rate -> more practice needed)
    weight_a = PFPMixin.pfp_weight(pool.get_win_rate("a"))
    weight_b = PFPMixin.pfp_weight(pool.get_win_rate("b"))
    assert weight_b > weight_a
