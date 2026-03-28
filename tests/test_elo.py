"""Tests for ELO rating calculation."""

from training_center.eval.elo import INITIAL_ELO, update_elo


def test_update_elo_a_wins():
    ra, rb = update_elo(INITIAL_ELO, INITIAL_ELO, 1.0)
    assert ra > INITIAL_ELO
    assert rb < INITIAL_ELO
    assert abs((ra - INITIAL_ELO) + (rb - INITIAL_ELO)) < 1e-9  # zero-sum


def test_update_elo_b_wins():
    ra, rb = update_elo(INITIAL_ELO, INITIAL_ELO, 0.0)
    assert ra < INITIAL_ELO
    assert rb > INITIAL_ELO


def test_update_elo_draw():
    ra, rb = update_elo(INITIAL_ELO, INITIAL_ELO, 0.5)
    assert abs(ra - INITIAL_ELO) < 1e-9
    assert abs(rb - INITIAL_ELO) < 1e-9


def test_update_elo_stronger_wins_less():
    # When a stronger player (higher ELO) wins, they gain fewer points
    ra_strong, rb_weak = update_elo(1700, 1300, 1.0)
    ra_equal, rb_equal = update_elo(1500, 1500, 1.0)
    gain_strong = ra_strong - 1700
    gain_equal = ra_equal - 1500
    assert gain_strong < gain_equal
