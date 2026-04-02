"""Tests for ELO rating calculation (batch Bradley-Terry)."""

from training_center.elo import INITIAL_ELO, compute_elo


def test_equal_players_get_equal_elo():
    results = {("A", "B"): (50, 50)}
    elos = compute_elo(results)
    assert abs(elos["A"] - elos["B"]) < 1e-3


def test_winner_gets_higher_elo():
    results = {("A", "B"): (80, 20)}
    elos = compute_elo(results)
    assert elos["A"] > elos["B"]


def test_order_independent():
    """Swapping input order must not change ratings."""
    r1 = {("A", "B"): (70, 30), ("A", "C"): (60, 40), ("B", "C"): (55, 45)}
    r2 = {("B", "C"): (55, 45), ("A", "C"): (60, 40), ("A", "B"): (70, 30)}
    elos1 = compute_elo(r1)
    elos2 = compute_elo(r2)
    for p in ("A", "B", "C"):
        assert abs(elos1[p] - elos2[p]) < 1e-3


def test_three_players_ranking():
    results = {("A", "B"): (90, 10), ("B", "C"): (80, 20), ("A", "C"): (95, 5)}
    elos = compute_elo(results)
    assert elos["A"] > elos["B"] > elos["C"]


def test_mean_elo_is_initial():
    """Geometric mean of strengths maps to INITIAL_ELO."""
    results = {("A", "B"): (70, 30), ("A", "C"): (60, 40), ("B", "C"): (55, 45)}
    elos = compute_elo(results)
    mean_elo = sum(elos.values()) / len(elos)
    assert abs(mean_elo - INITIAL_ELO) < 5  # approximately centered


def test_no_games_played():
    elos = compute_elo({("A", "B"): (0, 0)})
    # No games played, should still return ratings
    assert "A" in elos and "B" in elos
