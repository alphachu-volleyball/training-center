"""ELO rating calculation."""

from __future__ import annotations

from itertools import combinations

import numpy as np

from training_center.eval.game import Player, make_player, play_game

INITIAL_ELO = 1500
K_FACTOR = 32


def update_elo(ra: float, rb: float, result: float, k: float = K_FACTOR) -> tuple[float, float]:
    """Update ELO ratings. result: 1=A wins, 0=B wins, 0.5=draw."""
    ea = 1 / (1 + 10 ** ((rb - ra) / 400))
    eb = 1 - ea
    ra_new = ra + k * (result - ea)
    rb_new = rb + k * ((1 - result) - eb)
    return ra_new, rb_new


def round_robin(
    players: list[Player],
    games_per_pair: int = 100,
    winning_score: int = 15,
    seed: int | None = None,
) -> tuple[dict[tuple[str, str], tuple[int, int]], dict[str, float]]:
    """Run round-robin tournament. Returns (results, elo_ratings)."""
    rng = np.random.default_rng(seed)
    results: dict[tuple[str, str], tuple[int, int]] = {}
    elos = {p.name: INITIAL_ELO for p in players}

    for p1, p2 in combinations(players, 2):
        key = (p1.name, p2.name)
        wins = 0
        for _i in range(games_per_pair):
            game_seed = int(rng.integers(0, 2**31))
            stats = play_game(p1, p2, winning_score=winning_score, seed=game_seed)
            result = 1 if stats.winner == "player_1" else 0
            wins += result
            elos[p1.name], elos[p2.name] = update_elo(elos[p1.name], elos[p2.name], result)
        results[key] = (wins, games_per_pair - wins)

    return results, elos


def evaluate_model(
    model_path: str,
    opponents: tuple[str, ...] = ("random", "builtin"),
    games: int = 100,
    winning_score: int = 15,
    seed: int | None = None,
) -> tuple[dict[str, tuple[int, int]], float]:
    """Evaluate a single model against multiple opponents. Returns (results, model_elo)."""
    model_player = make_player(model_path)
    rng = np.random.default_rng(seed)
    elos: dict[str, float] = {model_player.name: INITIAL_ELO}

    results: dict[str, tuple[int, int]] = {}
    for opp_spec in opponents:
        opp = make_player(opp_spec)
        elos.setdefault(opp.name, INITIAL_ELO)

        wins = 0
        for _ in range(games):
            game_seed = int(rng.integers(0, 2**31))
            stats = play_game(model_player, opp, winning_score=winning_score, seed=game_seed)
            result = 1 if stats.winner == "player_1" else 0
            wins += result
            elos[model_player.name], elos[opp.name] = update_elo(elos[model_player.name], elos[opp.name], result)
        results[opp.name] = (wins, games - wins)

    return results, elos[model_player.name]
