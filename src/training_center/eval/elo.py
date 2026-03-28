"""ELO rating calculation and match utilities."""

from __future__ import annotations

from itertools import combinations

import numpy as np
from pika_zoo.ai import BuiltinAI
from pika_zoo.env.pikachu_volleyball import PikachuVolleyballEnv
from pika_zoo.wrappers.normalize_observation import NormalizeObservation
from pika_zoo.wrappers.simplify_action import SimplifyAction
from stable_baselines3 import PPO

INITIAL_ELO = 1500
K_FACTOR = 32


def update_elo(ra: float, rb: float, result: float, k: float = K_FACTOR) -> tuple[float, float]:
    """Update ELO ratings. result: 1=A wins, 0=B wins, 0.5=draw."""
    ea = 1 / (1 + 10 ** ((rb - ra) / 400))
    eb = 1 - ea
    ra_new = ra + k * (result - ea)
    rb_new = rb + k * ((1 - result) - eb)
    return ra_new, rb_new


class Player:
    """A player that can participate in matches."""

    def __init__(self, name: str, player_type: str, model_path: str | None = None, model: PPO | None = None) -> None:
        self.name = name
        self.player_type = player_type  # "random", "builtin", "model"
        self.model = model
        if player_type == "model" and model_path and model is None:
            self.model = PPO.load(model_path, device="cpu")

    def get_action(self, obs: np.ndarray, env: PikachuVolleyballEnv, agent_id: str) -> int:
        if self.player_type == "random":
            return env.action_space(agent_id).sample()
        elif self.player_type == "builtin":
            return 0  # ai_policies handles the actual action
        elif self.player_type == "model":
            action, _ = self.model.predict(obs, deterministic=True)
            return int(action)
        raise ValueError(f"Unknown player type: {self.player_type}")


def make_player(spec: str) -> Player:
    """Create a Player from a string spec: 'random', 'builtin', or a model path."""
    if spec == "random":
        return Player("random", "random")
    elif spec == "builtin":
        return Player("builtin", "builtin")
    else:
        name = spec.rstrip("/").split("/")[-1]
        return Player(name, "model", model_path=spec)


def play_game(p1: Player, p2: Player, winning_score: int = 15, seed: int | None = None) -> int:
    """Play one game between two players. Returns 1 if p1 wins, 0 if p2 wins."""
    ai_policies: dict[str, BuiltinAI] = {}
    if p1.player_type == "builtin":
        ai_policies["player_1"] = BuiltinAI()
    if p2.player_type == "builtin":
        ai_policies["player_2"] = BuiltinAI()

    env = PikachuVolleyballEnv(winning_score=winning_score, serve="winner", ai_policies=ai_policies or None)
    env = SimplifyAction(env)
    env = NormalizeObservation(env)

    obs, _info = env.reset(seed=seed)

    total_rewards: dict[str, float] = {"player_1": 0.0, "player_2": 0.0}
    while env.agents:
        actions = {
            "player_1": p1.get_action(obs.get("player_1"), env, "player_1"),
            "player_2": p2.get_action(obs.get("player_2"), env, "player_2"),
        }
        obs, rewards, _terminated, _truncated, _infos = env.step(actions)
        for agent in rewards:
            total_rewards[agent] += rewards[agent]

    env.close()
    return 1 if total_rewards["player_1"] > total_rewards["player_2"] else 0


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
            result = play_game(p1, p2, winning_score=winning_score, seed=game_seed)
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
            result = play_game(model_player, opp, winning_score=winning_score, seed=game_seed)
            wins += result
            elos[model_player.name], elos[opp.name] = update_elo(elos[model_player.name], elos[opp.name], result)
        results[opp.name] = (wins, games - wins)

    return results, elos[model_player.name]
