"""Game simulation and statistics collection using pika-zoo's RecordEpisode."""

from __future__ import annotations

import numpy as np
from pika_zoo.ai import BuiltinAI
from pika_zoo.env.pikachu_volleyball import PikachuVolleyballEnv
from pika_zoo.wrappers.normalize_observation import NormalizeObservation
from pika_zoo.wrappers.record_episode import EpisodeRecord, RecordEpisode
from pika_zoo.wrappers.simplify_action import SimplifyAction
from stable_baselines3 import PPO


class Player:
    """A player that can participate in games."""

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


MAX_GAME_STEPS = 30000


def play_game(
    p1: Player,
    p2: Player,
    winning_score: int = 15,
    seed: int | None = None,
    max_game_steps: int = MAX_GAME_STEPS,
) -> EpisodeRecord:
    """Play one game and return an EpisodeRecord with round-by-round statistics.

    Uses pika-zoo's RecordEpisode wrapper for round detection and server tracking.
    Truncates games at max_game_steps.
    """
    ai_policies: dict[str, BuiltinAI] = {}
    if p1.player_type == "builtin":
        ai_policies["player_1"] = BuiltinAI()
    if p2.player_type == "builtin":
        ai_policies["player_2"] = BuiltinAI()

    raw_env = PikachuVolleyballEnv(winning_score=winning_score, serve="winner", ai_policies=ai_policies or None)
    env = RecordEpisode(raw_env, record_frames=False)
    env = SimplifyAction(env)
    env = NormalizeObservation(env)

    obs, _info = env.reset(seed=seed)
    total_steps = 0

    while env.agents:
        actions = {
            "player_1": p1.get_action(obs.get("player_1"), env, "player_1"),
            "player_2": p2.get_action(obs.get("player_2"), env, "player_2"),
        }
        obs, _rewards, _terminated, _truncated, _infos = env.step(actions)
        total_steps += 1

        if total_steps >= max_game_steps:
            break

    # Get the record from RecordEpisode (traverse wrapper chain)
    record_wrapper = env
    while not isinstance(record_wrapper, RecordEpisode):
        record_wrapper = record_wrapper.env
    episode = record_wrapper.get_episode_record()

    env.close()
    return episode


def analyze_games(
    p1_spec: str,
    p2_spec: str,
    games: int = 100,
    winning_score: int = 15,
    seed: int = 42,
) -> list[EpisodeRecord]:
    """Run multiple games and print aggregate statistics."""
    p1 = make_player(p1_spec)
    p2 = make_player(p2_spec)
    rng = np.random.default_rng(seed)

    all_episodes: list[EpisodeRecord] = []
    for _ in range(games):
        game_seed = int(rng.integers(0, 2**31))
        episode = play_game(p1, p2, winning_score=winning_score, seed=game_seed)
        all_episodes.append(episode)

    p1_wins = sum(1 for e in all_episodes if e.winner == "player_1")
    all_rounds = [r for e in all_episodes for r in e.rounds]
    p1_serve_rounds = [r for r in all_rounds if r.server == "player_1"]
    p2_serve_rounds = [r for r in all_rounds if r.server == "player_2"]
    durations = [r.duration for r in all_rounds]

    print(f"=== {p1.name} (p1) vs {p2.name} (p2) — {games} games, {winning_score} pts ===\n")
    print(f"Record: p1 {p1_wins}W {games - p1_wins}L ({p1_wins / games * 100:.0f}%)\n")

    if p1_serve_rounds:
        p1s_p1w = sum(1 for r in p1_serve_rounds if r.scorer == "player_1")
        pct = p1s_p1w / len(p1_serve_rounds) * 100
        print(f"p1 serve → p1 wins: {p1s_p1w}/{len(p1_serve_rounds)} ({pct:.1f}%)")
    if p2_serve_rounds:
        p2s_p2w = sum(1 for r in p2_serve_rounds if r.scorer == "player_2")
        pct = p2s_p2w / len(p2_serve_rounds) * 100
        print(f"p2 serve → p2 wins: {p2s_p2w}/{len(p2_serve_rounds)} ({pct:.1f}%)")

    if durations:
        print(
            f"\nRound frames: mean {np.mean(durations):.1f}, "
            f"median {np.median(durations):.1f}, "
            f"range {np.min(durations)}-{np.max(durations)}"
        )

    return all_episodes
