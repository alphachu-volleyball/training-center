"""Game simulation and statistics collection using pika-zoo's RecordGame."""

from __future__ import annotations

import numpy as np
from pika_zoo.ai import BuiltinAI, RandomAI
from pika_zoo.ai.protocol import AIPolicy
from pika_zoo.ai.sb3_adapter import SB3ModelPolicy
from pika_zoo.env.pikachu_volleyball import PikachuVolleyballEnv
from pika_zoo.records.types import GameRecord, GamesRecord
from pika_zoo.wrappers.normalize_observation import NormalizeObservation
from pika_zoo.wrappers.record_game import RecordGame
from pika_zoo.wrappers.simplify_action import SimplifyAction
from pika_zoo.wrappers.simplify_observation import SimplifyObservation
from stable_baselines3 import PPO


class Player:
    """A player that can participate in games.

    Supports two modes:
    - AIPolicy-based: builtin, random, or SB3ModelPolicy (from path). Used for
      evaluation and analysis — the policy is registered in the env's ai_policies.
    - In-memory model: PPO model passed directly (from training). Actions are
      predicted through the wrapper chain.
    """

    def __init__(self, name: str, policy: AIPolicy | None = None, model: PPO | None = None) -> None:
        self.name = name
        self.policy = policy
        self.model = model

    def get_action(self, obs: np.ndarray) -> int:
        """Predict action from in-memory model (wrapper-chain path only)."""
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)


def make_player(spec: str, agent: str = "player_1", simplify_observation: bool = False) -> Player:
    """Create a Player from a string spec: 'random', 'builtin', or a model path.

    For model paths, ``agent`` determines which side the model was trained on,
    so that SB3ModelPolicy can correctly map actions and observations.
    ``simplify_observation`` must match the training-time SimplifyObservation setting.
    """
    if spec == "random":
        return Player("random", policy=RandomAI())
    elif spec == "builtin":
        return Player("builtin", policy=BuiltinAI())
    else:
        name = spec.rstrip("/").split("/")[-1]
        return Player(
            name,
            policy=SB3ModelPolicy(
                model_path=spec,
                agent=agent,
                action_simplified=True,
                observation_simplified=simplify_observation,
                observation_normalized=True,
            ),
        )


MAX_GAME_STEPS = 30000


def play_game(
    p1: Player,
    p2: Player,
    winning_score: int = 15,
    seed: int | None = None,
    max_game_steps: int = MAX_GAME_STEPS,
    simplify_observation: bool = False,
) -> GameRecord:
    """Play one game and return a GameRecord with round-by-round statistics.

    When both players have AIPolicy, the env handles all actions internally.
    When a player has an in-memory model, a wrapper chain is used for prediction.
    ``simplify_observation`` must match the training-time setting for in-memory models.
    Truncates games at max_game_steps.
    """
    ai_policies: dict[str, AIPolicy] = {}
    if p1.policy:
        ai_policies["player_1"] = p1.policy
    if p2.policy:
        ai_policies["player_2"] = p2.policy

    needs_wrappers = p1.model is not None or p2.model is not None

    raw_env = PikachuVolleyballEnv(winning_score=winning_score, serve="winner", ai_policies=ai_policies or None)
    env = RecordGame(raw_env, record_frames=False)

    if needs_wrappers:
        env = SimplifyAction(env)
        if simplify_observation:
            env = SimplifyObservation(env)
        env = NormalizeObservation(env)

    obs, _info = env.reset(seed=seed)
    total_steps = 0

    while env.agents:
        if needs_wrappers:
            actions = {}
            for agent_id, player in [("player_1", p1), ("player_2", p2)]:
                if player.model:
                    actions[agent_id] = player.get_action(obs.get(agent_id))
                else:
                    actions[agent_id] = 0  # ai_policy handles it
        else:
            actions = {a: 0 for a in env.agents}

        obs, _rewards, _terminated, _truncated, _infos = env.step(actions)
        total_steps += 1

        if total_steps >= max_game_steps:
            break

    # Traverse wrapper chain to find RecordGame
    record_wrapper = env
    while not isinstance(record_wrapper, RecordGame):
        record_wrapper = record_wrapper.env
    game = record_wrapper.get_game_record()

    env.close()
    return game


def analyze_games(
    p1_spec: str,
    p2_spec: str,
    games: int = 100,
    winning_score: int = 15,
    seed: int = 42,
) -> GamesRecord:
    """Run multiple games and print aggregate statistics."""
    p1 = make_player(p1_spec, agent="player_1")
    p2 = make_player(p2_spec, agent="player_2")
    rng = np.random.default_rng(seed)

    all_games: list[GameRecord] = []
    for _ in range(games):
        game_seed = int(rng.integers(0, 2**31))
        game = play_game(p1, p2, winning_score=winning_score, seed=game_seed)
        all_games.append(game)

    record = GamesRecord(games=all_games)

    p1_wins = record.win_counts.get("player_1", 0)
    all_rounds = [r for g in all_games for r in g.rounds]
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

    return record
