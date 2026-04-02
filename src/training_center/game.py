"""Game simulation and statistics collection using pika-zoo's RecordGame."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pika_zoo.ai import BuiltinAI, DuckllAI, RandomAI, StoneAI
from pika_zoo.ai.protocol import AIPolicy
from pika_zoo.ai.sb3_adapter import SB3ModelPolicy
from pika_zoo.env.pikachu_volleyball import PikachuVolleyballEnv
from pika_zoo.records.types import GameRecord
from pika_zoo.wrappers.normalize_observation import NormalizeObservation
from pika_zoo.wrappers.record_game import RecordGame
from pika_zoo.wrappers.simplify_action import SimplifyAction
from pika_zoo.wrappers.simplify_observation import SimplifyObservation
from stable_baselines3 import PPO

from training_center.model_config import load_model_config


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
    """Create a Player from a string spec.

    Supported specs:
    - 'random', 'builtin', 'stone': built-in AI policies
    - 'duckll' or 'duckll:N': DuckllAI with optional preset level (0-10)
    - directory path: loads model.zip + model.json config from inside
    - .zip file path: loads with default config (legacy compatibility)

    For model paths, ``agent`` determines which side the model was trained on.
    If a model.json config exists, wrapper settings are loaded from it.
    Otherwise, defaults are used (action_simplified=True, observation_normalized=True).
    The ``simplify_observation`` parameter is only used as fallback when no config exists.
    """
    if spec == "random":
        return Player("random", policy=RandomAI())
    elif spec == "builtin":
        return Player("builtin", policy=BuiltinAI())
    elif spec == "stone":
        return Player("stone", policy=StoneAI())
    elif spec == "duckll" or spec.startswith("duckll:"):
        if ":" in spec:
            preset = int(spec.split(":")[1])
            return Player(f"duckll:{preset}", policy=DuckllAI(preset=preset))
        return Player("duckll", policy=DuckllAI())
    else:
        zip_path, config = load_model_config(spec)
        name = Path(spec).name
        return Player(
            name,
            policy=SB3ModelPolicy(
                model_path=zip_path,
                agent=agent if config.side == "both" else config.side,
                action_simplified=config.action_simplified,
                observation_simplified=config.observation_simplified,
                observation_normalized=config.observation_normalized,
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
    record_frames: bool = False,
) -> GameRecord:
    """Play one game and return a GameRecord with round-by-round statistics.

    When both players have AIPolicy, the env handles all actions internally.
    When a player has an in-memory model, a wrapper chain is used for prediction.
    ``simplify_observation`` must match the training-time setting for in-memory models.
    ``record_frames`` enables per-frame recording for detailed analysis.
    Truncates games at max_game_steps.
    """
    ai_policies: dict[str, AIPolicy] = {}
    if p1.policy:
        ai_policies["player_1"] = p1.policy
    if p2.policy:
        ai_policies["player_2"] = p2.policy

    needs_wrappers = p1.model is not None or p2.model is not None

    raw_env = PikachuVolleyballEnv(winning_score=winning_score, serve="winner", ai_policies=ai_policies or None)
    env = RecordGame(raw_env, record_frames=record_frames)

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


