"""Environment factory: wrapper chain construction and opponent policy management."""

from __future__ import annotations

import platform
import resource
from collections.abc import Callable
from typing import Any

import numpy as np
from pika_zoo.ai.protocol import AIPolicy
from pika_zoo.env.pikachu_volleyball import NoiseConfig, PikachuVolleyballEnv
from pika_zoo.wrappers.convert_single_agent import ConvertSingleAgent
from pika_zoo.wrappers.normalize_observation import NormalizeObservation
from pika_zoo.wrappers.reward_shaping import RewardShaping
from pika_zoo.wrappers.simplify_action import SimplifyAction
from pika_zoo.wrappers.simplify_observation import SimplifyObservation
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

# DuckllAI's ball prediction can exhaust the default 8MB stack in
# forkserver / SubprocVecEnv workers on Linux.  Raising the soft limit
# early in the main process ensures every child inherits a larger stack.
_STACK_SIZE = 64 * 1024 * 1024  # 64 MB


def ensure_stack_size(size: int = _STACK_SIZE) -> None:
    """Raise the process stack soft limit if below *size* (Linux only)."""
    if platform.system() != "Linux":
        return
    soft, hard = resource.getrlimit(resource.RLIMIT_STACK)
    if soft < size:
        new_hard = hard if hard == -1 else max(hard, size)
        resource.setrlimit(resource.RLIMIT_STACK, (size, new_hard))


def make_env(
    agent: str = "player_1",
    opponent_policy: AIPolicy | Callable[[np.ndarray], int] | None = None,
    winning_score: int = 15,
    serve: str = "winner",
    simplify_observation: bool = False,
    reward_shaping: bool = False,
    ball_position_coeff: float = 0.01,
    normal_state_coeff: float = 0.0,
    noise: NoiseConfig | None = None,
    seed: int | None = None,
) -> ConvertSingleAgent:
    """Build the full wrapper chain and return a gym.Env for SB3.

    Chain: PikachuVolleyballEnv → SimplifyAction → [SimplifyObservation]
           → NormalizeObservation → [RewardShaping] → ConvertSingleAgent
    """
    env = PikachuVolleyballEnv(winning_score=winning_score, serve=serve, noise=noise)

    env = SimplifyAction(env)
    if simplify_observation:
        env = SimplifyObservation(env)
    env = NormalizeObservation(env)

    if reward_shaping:
        env = RewardShaping(env, ball_position_coeff=ball_position_coeff, normal_state_coeff=normal_state_coeff)

    gym_env = ConvertSingleAgent(env, agent=agent, opponent_policy=opponent_policy)

    if seed is not None:
        gym_env.reset(seed=seed)

    return gym_env


def set_opponent_policy(
    env: ConvertSingleAgent,
    policy: AIPolicy | Callable[[np.ndarray], int] | None,
) -> None:
    """Swap the opponent policy on a ConvertSingleAgent in-place.

    Handles the difference between AIPolicy (physics-level) and callable (obs-level).
    """
    opponent_id = env._opponent

    # Clear previous AIPolicy if it was set
    if env._opponent_is_ai and opponent_id in env._env.ai_policies:
        del env._env.ai_policies[opponent_id]

    env._opponent_policy = policy

    if isinstance(policy, AIPolicy):
        env._env.ai_policies[opponent_id] = policy
        env._opponent_is_ai = True
    else:
        env._opponent_is_ai = False


def _find_parallel_env(env: ConvertSingleAgent) -> PikachuVolleyballEnv:
    """Traverse the wrapper chain to find the underlying PikachuVolleyballEnv."""
    inner = env._env
    while hasattr(inner, "env"):
        if isinstance(inner, PikachuVolleyballEnv):
            return inner
        inner = inner.env
    return inner


def make_vec_env(
    n_envs: int,
    agent: str = "player_1",
    opponent_policy: AIPolicy | Callable[[np.ndarray], int] | None = None,
    use_subproc: bool = True,
    seed: int = 0,
    **env_kwargs: Any,
) -> VecEnv:
    """Create a vectorized environment.

    Args:
        n_envs: Number of parallel environments.
        agent: Which agent the learner controls.
        opponent_policy: Opponent policy (AIPolicy, callable, or None for random).
        use_subproc: True for SubprocVecEnv (train_ppo), False for DummyVecEnv (cross-play).
        seed: Base seed (each env gets seed + rank).
        **env_kwargs: Forwarded to make_env.
    """

    def _make(rank: int) -> Callable[[], ConvertSingleAgent]:
        def _init() -> ConvertSingleAgent:
            return make_env(agent=agent, opponent_policy=opponent_policy, seed=seed + rank, **env_kwargs)

        return _init

    env_fns = [_make(i) for i in range(n_envs)]

    if use_subproc:
        return SubprocVecEnv(env_fns)
    return DummyVecEnv(env_fns)
