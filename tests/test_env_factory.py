"""Tests for env_factory module."""

import numpy as np
from pika_zoo.ai.builtin import BuiltinAI
from pika_zoo.wrappers.convert_single_agent import ConvertSingleAgent

from training_center.env_factory import make_env, make_vec_env, set_opponent_policy


def test_make_env_returns_gym_env():
    env = make_env(agent="player_1", seed=0)
    assert isinstance(env, ConvertSingleAgent)
    obs, info = env.reset()
    assert obs.shape == env.observation_space.shape
    env.close()


def test_make_env_with_builtin():
    env = make_env(agent="player_1", opponent_policy=BuiltinAI(), seed=0)
    assert env._opponent_is_ai
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(0)
    assert obs.shape == env.observation_space.shape
    env.close()


def test_set_opponent_policy_callable():
    env = make_env(agent="player_1", seed=0)
    assert not env._opponent_is_ai

    def dummy_policy(obs: np.ndarray) -> int:
        return 0

    set_opponent_policy(env, dummy_policy)
    assert not env._opponent_is_ai
    assert env._opponent_policy is dummy_policy
    env.close()


def test_set_opponent_policy_ai():
    env = make_env(agent="player_1", seed=0)
    builtin = BuiltinAI()
    set_opponent_policy(env, builtin)
    assert env._opponent_is_ai
    assert env._opponent == "player_2"

    # Switch back to callable
    set_opponent_policy(env, lambda obs: 0)
    assert not env._opponent_is_ai
    env.close()


def test_make_vec_env_dummy():
    vec_env = make_vec_env(n_envs=2, agent="player_1", use_subproc=False, seed=0)
    obs = vec_env.reset()
    assert obs.shape[0] == 2
    vec_env.close()
