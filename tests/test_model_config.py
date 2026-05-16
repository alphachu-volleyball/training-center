"""Tests for model wrapper configuration."""

from training_center.model_config import ModelConfig


def test_model_config_round_trips_frame_stack(tmp_path):
    path = tmp_path / "model.json"
    config = ModelConfig(
        side="both",
        observation_simplified=True,
        frame_stack=4,
        policy_kwargs={"net_arch": {"pi": [128, 128], "vf": [128, 128]}},
    )

    config.save(path)
    loaded = ModelConfig.load(path)

    assert loaded.side == "both"
    assert loaded.observation_simplified
    assert loaded.frame_stack == 4
    assert loaded.policy == "MlpPolicy"
    assert loaded.policy_kwargs == {"net_arch": {"pi": [128, 128], "vf": [128, 128]}}


def test_model_config_uses_default_mlp_policy_kwargs():
    config = ModelConfig()

    assert config.policy == "MlpPolicy"
    assert config.policy_kwargs == {"net_arch": {"pi": [64, 64], "vf": [64, 64]}}
