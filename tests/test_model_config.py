"""Tests for model wrapper configuration."""

from training_center.model_config import ModelConfig


def test_model_config_round_trips_frame_stack(tmp_path):
    path = tmp_path / "model.json"
    config = ModelConfig(side="both", observation_simplified=True, frame_stack=4)

    config.save(path)
    loaded = ModelConfig.load(path)

    assert loaded.side == "both"
    assert loaded.observation_simplified
    assert loaded.frame_stack == 4
