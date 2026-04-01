"""Model configuration: save/load wrapper settings alongside SB3 model files.

Model directory layout:
    model_dir/
    ├── model.zip         # SB3 PPO weights
    └── model.json        # Wrapper configuration

When a .zip path is given directly (legacy), default config is assumed.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from stable_baselines3 import PPO

MODEL_ZIP_NAME = "model.zip"
MODEL_CONFIG_NAME = "model.json"


@dataclass
class ModelConfig:
    """Wrapper chain configuration for an SB3 model."""

    side: str = "player_1"
    action_simplified: bool = True
    observation_simplified: bool = False
    observation_normalized: bool = True

    def save(self, path: Path) -> None:
        """Save config as JSON."""
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> ModelConfig:
        """Load config from JSON."""
        data = json.loads(path.read_text())
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def default(cls) -> ModelConfig:
        return cls()


def save_model(model: PPO, save_dir: str | Path, config: ModelConfig) -> Path:
    """Save SB3 model + config to a directory.

    Args:
        model: Trained PPO model.
        save_dir: Directory to save into (created if needed).
        config: Wrapper configuration.

    Returns:
        Path to the save directory.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model_path = save_dir / MODEL_ZIP_NAME.removesuffix(".zip")
    model.save(str(model_path))

    config.save(save_dir / MODEL_CONFIG_NAME)
    return save_dir


def load_model_config(spec: str) -> tuple[str, ModelConfig]:
    """Resolve a model spec to (zip_path, config).

    Supported specs:
    - Directory path: reads model.zip and model.json from inside.
    - .zip file path: uses default config (legacy compatibility).
    - Path without extension: tries as directory first, then appends .zip.

    Returns:
        (zip_path, ModelConfig)
    """
    p = Path(spec)

    # Directory: look for model.zip + model.json inside
    if p.is_dir():
        zip_path = p / MODEL_ZIP_NAME
        config_path = p / MODEL_CONFIG_NAME
        if not zip_path.exists():
            raise FileNotFoundError(f"No {MODEL_ZIP_NAME} in {p}")
        config = ModelConfig.load(config_path) if config_path.exists() else ModelConfig.default()
        return str(zip_path), config

    # Explicit .zip file
    if p.suffix == ".zip" and p.exists():
        config_path = p.parent / MODEL_CONFIG_NAME
        config = ModelConfig.load(config_path) if config_path.exists() else ModelConfig.default()
        return str(p), config

    # Path without extension — try as directory, then .zip
    if p.exists() and p.is_dir():
        return load_model_config(str(p))
    zip_candidate = Path(str(p) + ".zip")
    if zip_candidate.exists():
        config_path = p.parent / MODEL_CONFIG_NAME
        config = ModelConfig.load(config_path) if config_path.exists() else ModelConfig.default()
        return str(zip_candidate), config

    raise FileNotFoundError(f"Cannot resolve model spec: {spec}")
