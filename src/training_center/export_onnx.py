"""Export SB3 PPO policy network to ONNX for browser inference.

Extracts only the actor (policy) network from an SB3 PPO model and exports
it as an ONNX file. The value network is excluded since it's not needed
for inference.

Output layout:
    output_dir/
    ├── model.onnx        # Policy network
    └── model.json        # Wrapper configuration (copied from source)

The ONNX model expects:
    Input:  "obs"            float32 [1, 35]
    Output: "action_logits"  float32 [1, 13]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from training_center.model_config import MODEL_CONFIG_NAME, load_model_config

ONNX_MODEL_NAME = "model.onnx"


class PolicyNetwork(nn.Module):
    """Standalone policy network extracted from SB3 ActorCriticPolicy.

    Architecture: obs → FlattenExtractor → MLP policy_net → action_net → logits
    """

    def __init__(self, policy) -> None:
        super().__init__()
        self.features_extractor = policy.features_extractor
        self.policy_net = policy.mlp_extractor.policy_net
        self.action_net = policy.action_net

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.features_extractor(obs)
        latent = self.policy_net(features)
        return self.action_net(latent)


def export_onnx(model_spec: str, output_dir: str | Path, *, name: str | None = None, opset_version: int = 17) -> Path:
    """Export an SB3 PPO model's policy network to ONNX.

    Args:
        model_spec: Path to model directory or .zip file.
        output_dir: Directory to write model.onnx and model.json into.
        name: Display name for the model (written to model.json).
        opset_version: ONNX opset version (default 17).

    Returns:
        Path to the output directory.
    """
    from stable_baselines3 import PPO

    zip_path, config = load_model_config(model_spec)
    if name is not None:
        config.name = name
    model = PPO.load(zip_path, device="cpu")

    policy_net = PolicyNetwork(model.policy)
    policy_net.eval()

    obs_size = model.observation_space.shape[0]
    dummy_input = torch.randn(1, obs_size)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / ONNX_MODEL_NAME

    torch.onnx.export(
        policy_net,
        dummy_input,
        str(onnx_path),
        input_names=["obs"],
        output_names=["action_logits"],
        dynamic_axes={"obs": {0: "batch"}, "action_logits": {0: "batch"}},
        opset_version=opset_version,
        dynamo=False,
    )

    # Validate exported model
    import onnx

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    # Copy config
    config.save(output_dir / MODEL_CONFIG_NAME)

    # Print summary
    input_shape = [d.dim_value for d in onnx_model.graph.input[0].type.tensor_type.shape.dim]
    output_shape = [d.dim_value for d in onnx_model.graph.output[0].type.tensor_type.shape.dim]
    onnx_size_kb = onnx_path.stat().st_size / 1024

    print(f"Exported: {onnx_path}")
    if config.name:
        print(f"  Name:   {config.name}")
    print(f"  Input:  obs {input_shape}")
    print(f"  Output: action_logits {output_shape}")
    print(f"  Size:   {onnx_size_kb:.1f} KB")
    print(
        f"  Config: side={config.side}, action_simplified={config.action_simplified}, "
        f"obs_simplified={config.observation_simplified}, obs_normalized={config.observation_normalized}"
    )

    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Export SB3 PPO policy to ONNX")
    parser.add_argument("model", help="Path to model directory or .zip file")
    parser.add_argument("output", help="Output directory for model.onnx + model.json")
    parser.add_argument("--name", help="Display name for the model (saved in model.json)")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version (default: 17)")
    args = parser.parse_args()

    export_onnx(args.model, args.output, name=args.name, opset_version=args.opset)


if __name__ == "__main__":
    main()
