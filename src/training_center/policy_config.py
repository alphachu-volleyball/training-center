"""Policy architecture configuration helpers for SB3 PPO training."""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from typing import Any

DEFAULT_POLICY = "MlpPolicy"
DEFAULT_NET_ARCH = [64, 64]


def _mlp_net_arch_kwargs(net_arch: list[int]) -> dict[str, Any]:
    """Return SB3 policy kwargs for symmetric policy/value MLP heads."""
    return {"net_arch": {"pi": list(net_arch), "vf": list(net_arch)}}


def default_policy_kwargs(policy: str = DEFAULT_POLICY) -> dict[str, Any]:
    """Default policy kwargs used for new models.

    SB3's default MlpPolicy uses 64-64 policy/value heads. We store that
    explicitly so model artifacts and W&B configs record the architecture.
    Other policy types default to no extra kwargs unless the CLI provides them.
    """
    if policy == DEFAULT_POLICY:
        return _mlp_net_arch_kwargs(DEFAULT_NET_ARCH)
    return {}


def parse_policy_kwargs_json(value: str | dict[str, Any] | None) -> dict[str, Any]:
    """Parse a JSON object passed to ``--policy-kwargs-json``."""
    if value is None or value == "":
        return {}
    if isinstance(value, dict):
        return deepcopy(value)
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("--policy-kwargs-json must decode to a JSON object")
    return parsed


def resolve_policy_config(
    *,
    policy: str | None = None,
    net_arch: list[int] | None = None,
    policy_kwargs_json: str | dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    """Resolve CLI policy settings into ``(policy, policy_kwargs)`` for PPO."""
    resolved_policy = policy or DEFAULT_POLICY
    policy_kwargs = parse_policy_kwargs_json(policy_kwargs_json)

    if net_arch is not None:
        if not net_arch:
            raise ValueError("--net-arch requires at least one layer size")
        if any(width <= 0 for width in net_arch):
            raise ValueError("--net-arch layer sizes must be positive integers")
        policy_kwargs.update(_mlp_net_arch_kwargs(net_arch))
    elif "net_arch" not in policy_kwargs:
        policy_kwargs.update(default_policy_kwargs(resolved_policy))

    return resolved_policy, policy_kwargs


def policy_request_is_explicit(
    *, policy: str | None, net_arch: list[int] | None, policy_kwargs_json: str | dict[str, Any] | None
) -> bool:
    """Whether the user/sweep explicitly requested a policy architecture."""
    return policy is not None or net_arch is not None or policy_kwargs_json not in (None, "")


def ensure_policy_config_matches_init(
    *,
    init_policy: str,
    init_policy_kwargs: dict[str, Any],
    requested_policy: str,
    requested_policy_kwargs: dict[str, Any],
) -> None:
    """Reject ambiguous attempts to resume a model with a different architecture."""
    if init_policy != requested_policy or init_policy_kwargs != requested_policy_kwargs:
        raise ValueError(
            "--init-model already contains a policy architecture. "
            "Do not pass --policy, --net-arch, or --policy-kwargs-json unless they match the init model. "
            f"init=({init_policy}, {init_policy_kwargs}), requested=({requested_policy}, {requested_policy_kwargs})"
        )


def add_policy_args(parser: argparse.ArgumentParser) -> None:
    """Add common policy architecture arguments to a training CLI parser."""
    parser.add_argument(
        "--policy",
        default=None,
        help=f"SB3 policy type/class name for new models (default: {DEFAULT_POLICY})",
    )
    parser.add_argument(
        "--net-arch",
        nargs="+",
        type=int,
        default=None,
        metavar="N",
        help=(
            "MLP layer sizes for symmetric policy/value heads, e.g. --net-arch 128 128 "
            f"(default for MlpPolicy: {' '.join(map(str, DEFAULT_NET_ARCH))})"
        ),
    )
    parser.add_argument(
        "--policy-kwargs-json",
        default=None,
        help=(
            "Advanced SB3 policy_kwargs JSON object. "
            "Use with --policy for non-default architectures; --net-arch overrides its net_arch key."
        ),
    )
