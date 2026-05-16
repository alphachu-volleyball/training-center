import pytest

from training_center.policy_config import (
    ensure_policy_config_matches_init,
    policy_request_is_explicit,
    resolve_policy_config,
)


def test_resolve_policy_config_defaults_to_mlp_64x64():
    policy, kwargs = resolve_policy_config()

    assert policy == "MlpPolicy"
    assert kwargs == {"net_arch": {"pi": [64, 64], "vf": [64, 64]}}


def test_resolve_policy_config_accepts_net_arch_shorthand():
    policy, kwargs = resolve_policy_config(net_arch=[128, 128])

    assert policy == "MlpPolicy"
    assert kwargs == {"net_arch": {"pi": [128, 128], "vf": [128, 128]}}


def test_resolve_policy_config_merges_policy_kwargs_json():
    policy, kwargs = resolve_policy_config(
        policy="MlpPolicy",
        net_arch=[128, 128],
        policy_kwargs_json='{"ortho_init": false, "net_arch": {"pi": [32], "vf": [32]}}',
    )

    assert policy == "MlpPolicy"
    assert kwargs == {
        "ortho_init": False,
        "net_arch": {"pi": [128, 128], "vf": [128, 128]},
    }


def test_policy_request_is_explicit_only_for_user_supplied_architecture():
    assert not policy_request_is_explicit(policy=None, net_arch=None, policy_kwargs_json=None)
    assert policy_request_is_explicit(policy="MlpPolicy", net_arch=None, policy_kwargs_json=None)
    assert policy_request_is_explicit(policy=None, net_arch=[128, 128], policy_kwargs_json=None)
    assert policy_request_is_explicit(policy=None, net_arch=None, policy_kwargs_json='{"ortho_init": false}')


def test_init_policy_mismatch_is_rejected():
    with pytest.raises(ValueError, match="init-model"):
        ensure_policy_config_matches_init(
            init_policy="MlpPolicy",
            init_policy_kwargs={"net_arch": {"pi": [64, 64], "vf": [64, 64]}},
            requested_policy="MlpPolicy",
            requested_policy_kwargs={"net_arch": {"pi": [128, 128], "vf": [128, 128]}},
        )
