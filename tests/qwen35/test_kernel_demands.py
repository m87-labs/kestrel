"""Model requirements must not shrink to the packaged kernel inventory."""
from dataclasses import replace

import pytest

from kestrel.models.qwen35.qwen_config import Qwen3_5TextConfig
from kestrel_kernels.native_coverage import ExtentBound


def config():
    return Qwen3_5TextConfig(
        vocab_size=32, hidden_size=8, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1,
        max_position_embeddings=128, rms_norm_eps=1e-6,
        tie_word_embeddings=False, rope_theta=10000.,
        partial_rotary_factor=1., mrope_section=(1, 1, 1), head_dim=4,
        linear_conv_kernel_dim=4, linear_key_head_dim=2,
        linear_value_head_dim=4, linear_num_key_heads=1,
        linear_num_value_heads=2,
        layer_types=("linear_attention", "full_attention"), intermediate_size=12,
    )


@pytest.mark.parametrize("capacity", [1, 2, 4, 8, 13])
@pytest.mark.parametrize("width", [1, 8, 17])
def test_prefix_demand_includes_all_retiring_cohorts_and_legal_commits(capacity, width):
    demand = config().retained_prefix_demand(capacity=capacity, verification_width=width)
    bounds = dict(demand.dynamic_bounds)
    assert bounds == {"sequences": ExtentBound(1, capacity),
                      "accepted_length": ExtentBound(1, width)}
    assert dict(demand.semantics) == {
        "initial_state": True, "preserve_initial_state": True,
        "commit": "per_sequence_accepted_prefix", "graph_capture": True,
    }
    assert demand.dimension("key_head_dim") == 2
    assert demand.dimension("value_head_dim") == 4
    assert demand.dimension("verification_width") == width


def test_new_geometry_changes_requirement_without_consulting_inventory():
    old = config()
    new = replace(old, linear_num_key_heads=16, linear_num_value_heads=48,
                  linear_key_head_dim=128, linear_value_head_dim=128)
    before = old.retained_prefix_demand(capacity=8, verification_width=8)
    after = new.retained_prefix_demand(capacity=8, verification_width=8)
    assert before != after
    assert after.dimension("num_k_heads") == 16
    assert after.dimension("num_v_heads") == 48


@pytest.mark.parametrize("name", ["capacity", "verification_width"])
@pytest.mark.parametrize("invalid", [0, -1, True, 1.5])
def test_invalid_serving_range_cannot_produce_vacuous_coverage(name, invalid):
    args = dict(capacity=8, verification_width=8)
    args[name] = invalid
    with pytest.raises(ValueError, match=name):
        config().retained_prefix_demand(**args)


def test_nonrecurrent_model_does_not_claim_recurrent_requirement():
    target = replace(config(), layer_types=("full_attention",) * 2)
    with pytest.raises(ValueError, match="no retained recurrent"):
        target.retained_prefix_demand(capacity=8, verification_width=8)
