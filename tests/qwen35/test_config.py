import pytest

from kestrel.models.qwen35.qwen_config import Qwen3_5Config


def _config(quantization):
    return {
        "model_type": "qwen3_5", "tie_word_embeddings": False,
        "image_token_id": 248056, "quantization_config": quantization,
        "text_config": {
            "vocab_size": 248320, "hidden_size": 5120,
            "intermediate_size": 17408, "num_hidden_layers": 64,
            "num_attention_heads": 24, "num_key_value_heads": 4,
            "max_position_embeddings": 262144, "rms_norm_eps": 1e-6,
            "head_dim": 256, "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128, "linear_value_head_dim": 128,
            "linear_num_key_heads": 16, "linear_num_value_heads": 48,
            "full_attention_interval": 4, "hidden_act": "silu",
            "mamba_ssm_dtype": "float32", "attention_bias": False,
            "rope_parameters": {
                "rope_type": "default", "mrope_interleaved": True,
                "rope_theta": 10000000, "partial_rotary_factor": 0.25,
                "mrope_section": [11, 11, 10],
            },
        },
        "vision_config": {
            "depth": 27, "hidden_size": 1152,
            "hidden_act": "gelu_pytorch_tanh", "intermediate_size": 4304,
            "num_heads": 16, "in_channels": 3, "patch_size": 16,
            "spatial_merge_size": 2, "temporal_patch_size": 2,
            "out_hidden_size": 5120, "num_position_embeddings": 2304,
        },
    }


@pytest.mark.parametrize("fmt", [None, "e4m3"])
def test_official_block_fp8_metadata_keeps_native_storage(fmt):
    quantization = {"quant_method": "fp8", "activation_scheme": "dynamic",
                    "weight_block_size": [128, 128]}
    if fmt is not None:
        quantization["fmt"] = fmt
    config = Qwen3_5Config.from_dict(_config(quantization)).text_config
    assert config.dense_weight_format == "fp8_e4m3"
    assert config.expert_weight_format == "fp8_e4m3"
    assert config.hidden_size == 5120


def test_unquantized_checkpoint_remains_bf16():
    config = Qwen3_5Config.from_dict(_config({})).text_config
    assert config.dense_weight_format == "bf16"


@pytest.mark.parametrize("quantization", [
    {"quant_method": "fp8", "fmt": "e5m2"},
    {"quant_method": "fp8", "fmt": None},
    {"quant_method": "fp8", "weight_block_size": [64, 128]},
])
def test_unsupported_fp8_metadata_does_not_select_bf16(quantization):
    with pytest.raises(ValueError, match="Qwen FP8 requires"):
        Qwen3_5Config.from_dict(_config(quantization))
