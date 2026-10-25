"""Installed API admission is checked before checkpoint construction or I/O."""
import importlib
from types import SimpleNamespace

import pytest
import torch

from kestrel.models.qwen35 import generated_verification as module


@pytest.mark.parametrize("arch,graphs,selector,expected", [
    ((10, 0), True, 0, {"stage_candidate"}),
    ((9, 0), False, 0, {"stage_candidate", "SegmentedVerification"}),
    ((9, 0), True, 1, {"stage_candidate", "SegmentedVerification"}),
    ((9, 0), True, 0, {"stage_candidate", "SegmentedVerification", "argmax_with_confidence"}),
])
def test_selected_generated_reports_all_missing_apis(tmp_path, monkeypatch, arch, graphs, selector, expected):
    (tmp_path / "config.json").write_text("{}")
    runtime = SimpleNamespace(_cfg=SimpleNamespace(verification_path="generated", page_size=1,
        enable_cuda_graphs=graphs), decode_path="auto", max_batch_size=1,
        device=torch.device("cuda"), _draft_model_path=tmp_path)
    from kestrel.models.qwen35.dflash import DFlashConfig
    monkeypatch.setattr(DFlashConfig, "from_dict", lambda _: SimpleNamespace(selector_rank=selector))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: arch)
    def missing(name):
        raise ModuleNotFoundError(name)
    monkeypatch.setattr(importlib, "import_module", missing)
    with pytest.raises(RuntimeError, match="backend fallback is disabled") as error:
        module.validate_verification_capabilities(runtime, object())
    for symbol in expected:
        assert symbol in str(error.value)
    assert ("argmax_with_confidence" in str(error.value)) == ("argmax_with_confidence" in expected)


def test_explicit_native_does_not_probe_generated_modules():
    module.validate_verification_capabilities(SimpleNamespace(
        _cfg=SimpleNamespace(verification_path="native")), object())


@pytest.mark.parametrize("supported", [False, True])
def test_automatic_selection_and_callable_contract(tmp_path, monkeypatch, supported):
    (tmp_path / "config.json").write_text("{}")
    runtime = SimpleNamespace(_cfg=SimpleNamespace(verification_path="auto", page_size=1,
        enable_cuda_graphs=True), decode_path="auto", max_batch_size=1,
        device=torch.device("cuda"), _draft_model_path=tmp_path)
    from kestrel.models.qwen35.dflash import DFlashConfig
    monkeypatch.setattr(DFlashConfig, "from_dict", lambda _: SimpleNamespace(selector_rank=0))
    monkeypatch.setattr(module.Qwen35GeneratedVerification, "supports", lambda *a: supported)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (10, 0))
    calls = []
    monkeypatch.setattr(importlib, "import_module", lambda name: calls.append(name) or SimpleNamespace(stage_candidate=None))
    if supported:
        with pytest.raises(RuntimeError, match="not callable"):
            module.validate_verification_capabilities(runtime, object())
        monkeypatch.setattr(importlib, "import_module", lambda name: SimpleNamespace(stage_candidate=lambda: None))
        module.validate_verification_capabilities(runtime, object())
    else:
        module.validate_verification_capabilities(runtime, object())
        assert not calls


def test_loader_validates_before_model_construction_or_weight_resolution(tmp_path, monkeypatch):
    from kestrel.models.qwen35 import qwen_loader
    config_file = tmp_path / "config.json"
    config_file.write_text("{}")
    config = object()
    def resolve(source, filename, **kwargs):
        assert filename == "config.json", "weight resolution preceded capability validation"
        return config_file
    monkeypatch.setattr(qwen_loader, "_resolve_checkpoint_file", resolve)
    monkeypatch.setattr(qwen_loader.Qwen3_5Config, "from_dict", lambda _: config)
    monkeypatch.setattr(qwen_loader, "Qwen3_5ForConditionalGeneration", lambda *a: pytest.fail("model constructed"))
    def validate(value):
        assert value is config
        raise RuntimeError("missing native capability")
    with pytest.raises(RuntimeError, match="missing native capability"):
        qwen_loader.load_qwen35_model(tmp_path, device=torch.device("cpu"),
            dtype=torch.bfloat16, validate_config=validate)


@pytest.mark.parametrize("capacity", [1, 8])
def test_real_dense_27b_hopper_auto_keeps_native_backend(tmp_path, monkeypatch, capacity):
    from kestrel.models.qwen35.qwen_config import Qwen3_5TextConfig
    from kestrel.models.qwen35.dflash import DFlashConfig
    target = Qwen3_5TextConfig(vocab_size=248320, hidden_size=5120,
        num_hidden_layers=64, intermediate_size=17408, num_attention_heads=24,
        num_key_value_heads=4, max_position_embeddings=262144, rms_norm_eps=1e-6,
        tie_word_embeddings=False, rope_theta=10000000, partial_rotary_factor=.25,
        mrope_section=(11, 11, 10), head_dim=256, linear_conv_kernel_dim=4,
        linear_key_head_dim=128, linear_value_head_dim=128,
        linear_num_key_heads=16, linear_num_value_heads=48,
        layer_types=("linear_attention", "linear_attention", "linear_attention", "full_attention") * 16)
    (tmp_path / "config.json").write_text("{}")
    draft = SimpleNamespace(selector_rank=0, block_size=8, target_layer_ids=(5, 19, 33, 47, 61))
    monkeypatch.setattr(DFlashConfig, "from_dict", lambda _: draft)
    runtime = SimpleNamespace(_cfg=SimpleNamespace(verification_path="auto", page_size=1,
        enable_cuda_graphs=True), decode_path="auto", max_batch_size=capacity,
        device=torch.device("cuda"), _draft_model_path=tmp_path)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (9, 0))
    monkeypatch.setattr(importlib, "import_module", lambda name: pytest.fail("native27 probed generated API"))
    module.validate_verification_capabilities(runtime, target)


def test_auto_decode_generated_requires_capabilities(tmp_path, monkeypatch):
    from kestrel.models.qwen35.dflash import DFlashConfig
    (tmp_path / "config.json").write_text("{}")
    monkeypatch.setattr(DFlashConfig, "from_dict", lambda _: SimpleNamespace(selector_rank=0))
    runtime = SimpleNamespace(_cfg=SimpleNamespace(verification_path="auto", page_size=1,
        enable_cuda_graphs=False), decode_path="generated", max_batch_size=1,
        device=torch.device("cuda"), _draft_model_path=tmp_path)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (10, 0))
    monkeypatch.setattr(module.Qwen35GeneratedVerification, "supports", lambda *a: pytest.fail("explicit selection probed auto support"))
    monkeypatch.setattr(importlib, "import_module", lambda name: SimpleNamespace())
    with pytest.raises(RuntimeError, match="stage_candidate"):
        module.validate_verification_capabilities(runtime, object())
