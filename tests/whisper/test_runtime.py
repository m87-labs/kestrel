from __future__ import annotations

import ast
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from safetensors.torch import save_file

from kestrel.kv_cache import KVMemoryPool
from kestrel.runtime.tokens import TextToken
from kestrel.models.whisper import MODEL_NAME, _runtime_factory
from kestrel.models.whisper.audio import AudioSource, PreparedAudio
from kestrel.models.whisper.config import WhisperPreprocessorConfig, WhisperTurboConfig
from kestrel.models.whisper.runtime import WhisperRuntime, WhisperRuntimeComponents
from kestrel.models.whisper.runtime_abi import WhisperExecutionBindings
from kestrel.models.whisper.skill import (
    WhisperDecodeContext,
    WhisperTranscribeSkill,
    WhisperTranscribeState,
)
from kestrel.models.whisper.tokenizer import WhisperControlTokens, WhisperTokenizer
from kestrel.models.whisper.weights import (
    expected_whisper_checkpoint_shapes,
    load_whisper_safetensors,
)


def test_native_whisper_ops_use_the_uniform_kernel_runtime_surface() -> None:
    source_root = Path(__file__).parents[2] / "kestrel" / "models" / "whisper"
    forbidden_modules = (
        "kestrel_kernels.flash_attn",
        "kestrel_kernels.fused_linear_residual_ops",
        "kestrel_kernels.fused_mlp_ops",
        "kestrel_kernels.gelu_residual",
        "kestrel_kernels.kv_cache_write_ops",
        "kestrel_kernels.layernorm_cuda_ops",
        "kestrel_kernels.linear_ops",
        "kestrel_kernels.sampling",
    )
    for name in (
        "prefill_stem.py",
        "prefill_encoder.py",
        "prefill_decoder_prefix.py",
        "runtime.py",
    ):
        tree = ast.parse((source_root / name).read_text(encoding="utf-8"))
        imported_modules = [
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module is not None
        ]
        imported_modules.extend(
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        assert not {
            module
            for module in imported_modules
            if any(
                module == forbidden or module.startswith(f"{forbidden}.")
                for forbidden in forbidden_modules
            )
        }, name


class _Encoding:
    ids = [10]


class _TokenizerBackend:
    def get_vocab_size(self, *, with_added_tokens):
        return 51866

    def encode(self, text, *, add_special_tokens):
        return _Encoding()

    def decode(self, ids, *, skip_special_tokens):
        return ""


@dataclass
class _FakeSession:
    kind: str
    buffers: tuple[Any, ...]
    warmup_failures: int = 0
    warmup_calls: int = 0
    launches: list[tuple[int, int]] = field(default_factory=list)
    shutdown_calls: int = 0
    artifact_identities: tuple[dict[str, object], ...] = ()
    artifact_receipts: tuple[dict[str, object], ...] = ()

    def warmup(self) -> None:
        self.warmup_calls += 1
        if self.warmup_failures:
            self.warmup_failures -= 1
            raise RuntimeError(f"{self.kind} warmup failed")

    def launch(self, slot_id: int, batch_size: int) -> None:
        self.launches.append((slot_id, batch_size))
        slot = self.buffers[slot_id]
        slot.logits_out[:batch_size].fill_(7.0)

    def run(self, slot: Any, batch_size: int) -> None:
        self.warmup_calls += 1
        if self.warmup_failures:
            self.warmup_failures -= 1
            raise RuntimeError(f"{self.kind} warmup failed")
        self.launches.append((int(slot.slot_id), int(batch_size)))
        slot.logits[:batch_size].fill_(11.0)

    def shutdown(self) -> None:
        self.shutdown_calls += 1


class _FakeSessionFactory:
    def __init__(self, *, decode_warmup_failures: int = 0) -> None:
        self.decode_warmup_failures = decode_warmup_failures
        self.bindings: WhisperExecutionBindings | None = None
        self.prefill: _FakeSession | None = None
        self.decode: _FakeSession | None = None

    def create_prefill(self, bindings: WhisperExecutionBindings) -> _FakeSession:
        self.bindings = bindings
        self.prefill = _FakeSession("prefill", bindings.prefill_buffers)
        return self.prefill

    def create_decode(self, bindings: WhisperExecutionBindings) -> _FakeSession:
        assert self.bindings is bindings
        self.decode = _FakeSession(
            "decode",
            bindings.decode_buffers,
            warmup_failures=self.decode_warmup_failures,
        )
        return self.decode

    def native_provenance(
        self,
        bindings: WhisperExecutionBindings,
        decode_session: _FakeSession,
    ) -> dict[str, Any]:
        assert self.bindings is bindings
        assert self.decode is decode_session
        return {}


class _FailingDecodeFactory(_FakeSessionFactory):
    def create_decode(self, bindings: WhisperExecutionBindings) -> _FakeSession:
        assert self.bindings is bindings
        raise RuntimeError("generated decode construction failed")


@pytest.fixture(scope="module")
def runtime_model_config() -> WhisperTurboConfig:
    # Runtime tests retain production audio/vocabulary geometry while shrinking
    # the hidden stack and target length enough to keep CPU allocations cheap.
    return WhisperTurboConfig(
        d_model=8,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=16,
        decoder_ffn_dim=16,
        max_source_positions=1500,
        max_target_positions=16,
        num_mel_bins=128,
    )


@pytest.fixture(scope="module")
def runtime_weights(runtime_model_config, tmp_path_factory):
    tensors = {
        name: torch.zeros(shape, dtype=torch.float32)
        for name, shape in expected_whisper_checkpoint_shapes(
            runtime_model_config
        ).items()
    }
    path = tmp_path_factory.mktemp("whisper-runtime") / "model.safetensors"
    save_file(tensors, path, metadata={"format": "pt"})
    return load_whisper_safetensors(
        path,
        runtime_model_config,
        checkpoint_dtype=torch.float32,
    ).to(dtype=torch.bfloat16)


@pytest.fixture
def prepared_audio() -> PreparedAudio:
    return PreparedAudio(
        input_features=torch.zeros((128, 3000), dtype=torch.float32),
        duration_seconds=0.1,
        original_num_samples=1600,
        original_sample_rate=16000,
        resampled_num_samples=1600,
    )


def _components(config, weights, factory) -> WhisperRuntimeComponents:
    return WhisperRuntimeComponents(
        config=config,
        preprocessor_config=WhisperPreprocessorConfig(),
        tokenizer=WhisperTokenizer(
            _TokenizerBackend(),
            WhisperControlTokens(suppress_tokens=()),
        ),
        weights=weights,
        session_factory=factory,
    )


def _make_runtime(
    config,
    weights,
    factory,
    *,
    max_batch_size: int = 2,
    kv_cache_pages: int = 64,
    compute_stream=None,
) -> WhisperRuntime:
    cfg = SimpleNamespace(
        model=MODEL_NAME,
        device="cpu",
        dtype=torch.bfloat16,
        max_batch_size=max_batch_size,
        page_size=1,
        kv_cache_pages=kv_cache_pages,
        enable_prefix_cache=False,
    )
    pool = KVMemoryPool(device="cpu")
    return WhisperRuntime(
        cfg,
        kv_pool=pool,
        compute_stream=compute_stream,
        _components=_components(config, weights, factory),
    )


def test_runtime_factory_warms_before_return(monkeypatch) -> None:
    calls: list[object] = []

    class FakeRuntime:
        def __init__(self, cfg, **kwargs):
            calls.append(("construct", cfg, kwargs))

        def warmup(self) -> None:
            calls.append("warmup")

        def shutdown(self) -> None:
            calls.append("shutdown")

    monkeypatch.setattr("kestrel.models.whisper.runtime.WhisperRuntime", FakeRuntime)
    cfg = object()
    runtime = _runtime_factory(cfg, marker="value")

    assert isinstance(runtime, FakeRuntime)
    assert calls == [("construct", cfg, {"marker": "value"}), "warmup"]


def test_runtime_factory_shuts_down_after_warmup_failure(monkeypatch) -> None:
    calls: list[str] = []

    class FakeRuntime:
        def __init__(self, cfg, **kwargs):
            del cfg, kwargs

        def warmup(self) -> None:
            calls.append("warmup")
            raise RuntimeError("warmup failed")

        def shutdown(self) -> None:
            calls.append("shutdown")

        def _abort_failed_warmup(self) -> BaseException | None:
            self.shutdown()
            self.__dict__.clear()
            return None

    monkeypatch.setattr("kestrel.models.whisper.runtime.WhisperRuntime", FakeRuntime)

    with pytest.raises(RuntimeError, match="warmup failed"):
        _runtime_factory(object())
    assert calls == ["warmup", "shutdown"]


@pytest.mark.parametrize("field", ("model_path", "tokenizer_path"))
def test_production_runtime_rejects_unverified_local_checkpoint_overrides(
    monkeypatch,
    tmp_path,
    field: str,
) -> None:
    import kestrel.models.whisper.runtime as runtime_module

    monkeypatch.setattr(runtime_module, "_CUTE_JIT_ENABLED_AT_RUNTIME_IMPORT", False)
    monkeypatch.setattr(runtime_module, "is_cute_jit_enabled", lambda: False)
    cfg = SimpleNamespace(model=MODEL_NAME, **{field: tmp_path})

    with pytest.raises(ValueError, match="does not accept local checkpoint overrides"):
        runtime_module._load_production_components(
            cfg,
            device=torch.device("cuda"),
            dtype=torch.bfloat16,
        )


def _explicit_prefix() -> list[TextToken]:
    return [
        TextToken(token_id=50258),
        TextToken(token_id=50259),
        TextToken(token_id=50360),
    ]


def test_prepared_audio_cross_row_and_prefix_slot_lifecycle(
    runtime_model_config,
    runtime_weights,
    prepared_audio,
) -> None:
    factory = _FakeSessionFactory()
    runtime = _make_runtime(runtime_model_config, runtime_weights, factory)
    cross_key_ptr = runtime.cross_kv.keys.data_ptr()
    prepared = runtime.prepare_sequence(
        _explicit_prefix(),
        encoder_input=prepared_audio,
        max_new_tokens=4,
    )
    row = prepared.state.batch_idx
    assert row in runtime._owned_cross_rows
    assert runtime._prepared_audio[row] is prepared_audio

    slot = runtime.acquire_prefill_slot()
    logits = runtime.launch_prepared_batch(
        [prepared],
        slot,
        images=[None],
        image_crops_list=[None],
        encoder_inputs=[prepared_audio],
    )
    assert factory.prefill is not None
    assert factory.prefill.launches == [(slot.slot_id, 1)]
    assert torch.all(logits == 7)
    assert slot.metadata.control_token_ids.gpu[0].tolist() == [
        50258,
        50259,
        50360,
        0,
    ]
    assert slot.metadata.prefix_lengths.gpu[0].item() == 3
    assert slot.batch_idx[0].item() == row
    assert (
        slot.metadata.slot_mapping.gpu[0, :3].tolist()
        == (runtime.page_table.page_table_cpu[row][:3])
    )
    assert runtime.cross_kv.keys.data_ptr() == cross_key_ptr

    runtime.finalize_prepared_sequence_after_prefill(prepared)
    assert row in runtime.active_sequences
    assert row not in runtime._prepared_audio
    assert row in runtime._owned_cross_rows

    runtime.release_sequence(prepared.state)
    assert row not in runtime.active_sequences
    assert row not in runtime._owned_cross_rows
    assert row in runtime.page_table.free_batch_idx
    runtime.release_prefill_slot(slot)
    runtime.shutdown()


def test_prefill_rejects_replaced_audio_and_abort_releases_every_owner(
    runtime_model_config,
    runtime_weights,
    prepared_audio,
) -> None:
    factory = _FakeSessionFactory()
    runtime = _make_runtime(runtime_model_config, runtime_weights, factory)
    prepared = runtime.prepare_sequence(
        _explicit_prefix(),
        encoder_input=prepared_audio,
        max_new_tokens=2,
    )
    replacement = PreparedAudio(
        input_features=prepared_audio.input_features.clone(),
        duration_seconds=0.1,
        original_num_samples=1600,
        original_sample_rate=16000,
        resampled_num_samples=1600,
    )
    slot = runtime.acquire_prefill_slot()
    with pytest.raises(RuntimeError, match="ownership mismatch"):
        runtime.launch_prepared_batch([prepared], slot, encoder_inputs=[replacement])
    runtime.abort_prepared_sequence(prepared)
    assert prepared.state.batch_idx in runtime.page_table.free_batch_idx
    assert prepared.state.batch_idx not in runtime._owned_cross_rows
    assert prepared.state.batch_idx not in runtime._prepared_audio
    runtime.release_prefill_slot(slot)
    runtime.shutdown()


def test_generated_decode_pads_to_compiled_capacity_with_noop_row(
    runtime_model_config,
    runtime_weights,
) -> None:
    factory = _FakeSessionFactory()
    runtime = _make_runtime(
        runtime_model_config,
        runtime_weights,
        factory,
        max_batch_size=3,
    )
    slot = runtime.decode_slots[0]
    slot.decode_token_ids[:3] = torch.tensor([7, 8, 9])
    slot.meta.input_pos.cpu[:3] = torch.tensor([2, 3, 4], dtype=torch.int32)
    slot.meta.batch_idx.cpu[:3] = torch.tensor([1, 2, 3])
    slot.meta.inputs.copy_to_gpu()

    runtime.decode_with_slot(slot, 3)

    assert factory.decode is not None
    assert factory.decode.launches == [(0, 4)]
    assert slot.decode_token_ids[:4].tolist() == [7, 8, 9, 0]
    assert slot.meta.input_pos.gpu[:4].tolist() == [2, 3, 4, 0]
    assert slot.meta.batch_idx.gpu[:4].tolist() == [1, 2, 3, 0]
    assert torch.all(slot.logits[:4] == 11)
    runtime.shutdown()


def test_timestamp_grammar_stages_generic_batch_aligned_constraints(
    runtime_model_config,
    runtime_weights,
    prepared_audio,
) -> None:
    factory = _FakeSessionFactory()
    runtime = _make_runtime(runtime_model_config, runtime_weights, factory)
    skill = WhisperTranscribeSkill()
    explicit = WhisperTranscribeState(
        skill,
        SimpleNamespace(),
        WhisperDecodeContext(
            language="en",
            timestamps="segment",
            max_transcript_tokens=4,
            temperature=0.0,
        ),
        runtime.tokenizer,
        prepared_audio,
    )
    automatic = WhisperTranscribeState(
        skill,
        SimpleNamespace(),
        WhisperDecodeContext(
            language=None,
            timestamps="segment",
            max_transcript_tokens=4,
            temperature=0.0,
        ),
        runtime.tokenizer,
        prepared_audio,
    )
    hook = runtime.sampling_hooks.process_logits
    assert hook is not None
    logits = torch.zeros((1, runtime.vocab_size), dtype=torch.bfloat16)
    logits[0, 42] = 3
    batch_idx = runtime.prefill_slots[0].batch_idx[:1]
    batch_idx.fill_(1)
    returned = hook(
        logits,
        sequences=[SimpleNamespace(skill_state=explicit)],
        batch_idx=batch_idx,
    )
    plan = runtime._logits_constraints_by_batch_idx_ptr[int(batch_idx.data_ptr())].gpu[
        :1
    ]
    expected = explicit.timestamp_plan()
    assert expected is not None
    assert returned is None
    assert plan.dtype is torch.int32
    assert plan.shape == (1, 8)
    assert plan[0, 0].item() == 3
    assert plan[0, 1].item() == 50365
    encoded_ranges = [
        tuple(plan[0, 2 + 2 * index : 4 + 2 * index].tolist())
        for index in range(3)
        if plan[0, 2 + 2 * index].item() < plan[0, 3 + 2 * index].item()
    ]
    assert encoded_ranges == list(expected.suppress_ranges)
    logits.zero_()
    logits[0, 17] = 5
    unchanged = logits.clone()
    hook(
        logits,
        sequences=[SimpleNamespace(skill_state=automatic)],
        batch_idx=batch_idx,
    )
    assert plan[0].tolist() == [0] * 8
    assert torch.equal(logits, unchanged)

    explicit.controls = replace(explicit.controls, timestamp_begin_id=0)
    with pytest.raises(RuntimeError, match="invalid partition split"):
        hook(
            logits,
            sequences=[SimpleNamespace(skill_state=explicit)],
            batch_idx=batch_idx,
        )
    runtime.shutdown()


def test_sampled_token_scores_use_untempered_masked_model_distribution(
    runtime_model_config,
    runtime_weights,
) -> None:
    runtime = _make_runtime(
        runtime_model_config, runtime_weights, _FakeSessionFactory()
    )
    hook = runtime.sampling_hooks.score_sampled_tokens
    assert hook is not None
    logits = torch.tensor(
        [[0.0, 1.0, 2.0], [2.0, -float("inf"), 0.0]],
        dtype=torch.bfloat16,
    )
    sampled_ids = torch.tensor([2, 0], dtype=torch.int64)
    actual = torch.zeros(2, dtype=torch.float32)

    returned = hook(
        logits,
        sampled_ids=sampled_ids,
        token_logprobs=actual,
        sequences=[object(), object()],
        batch_idx=torch.tensor([3, 5], dtype=torch.int64),
        temperatures=torch.tensor([0.0, 0.8]),
        top_ps=torch.tensor([1.0, 0.9]),
    )

    expected = (
        torch.log_softmax(logits.float(), dim=1)
        .gather(1, sampled_ids.unsqueeze(1))
        .squeeze(1)
    )
    assert returned is None
    torch.testing.assert_close(actual, expected)
    with pytest.raises(ValueError, match="invalid ABI"):
        hook(
            logits,
            sampled_ids=sampled_ids.to(torch.int32),
            token_logprobs=actual,
            sequences=[object(), object()],
            batch_idx=torch.tensor([3, 5], dtype=torch.int64),
            temperatures=torch.tensor([0.0, 0.8]),
            top_ps=torch.tensor([1.0, 0.9]),
        )
    runtime.shutdown()


def test_logits_constraint_staging_is_owned_by_each_pipeline_slot(
    runtime_model_config,
    runtime_weights,
    prepared_audio,
) -> None:
    runtime = _make_runtime(
        runtime_model_config, runtime_weights, _FakeSessionFactory()
    )
    skill = WhisperTranscribeSkill()
    explicit = WhisperTranscribeState(
        skill,
        SimpleNamespace(),
        WhisperDecodeContext(
            language="en",
            timestamps="segment",
            max_transcript_tokens=4,
            temperature=0.0,
        ),
        runtime.tokenizer,
        prepared_audio,
    )
    automatic = WhisperTranscribeState(
        skill,
        SimpleNamespace(),
        WhisperDecodeContext(
            language=None,
            timestamps="segment",
            max_transcript_tokens=4,
            temperature=0.0,
        ),
        runtime.tokenizer,
        prepared_audio,
    )
    resident_batch_idx = [
        runtime.prefill_slots[0].batch_idx[:1],
        runtime.prefill_slots[1].batch_idx[:1],
        runtime.decode_slots[0].meta.batch_idx.gpu[:1],
        runtime.decode_slots[1].meta.batch_idx.gpu[:1],
    ]
    for batch_idx in resident_batch_idx:
        batch_idx.fill_(1)
    buffers = [
        runtime._logits_constraints_by_batch_idx_ptr[int(batch_idx.data_ptr())]
        for batch_idx in resident_batch_idx
    ]
    assert len({id(buffer) for buffer in buffers}) == 4
    assert len({buffer.cpu.data_ptr() for buffer in buffers}) == 4
    assert len({buffer.gpu.data_ptr() for buffer in buffers}) == 4

    first = runtime._stage_logits_constraints(
        sequences=[SimpleNamespace(skill_state=explicit)],
        batch_idx=resident_batch_idx[0],
    )
    expected_first_cpu = buffers[0].cpu.clone()
    expected_first = first.clone()
    second = runtime._stage_logits_constraints(
        sequences=[SimpleNamespace(skill_state=automatic)],
        batch_idx=resident_batch_idx[1],
    )
    assert second[0].tolist() == [0] * 8
    # A second pipeline slot may stage immediately after the first H2D enqueue;
    # its CPU writes must not mutate the first slot's source or destination.
    assert torch.equal(buffers[0].cpu, expected_first_cpu)
    assert torch.equal(first, expected_first)

    runtime._stage_logits_constraints(
        sequences=[SimpleNamespace(skill_state=explicit)],
        batch_idx=resident_batch_idx[2],
    )
    runtime._stage_logits_constraints(
        sequences=[SimpleNamespace(skill_state=automatic)],
        batch_idx=resident_batch_idx[3],
    )
    assert torch.equal(buffers[0].cpu, expected_first_cpu)
    assert torch.equal(first, expected_first)

    with pytest.raises(RuntimeError, match="must alias a resident"):
        runtime._stage_logits_constraints(
            sequences=[SimpleNamespace(skill_state=explicit)],
            batch_idx=torch.tensor([1], dtype=torch.int64),
        )
    runtime.shutdown()


def test_warmup_retries_only_the_backend_that_failed(
    runtime_model_config,
    runtime_weights,
) -> None:
    factory = _FakeSessionFactory(decode_warmup_failures=1)
    runtime = _make_runtime(runtime_model_config, runtime_weights, factory)
    with pytest.raises(RuntimeError, match="decode warmup failed"):
        runtime.warmup()
    runtime.warmup()
    runtime.warmup()
    assert factory.prefill is not None and factory.prefill.warmup_calls == 1
    assert factory.decode is not None and factory.decode.warmup_calls == 3
    runtime.shutdown()
    runtime.shutdown()
    assert factory.prefill.shutdown_calls == 1
    assert factory.decode.shutdown_calls == 0


def _constructor_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        model=MODEL_NAME,
        device="cpu",
        dtype=torch.bfloat16,
        max_batch_size=1,
        page_size=1,
        kv_cache_pages=32,
        enable_prefix_cache=False,
    )


def test_async_preprocessor_accepts_only_validated_audio_source(
    runtime_model_config,
    runtime_weights,
) -> None:
    factory = _FakeSessionFactory()
    runtime = _make_runtime(runtime_model_config, runtime_weights, factory)
    source = AudioSource(
        value=np.zeros(1600, dtype=np.float32),
        kind="pcm",
        sample_rate=16000,
    )
    result = runtime.preprocess_encoder_input_async(source).result(timeout=10)
    assert isinstance(result, PreparedAudio)
    with pytest.raises(TypeError, match="validated AudioSource"):
        runtime.preprocess_encoder_input_async(source.value)
    runtime.shutdown()


def test_packed_receipt_validation_requires_exact_component_families() -> None:
    from kestrel.models.whisper.runtime import _validated_packed_receipts

    def receipt(family: str) -> dict[str, object]:
        return {
            "schema_version": 1,
            "family": family,
            "variant_key": f"{family}_variant",
            "architecture": "sm90",
            "payload_kind": "cubin",
            "payload_sha256": "12" * 32,
            "archive_path": "/opt/kestrel/kestrel_kernels/bundles.kstlc",
            "archive_size_bytes": 947_123,
            "archive_sha256": "34" * 32,
            "archive_root": "cu13",
        }

    receipts = tuple(receipt(family) for family in ("flash_attn", "gelu", "gelu_add"))
    assert {
        item["family"]
        for item in _validated_packed_receipts(
            receipts,
            expected_families={"flash_attn", "gelu", "gelu_add"},
            component="prefill",
        )
    } == {"flash_attn", "gelu", "gelu_add"}

    with pytest.raises(RuntimeError, match="expected"):
        _validated_packed_receipts(
            receipts[:-1],
            expected_families={"flash_attn", "gelu", "gelu_add"},
            component="prefill",
        )
