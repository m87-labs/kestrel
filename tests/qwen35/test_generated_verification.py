"""AOT verification session coverage; model paths are supplied on GPU hosts."""

import asyncio
import os
from itertools import groupby
from types import SimpleNamespace

import pytest
import torch

from kestrel.config import RuntimeConfig
from kestrel.engine import InferenceEngine
from kestrel.models.qwen35.generated_verification import Qwen35GeneratedVerification


def _verification_runtime(*, capacity=1, page_size=1, weight_format="fp8_e4m3"):
    target = SimpleNamespace(hidden_size=5120, num_hidden_layers=64,
                             intermediate_size=17408, vocab_size=248320,
                             dense_weight_format=weight_format)
    return SimpleNamespace(
        max_batch_size=capacity, page_size=page_size, device=torch.device("cuda"),
        dtype=torch.bfloat16, decode_path="auto", _cfg=SimpleNamespace(enable_cuda_graphs=False),
        model=SimpleNamespace(model=SimpleNamespace(language_model=SimpleNamespace(config=target))))


def _verification_draft(block=16):
    taps = (1, 10, 18, 27, 35, 44, 52, 61) if block == 16 else (5, 19, 33, 47, 61)
    return SimpleNamespace(config=SimpleNamespace(
        block_size=block, target_layer_ids=taps, hidden_size=5120,
        mask_token_id=1, selector_rank=0))


@pytest.mark.parametrize("capacity,block,page,arch,weight_format,expected", [
    (1, 16, 1, (10, 0), "fp8_e4m3", True),
    (1, 8, 1, (10, 0), "fp8_e4m3", True),
    (2, 8, 1, (10, 0), "fp8_e4m3", True),
    (2, 16, 1, (10, 0), "fp8_e4m3", False),
    (4, 16, 1, (10, 0), "fp8_e4m3", False),
    (1, 16, 64, (10, 0), "fp8_e4m3", False),
    (1, 16, 1, (9, 0), "fp8_e4m3", False),
    (1, 16, 1, (10, 0), "bf16", False),
])
def test_generated_verification_auto_coverage(monkeypatch, capacity, block, page, arch,
                                            weight_format, expected):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: arch)
    runtime = _verification_runtime(capacity=capacity, page_size=page, weight_format=weight_format)
    assert Qwen35GeneratedVerification.supports(runtime, _verification_draft(block)) is expected


def test_generated_verification_auto_rejects_other_target_and_taps(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (10, 0))
    runtime, draft = _verification_runtime(), _verification_draft()
    runtime.model.model.language_model.config.hidden_size = 2048
    assert not Qwen35GeneratedVerification.supports(runtime, draft)
    runtime.model.model.language_model.config.hidden_size = 5120
    draft.config.target_layer_ids = (1, 2, 3)
    assert not Qwen35GeneratedVerification.supports(runtime, draft)


@pytest.mark.parametrize("capacity,block,expected_rows", [(1, 16, [1]), (2, 8, [1, 2]), (2, 16, [])])
def test_dflash_auto_constructs_supported_generated_verifiers(monkeypatch, capacity, block, expected_rows):
    from kestrel.models.qwen35 import spec_decoder

    runtime, draft = _verification_runtime(capacity=capacity), _verification_draft(block)
    monkeypatch.setattr(spec_decoder, "load_dflash_drafter", lambda *a, **k: draft)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (10, 0))
    monkeypatch.setattr(torch.cuda, "Stream", lambda **kwargs: object())
    monkeypatch.setattr(torch.cuda, "Event", lambda **kwargs: object())
    rows = []
    def init(self, runtime, draft, *, sequences=1, weights=None):
        rows.append(sequences)
        self.weights = object()
        self.segmented = False
    monkeypatch.setattr(Qwen35GeneratedVerification, "__init__", init)
    decoder = spec_decoder.Qwen35DFlashDecoder(runtime, "draft")
    assert rows == expected_rows
    assert (decoder._generated_verification is not None) == bool(expected_rows)


def test_dflash_auto_does_not_hide_generated_binding_errors(monkeypatch):
    from kestrel.models.qwen35 import spec_decoder

    monkeypatch.setattr(spec_decoder, "load_dflash_drafter", lambda *a, **k: _verification_draft())
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (10, 0))
    monkeypatch.setattr(torch.cuda, "Stream", lambda **kwargs: object())
    monkeypatch.setattr(torch.cuda, "Event", lambda **kwargs: object())
    def init(self, *args, **kwargs):
        raise RuntimeError("missing AOT Qwen generated verification program")
    monkeypatch.setattr(Qwen35GeneratedVerification, "__init__", init)
    with pytest.raises(RuntimeError, match="missing AOT"):
        spec_decoder.Qwen35DFlashDecoder(_verification_runtime(), "draft")


def test_generated_verification_rejects_unqualified_concurrency():
    with pytest.raises(ValueError, match="up to eight sequences"):
        Qwen35GeneratedVerification(SimpleNamespace(max_batch_size=9), None)


@pytest.mark.parametrize("count", [0, 17, True, -1])
def test_generated_verification_rejects_invalid_commit(count):
    target = Qwen35GeneratedVerification.__new__(Qwen35GeneratedVerification)
    target.pending, target.block, target.verified = True, 16, object()
    with pytest.raises(ValueError, match="outstanding verification"):
        target.commit((None, target.verified, None, None, count))


def test_generated_verification_rejects_overlapping_blocks():
    target = Qwen35GeneratedVerification.__new__(Qwen35GeneratedVerification)
    target.pending = True
    with pytest.raises(RuntimeError, match="uncommitted block"):
        target.target([], None, 0)


@pytest.mark.parametrize("shape,dtype", [((16,), torch.int32), ((1, 15), torch.int32),
                                       ((1, 16), torch.float32)])
def test_generated_verification_rejects_invalid_device_candidates(shape, dtype):
    target = Qwen35GeneratedVerification.__new__(Qwen35GeneratedVerification)
    target.pending, target.sequences, target.block = False, 1, 16
    target.runtime = SimpleNamespace(device="cpu")
    with pytest.raises(ValueError, match="device-local integer token rows"):
        target.target_many(torch.zeros(shape, dtype=dtype), [None], [0])
    assert not target.pending


def test_generated_verification_commits_independent_rows_once(monkeypatch):
    target = Qwen35GeneratedVerification.__new__(Qwen35GeneratedVerification)
    target.pending, target.block, target.pending_rows = True, 8, {0, 1}
    target.verified = [object(), object()]
    target.sources = [SimpleNamespace(seq_length=10, layers=[]), SimpleNamespace(seq_length=20, layers=[])]
    target.start_positions = (10, 20)
    target.sequences = 2
    target.initial_states, target.initial_histories = [[], []], [[], []]
    target.accepted = [SimpleNamespace(), SimpleNamespace()]
    target.layers = ()
    replayed = []
    target.replays = [SimpleNamespace(launch=lambda states, histories, count: replayed.append((0, count))),
                      SimpleNamespace(launch=lambda states, histories, count: replayed.append((1, count)))]
    target.stream = object()
    target.runtime = SimpleNamespace(device="cuda")
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: target.stream)
    sessions = [SimpleNamespace(), SimpleNamespace()]
    features = torch.zeros(1, 8, 2)
    contexts = [(sessions[i], target.verified[i], features, list(range(8)), count)
                for i, count in enumerate((2, 4))]
    target.commit(contexts[1])
    assert target.pending and target.pending_rows == {0}
    assert sessions[1].cache.seq_length == 24 and sessions[1].bonus == 3
    with pytest.raises(ValueError, match="outstanding verification"):
        target.commit(contexts[1])
    target.commit(contexts[0])
    assert not target.pending
    assert sessions[0].cache.seq_length == 12 and sessions[0].bonus == 1
    assert replayed == [(1, 4), (0, 2)]


@pytest.mark.parametrize("device_candidates", [False, True])
def test_generated_verification_retains_sources_when_rows_swap(monkeypatch, device_candidates):
    target = Qwen35GeneratedVerification.__new__(Qwen35GeneratedVerification)
    target.pending, target.block, target.sequences = False, 8, 2
    target.invocation, target.stream = object(), object()
    target.layers = (0,)
    target.runtime = SimpleNamespace(
        device="cpu", page_table=SimpleNamespace(page_table=torch.zeros(2, 16)),
        _decode_rope_deltas=torch.zeros(2, 1))
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: target.stream)
    target.positions = torch.arange(8, dtype=torch.int32)
    target.inputs = dict(input_ids=torch.zeros(16, dtype=torch.int32),
                         input_pos=torch.zeros(16, dtype=torch.int32),
                         page_table=torch.zeros(2, 16), rope_delta_table=torch.zeros(2, 1),
                         verification_features=torch.zeros(16, 2))
    def cache(value, position):
        return SimpleNamespace(seq_length=position, layers=[SimpleNamespace(
            recurrent_states=torch.tensor([float(value)]), conv_states=torch.tensor([float(value)]))])
    target.accepted = [cache(10, 10), cache(20, 20)]
    target.verified = [cache(0, 0), cache(0, 0)]
    target.initial_states = [[torch.zeros(1)], [torch.zeros(1)]]
    target.initial_histories = [[torch.zeros(1)], [torch.zeros(1)]]
    target.state_destinations = {
        "gdn_recurrent_state": [c.layers[0].recurrent_states for c in target.verified],
        "gdn_conv_state": [c.layers[0].conv_states for c in target.verified]}
    target.launch = lambda: None
    def replay(row, states, histories, count):
        target.accepted[row].layers[0].recurrent_states.copy_(states[0] + count)
        target.accepted[row].layers[0].conv_states.copy_(histories[0] + count)
    target.replays = [SimpleNamespace(launch=lambda s, h, c: replay(0, s, h, c)),
                      SimpleNamespace(launch=lambda s, h, c: replay(1, s, h, c))]
    candidates = [list(range(8))] * 2
    if device_candidates:
        candidates = torch.tensor(candidates, dtype=torch.int32)
    results = target.target_many(candidates, list(reversed(target.accepted)), [1, 0])
    sessions = [SimpleNamespace(), SimpleNamespace()]
    for row, (expected, features, verified) in enumerate(results):
        target.commit((sessions[row], verified, features, expected, row + 2))
    assert sessions[0].cache.seq_length == 22
    assert sessions[1].cache.seq_length == 13
    assert sessions[0].cache.layers[0].recurrent_states.item() == 22
    assert sessions[1].cache.layers[0].recurrent_states.item() == 13
    assert not target.pending


def test_generated_verification_repeated_requests():
    target = os.environ.get("QWEN_VERIFICATION_TARGET")
    draft = os.environ.get("QWEN_VERIFICATION_DRAFT")
    if not target or not draft or not torch.cuda.is_available():
        pytest.skip("Qwen target/draft weights and B200 required")
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("B200 required")

    async def run():
        reference = None
        for path in ("auto", "generated"):
            engine = await InferenceEngine.create(RuntimeConfig(
                model="Qwen/Qwen3.5-27B-FP8", model_path=target, tokenizer_path=target,
                draft_model_path=draft, decode_path=path, max_batch_size=1,
                page_size=1, kv_cache_pages=4096, enable_prefix_cache=False))
            try:
                decoder = engine.runtime.spec.decoder
                assert decoder._generated_verification is not None
                for _ in range(3):
                    result = await engine.chat(
                        [{"role": "user", "content": "Explain why the sky is blue in two sentences."}],
                        reasoning=False, settings={"temperature": 0, "max_tokens": 32})
                    ids = [token.token_id for token in result.tokens]
                    assert ids
                    if reference is None:
                        reference = ids
                    assert ids == reference
                    assert decoder._generated_verification.pending is False
                result = await engine.chat(
                    [{"role": "user", "content": "Write a Python function to compute Fibonacci numbers."}],
                    reasoning=False, settings={"temperature": 0, "max_tokens": 32})
                assert result.tokens
            finally:
                await engine.shutdown()

    asyncio.run(run())


def test_generated_verification_long_reasoning_does_not_repeat_zeros():
    target = os.environ.get("QWEN_VERIFICATION_TARGET")
    draft = os.environ.get("QWEN_VERIFICATION_DRAFT")
    if not target or not draft or not torch.cuda.is_available():
        pytest.skip("Qwen target/draft weights and B200 required")
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("B200 required")

    async def run():
        engine = await InferenceEngine.create(RuntimeConfig(
            model="Qwen/Qwen3.5-27B-FP8", model_path=target, tokenizer_path=target,
            draft_model_path=draft, decode_path="generated", max_batch_size=1,
            page_size=1, kv_cache_pages=32768, enable_prefix_cache=False))
        try:
            for _ in range(2):
                result = await engine.chat(
                    [{"role": "user", "content":
                      "Explain how to derive the quadratic formula by completing the square. "
                      "Work through several examples and discuss numerical stability when implementing it."}],
                    reasoning=True, settings={"temperature": 0, "max_tokens": 16384})
                assert result.output.get("finish_reason") == "stop"
                ids = [token.token_id for token in result.tokens]
                assert ids
                longest_run = max(sum(1 for _ in values) for _, values in groupby(ids))
                assert longest_run < 64, "long reasoning collapsed into repeated tokens"
                assert not engine.runtime.spec.decoder._generated_verification.pending
        finally:
            await engine.shutdown()

    asyncio.run(run())


def _hopper_verification_runtime(capacity=1):
    runtime, draft = _verification_runtime(capacity=capacity, weight_format='bf16'), _verification_draft()
    target = runtime.model.model.language_model.config
    target.__dict__.update(
        hidden_size=2048, num_hidden_layers=40, num_attention_heads=16,
        num_key_value_heads=2, head_dim=256, num_experts=256, num_experts_per_tok=8,
        moe_intermediate_size=512, shared_expert_intermediate_size=512,
        linear_num_key_heads=16, linear_num_value_heads=32, linear_key_head_dim=128,
        linear_value_head_dim=128, expert_weight_format='bf16',
        linear_conv_kernel_dim=4, rms_norm_eps=1e-6, partial_rotary_factor=0.25,
        tie_word_embeddings=False, mrope_section=(11, 11, 10),
        layer_types=('linear_attention', 'linear_attention', 'linear_attention', 'full_attention') * 10)
    draft.config.target_layer_ids = (1, 6, 11, 16, 22, 27, 32, 37)
    return runtime, draft


@pytest.mark.parametrize('verification_rows', (4, 6, 8, 16))
@pytest.mark.parametrize('mutation', ['none', 'taps', 'capacity', 'experts', 'order', 'page'])
def test_hopper_verification_rejects_wrong_contract_before_materializing(monkeypatch, mutation, verification_rows):
    from kestrel.models.qwen35 import generated_verification as module

    runtime, draft = _hopper_verification_runtime()
    target = runtime.model.model.language_model.config
    if mutation == 'taps': draft.config.target_layer_ids = (0, 1)
    elif mutation == 'capacity': runtime.max_batch_size = 9
    elif mutation == 'experts': target.num_experts = 128
    elif mutation == 'order': target.layer_types = tuple(reversed(target.layer_types))
    elif mutation == 'page': runtime.page_size = 64
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda device: (9, 0))
    monkeypatch.setattr(torch.cuda, 'get_device_properties',
                        lambda device: SimpleNamespace(multi_processor_count=132))
    resolved = []
    def resolve(**kwargs):
        resolved.append(kwargs)
        return None
    monkeypatch.setattr(module, 'resolve_program', resolve)
    if mutation == 'none':
        with pytest.raises(RuntimeError, match='missing AOT'):
            Qwen35GeneratedVerification(runtime, draft, verification_rows=verification_rows)
        assert resolved[0]['registration'] == 'qwen35_moe_verification'
        assert resolved[0]['verification'] is True
        assert resolved[0]['program'].endswith(f'_t{verification_rows}')
        # Disabling graphs retains the ordinary path; only the exact validated
        # Hopper contract is admitted automatically with graphs enabled.
        assert not Qwen35GeneratedVerification.supports(runtime, draft)
        runtime._cfg.enable_cuda_graphs = True
        assert Qwen35GeneratedVerification.supports(runtime, draft)
    else:
        with pytest.raises(ValueError):
            Qwen35GeneratedVerification(runtime, draft, verification_rows=verification_rows)
        assert not resolved


@pytest.mark.parametrize('slot,start', [(-1, 0), (1, 0), (0, -1), (0, 32760)])
def test_segmented_verification_rejects_kv_overflow_before_prepare(slot, start):
    target = Qwen35GeneratedVerification.__new__(Qwen35GeneratedVerification)
    target.pending, target.block, target.sequences = False, 16, 1
    target.segmented, target.invocation = True, None
    target.runtime = SimpleNamespace(page_table=SimpleNamespace(page_table=torch.empty(1, 32768)))
    target._prepare = lambda *_: pytest.fail('invalid request reached GPU preparation')
    with pytest.raises(ValueError, match='page table'):
        target.target_many([list(range(16))], [SimpleNamespace(seq_length=start)], [slot])


def test_automatic_loading_does_not_allocate_for_unqualified_target(tmp_path, monkeypatch):
    from kestrel.models.qwen35 import generated_verification as module
    from kestrel.models.qwen35.dflash import DFlashConfig
    (tmp_path / 'config.json').write_text('{}')
    runtime = _verification_runtime(weight_format='bf16')
    runtime.decode_path = 'auto'
    runtime._draft_model_path = tmp_path
    runtime._cfg.enable_cuda_graphs = True
    runtime._cfg.page_size = 1
    draft = _verification_draft().config
    draft.target_layer_ids = (1,6,11,16,22,27,32,37)
    monkeypatch.setattr(DFlashConfig, 'from_dict', lambda data: draft)
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda device: (9,0))
    monkeypatch.setattr(module, 'resolve_program', lambda **kwargs: pytest.fail('unqualified target resolved an AOT program'))
    prepare, finalize = module.verification_weight_loading_callbacks(runtime)
    prepare(runtime.model)
    finalize(runtime.model)
    assert not hasattr(runtime, '_verification_weight_storage')
    runtime.decode_path = 'generated'
    prepare, finalize = module.verification_weight_loading_callbacks(runtime)
    with pytest.raises(ValueError, match='unsupported Hopper'):
        prepare(runtime.model)


@pytest.mark.parametrize('sequences', [5, 6, 7, 8])
@pytest.mark.parametrize('width', [2, 4])
def test_hopper_extended_capacity_requires_exact_packaged_span(monkeypatch, sequences, width):
    from kestrel.models.qwen35 import generated_verification as module
    runtime, draft = _hopper_verification_runtime(8)
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda device: (9, 0))
    monkeypatch.setattr(torch.cuda, 'get_device_properties',
                        lambda device: SimpleNamespace(multi_processor_count=132))
    seen = []
    monkeypatch.setattr(module, 'resolve_program', lambda **kwargs: seen.append(kwargs))
    if width == 2:
        with pytest.raises(RuntimeError, match='missing AOT'):
            Qwen35GeneratedVerification(runtime, draft, sequences=sequences,
                                         verification_rows=width)
        assert seen[0]['program'].endswith(f'_c{sequences}_t2')
    else:
        with pytest.raises(ValueError, match='unsupported concurrent'):
            Qwen35GeneratedVerification(runtime, draft, sequences=sequences,
                                         verification_rows=width)
        assert not seen
    # The opt-in compiled path still requires numerical qualification before
    # automatic admission is expanded beyond the existing supported contract.
    runtime._cfg.enable_cuda_graphs = True
    assert not Qwen35GeneratedVerification.supports(runtime, draft)
