"""GPU integration for dedicated Whisper control-prefix prefill."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from kestrel.models.whisper.prefill_decoder_prefix import (
    ATTENTION_HEADS,
    DECODER_LAYERS,
    HEAD_DIM,
    VOCAB_SIZE,
    WhisperDecoderPrefixWorkspace,
    prepare_whisper_decoder_weights,
    whisper_decoder_prefix,
    whisper_no_speech_probabilities,
)
from kestrel.models.whisper.config import WhisperTurboConfig
from kestrel.models.whisper.runtime_abi import WhisperSelfKVArenas
from kestrel.models.whisper.weights import (
    AttentionWeights,
    DecoderLayerWeights,
    LayerNormWeights,
    LinearWeights,
    WhisperDecoderWeights as EagerDecoderWeights,
)

from .eager_model import CrossAttentionKV, WhisperInferenceModel


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Whisper decoder-prefix kernels require CUDA",
)


def _random(shape, *, scale=0.01, offset=0.0):
    return (
        torch.randn(shape, device="cuda", dtype=torch.bfloat16) * scale + offset
    ).contiguous()


def _linear(out_features, in_features, *, bias, scale=0.004):
    return LinearWeights(
        weight=_random((out_features, in_features), scale=scale),
        bias=_random((out_features,), scale=scale) if bias else None,
    )


def _attention(config):
    width = config.d_model
    return AttentionWeights(
        query=_linear(width, width, bias=True),
        key=_linear(width, width, bias=False),
        value=_linear(width, width, bias=True),
        output=_linear(width, width, bias=True),
    )


def _eager_and_kernel_weights():
    torch.manual_seed(20260811)
    config = WhisperTurboConfig()
    width = config.d_model
    norm = LayerNormWeights(
        weight=_random((width,), scale=0.01, offset=1.0),
        bias=_random((width,), scale=0.01),
    )
    eager_layers = []
    for _ in range(config.decoder_layers):
        self_attention = _attention(config)
        cross_attention = _attention(config)
        eager_layer = DecoderLayerWeights(
            self_attention_layer_norm=norm,
            self_attention=self_attention,
            cross_attention_layer_norm=norm,
            cross_attention=cross_attention,
            final_layer_norm=norm,
            fc1=_linear(config.decoder_ffn_dim, width, bias=True, scale=0.002),
            fc2=_linear(width, config.decoder_ffn_dim, bias=True, scale=0.002),
        )
        eager_layers.append(eager_layer)

    token_embedding = _random((config.vocab_size, width), scale=0.02)
    position_embedding = _random((config.max_target_positions, width), scale=0.02)
    eager_decoder = EagerDecoderWeights(
        token_embedding=token_embedding,
        position_embedding=position_embedding,
        layers=tuple(eager_layers),
        final_layer_norm=norm,
    )
    eager = object.__new__(WhisperInferenceModel)
    eager.config = config
    eager.weights = SimpleNamespace(decoder=eager_decoder)
    kernel = prepare_whisper_decoder_weights(eager.weights)
    return config, eager, kernel


def _assert_close(actual, expected, *, cosine=0.997, atol=0.08):
    actual_f32 = actual.float()
    expected_f32 = expected.float()
    similarity = F.cosine_similarity(
        actual_f32.flatten(), expected_f32.flatten(), dim=0
    )
    assert similarity > cosine
    torch.testing.assert_close(actual, expected, rtol=8e-2, atol=atol)


def _paged_state(config, *, state_rows):
    n_pages = 1 + state_rows * config.max_target_positions
    shape = (n_pages, ATTENTION_HEADS, HEAD_DIM)
    keys = tuple(
        torch.full(shape, 7, device="cuda", dtype=torch.bfloat16)
        for _ in range(DECODER_LAYERS)
    )
    values = tuple(
        torch.full(shape, -7, device="cuda", dtype=torch.bfloat16)
        for _ in range(DECODER_LAYERS)
    )
    self_kv = WhisperSelfKVArenas(keys, values)
    generator = torch.Generator().manual_seed(19)
    physical = torch.randperm(n_pages - 1, generator=generator) + 1
    page_table = torch.zeros(
        (state_rows, config.max_target_positions), dtype=torch.int32
    )
    page_table.copy_(physical.view_as(page_table))
    return self_kv, page_table.cuda()


def test_mixed_prefix_matches_eager_oracle_paged_layout_and_graph() -> None:
    config, eager, weights = _eager_and_kernel_weights()
    batch = 4
    lengths = torch.tensor([1, 3, 4, 4], device="cuda", dtype=torch.int32)
    sot_positions = torch.tensor([0, 0, 2, 0], device="cuda", dtype=torch.int64)
    token_ids = torch.tensor(
        [
            [50258, 50257, 50257, 50257],
            [50258, 50259, 50360, 50257],
            [50362, 10, 50258, 50259],
            [50362, 10, 11, 12],
        ],
        device="cuda",
        dtype=torch.int64,
    )
    compact_keys = _random(
        (
            DECODER_LAYERS,
            batch,
            config.max_source_positions,
            ATTENTION_HEADS,
            HEAD_DIM,
        ),
        scale=0.02,
    )
    compact_values = _random(compact_keys.shape, scale=0.02)
    expected = []
    for row, length in enumerate((1, 3, 4, 4)):
        row_cross = CrossAttentionKV(
            compact_keys[:, row : row + 1].contiguous(),
            compact_values[:, row : row + 1].contiguous(),
        )
        expected.append(
            eager.decoder_prefix(
                token_ids[row : row + 1, :length].contiguous(), row_cross
            )
        )

    state_rows = 5
    batch_idx = torch.tensor([4, 1, 3, 2], device="cuda", dtype=torch.int64)
    self_kv, page_table = _paged_state(config, state_rows=state_rows)
    slot_mapping = page_table.index_select(0, batch_idx)[:, :4].long().contiguous()
    positions = torch.arange(4, device="cuda").view(1, 4)
    slot_mapping.masked_fill_(positions >= lengths.view(-1, 1), 0)
    workspace = WhisperDecoderPrefixWorkspace.allocate(batch, device="cuda")
    logits = torch.empty((batch, VOCAB_SIZE), device="cuda", dtype=torch.bfloat16)
    no_speech_probs = torch.empty((batch,), device="cuda", dtype=torch.float32)
    key_pool_ptrs = tuple(pool.data_ptr() for pool in self_kv.keys)
    value_pool_ptrs = tuple(pool.data_ptr() for pool in self_kv.values)

    actual = whisper_decoder_prefix(
        token_ids,
        lengths,
        sot_positions,
        slot_mapping,
        compact_keys,
        compact_values,
        weights,
        workspace,
        self_kv,
        logits_out=logits,
    )
    scored_no_speech = whisper_no_speech_probabilities(
        workspace,
        probabilities_out=no_speech_probs,
    )
    for row, length in enumerate((1, 3, 4, 4)):
        _assert_close(
            actual.last_hidden_state[row],
            expected[row].hidden_states[0, length - 1],
        )
        _assert_close(
            actual.logits[row], expected[row].logits[0, length - 1], atol=0.12
        )
        if row < 3:
            retained_sot_logits = workspace.projection_logits[batch + row]
            _assert_close(
                retained_sot_logits,
                expected[row].logits[0, int(sot_positions[row])],
                atol=0.12,
            )
            expected_no_speech = retained_sot_logits.float().softmax(dim=-1)[50361]
            torch.testing.assert_close(
                scored_no_speech[row],
                expected_no_speech,
                rtol=1e-5,
                atol=1e-8,
            )
        for layer in range(DECODER_LAYERS):
            for position in range(length):
                page = int(slot_mapping[row, position])
                _assert_close(
                    self_kv.keys[layer][page],
                    expected[row].self_kv.keys[layer, 0, position],
                    cosine=0.9999,
                    atol=0.02,
                )
                _assert_close(
                    self_kv.values[layer][page],
                    expected[row].self_kv.values[layer, 0, position],
                    cosine=0.9999,
                    atol=0.02,
                )

    untouched_page = int(page_table[2, 10])
    for layer in range(DECODER_LAYERS):
        assert torch.all(self_kv.keys[layer][untouched_page] == 7)
        assert torch.all(self_kv.values[layer][untouched_page] == -7)
    assert tuple(pool.data_ptr() for pool in self_kv.keys) == key_pool_ptrs
    assert tuple(pool.data_ptr() for pool in self_kv.values) == value_pool_ptrs
    assert actual.logits.data_ptr() == logits.data_ptr()
    assert actual.last_hidden_state.data_ptr() == workspace.last_hidden.data_ptr()
    assert workspace.last_hidden.data_ptr() == workspace.projection_hidden.data_ptr()

    # Automatic-language batches stage SOT as every row's last token. Their
    # score comes from the ordinary B-row logits, and the 2B projection buffers
    # must remain untouched.
    workspace.projection_logits.fill_(float("nan"))
    direct = whisper_decoder_prefix(
        token_ids,
        lengths,
        sot_positions,
        slot_mapping,
        compact_keys,
        compact_values,
        weights,
        workspace,
        self_kv,
        logits_out=logits,
        separate_sot_projection=False,
    )
    assert torch.isnan(workspace.projection_logits).all()
    for row, length in enumerate((1, 3, 4, 4)):
        _assert_close(
            direct.logits[row], expected[row].logits[0, length - 1], atol=0.12
        )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = whisper_decoder_prefix(
            token_ids,
            lengths,
            sot_positions,
            slot_mapping,
            compact_keys,
            compact_values,
            weights,
            workspace,
            self_kv,
            logits_out=logits,
        )
    graph.replay()
    whisper_no_speech_probabilities(
        workspace,
        probabilities_out=no_speech_probs,
    )
    torch.cuda.synchronize()
    assert captured.logits.data_ptr() == logits.data_ptr()
    assert captured.last_hidden_state.data_ptr() == workspace.last_hidden.data_ptr()
    for row, length in enumerate((1, 3, 4, 4)):
        _assert_close(
            captured.logits[row], expected[row].logits[0, length - 1], atol=0.12
        )
