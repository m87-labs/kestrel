"""GPU integration for the dedicated Whisper encoder kernel runtime."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from kestrel.models.whisper.prefill_encoder import (
    WhisperEncoderWorkspace,
    prepare_whisper_encoder_weights,
    whisper_cross_kv,
    whisper_encoder,
)
from kestrel.models.whisper.prefill_encoder import _run_encoder_layer
from kestrel.models.whisper.prefill_stem import whisper_audio_stem
from kestrel.models.whisper.config import WhisperTurboConfig
from kestrel.models.whisper.runtime_abi import WhisperCrossArenas
from kestrel.models.whisper.weights import (
    AttentionWeights,
    Conv1dWeights,
    EncoderLayerWeights,
    LayerNormWeights,
    LinearWeights,
    WhisperEncoderWeights as EagerEncoderWeights,
)

from .eager_model import WhisperInferenceModel, _encoder_layer


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Whisper encoder kernels require CUDA"
)


def _random(shape, *, scale=0.01, offset=0.0):
    return (
        torch.randn(shape, device="cuda", dtype=torch.bfloat16) * scale + offset
    ).contiguous()


def _linear(out_features, in_features, *, bias, scale=0.005):
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
    torch.manual_seed(20260810)
    config = WhisperTurboConfig()
    width = config.d_model
    norm = LayerNormWeights(
        weight=_random((width,), scale=0.01, offset=1.0),
        bias=_random((width,), scale=0.01),
    )
    attention = _attention(config)
    eager_layer = EncoderLayerWeights(
        self_attention_layer_norm=norm,
        self_attention=attention,
        final_layer_norm=norm,
        fc1=_linear(config.encoder_ffn_dim, width, bias=True, scale=0.003),
        fc2=_linear(width, config.encoder_ffn_dim, bias=True, scale=0.003),
    )
    conv1 = Conv1dWeights(
        weight=_random((width, config.num_mel_bins, 3), scale=0.01),
        bias=_random((width,), scale=0.01),
        stride=1,
        padding=1,
    )
    conv2 = Conv1dWeights(
        weight=_random((width, width, 3), scale=0.005),
        bias=_random((width,), scale=0.01),
        stride=2,
        padding=1,
    )
    eager_encoder = EagerEncoderWeights(
        conv1=conv1,
        conv2=conv2,
        position_embedding=_random((config.max_source_positions, width), scale=0.01),
        layers=(eager_layer,) * config.encoder_layers,
        final_layer_norm=norm,
    )
    cross_attention = _attention(config)
    eager = object.__new__(WhisperInferenceModel)
    eager.config = config
    eager.weights = SimpleNamespace(
        encoder=eager_encoder,
        decoder=SimpleNamespace(
            layers=tuple(
                SimpleNamespace(cross_attention=cross_attention)
                for _ in range(config.decoder_layers)
            )
        ),
    )

    kernel_weights = prepare_whisper_encoder_weights(eager.weights)
    return config, eager, kernel_weights


def _assert_close(actual, expected, *, cosine=0.999):
    actual_f32 = actual.float()
    expected_f32 = expected.float()
    similarity = F.cosine_similarity(
        actual_f32.flatten(), expected_f32.flatten(), dim=0
    )
    assert similarity > cosine
    torch.testing.assert_close(actual, expected, rtol=5e-2, atol=5e-2)


def test_native_encoder_intermediate_final_cross_kv_and_graph() -> None:
    config, eager, kernel_weights = _eager_and_kernel_weights()
    workspace = WhisperEncoderWorkspace.allocate(1, device="cuda")
    global_cross = WhisperCrossArenas.allocate(
        config,
        3,
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
    )
    batch_idx = torch.tensor([2], device="cuda", dtype=torch.int64)
    features = _random(
        (1, config.num_mel_bins, 2 * config.max_source_positions), scale=0.2
    )

    native_stem = whisper_audio_stem(
        features,
        kernel_weights.stem,
        out=workspace.hidden,
        require_native=True,
    )
    expected_layer0 = _encoder_layer(
        native_stem.clone(), eager.weights.encoder.layers[0], config
    )
    actual_layer0 = _run_encoder_layer(
        native_stem, kernel_weights.layers[0], workspace
    ).clone()
    _assert_close(actual_layer0, expected_layer0, cosine=0.9999)

    expected_encoder = eager.encode(features)
    actual_encoder = whisper_encoder(features, kernel_weights, workspace)
    _assert_close(actual_encoder, expected_encoder)
    assert actual_encoder.data_ptr() == workspace.encoder_output.data_ptr()

    expected_cross = eager.preproject_cross_kv(actual_encoder)
    actual_cross = whisper_cross_kv(
        actual_encoder,
        kernel_weights,
        workspace,
        global_cross,
        batch_idx,
    )
    _assert_close(actual_cross.keys[:, 2:3], expected_cross.keys, cosine=0.9999)
    _assert_close(actual_cross.values[:, 2:3], expected_cross.values, cosine=0.9999)
    assert actual_cross.keys.data_ptr() == global_cross.keys.data_ptr()
    assert actual_cross.values.data_ptr() == global_cross.values.data_ptr()
    assert actual_cross.keys[0].is_contiguous()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_encoder = whisper_encoder(features, kernel_weights, workspace)
        captured_cross = whisper_cross_kv(
            captured_encoder,
            kernel_weights,
            workspace,
            global_cross,
            batch_idx,
        )
    assert captured_encoder.data_ptr() == workspace.encoder_output.data_ptr()
    assert captured_cross.keys.data_ptr() == global_cross.keys.data_ptr()
    assert captured_cross.values.data_ptr() == global_cross.values.data_ptr()
    graph.replay()
    torch.cuda.synchronize()
    _assert_close(captured_encoder, expected_encoder)
    _assert_close(captured_cross.keys[:, 2:3], expected_cross.keys, cosine=0.9999)


def test_cross_kv_scatter_uses_permuted_global_rows_and_preserves_others() -> None:
    config, eager, kernel_weights = _eager_and_kernel_weights()
    workspace = WhisperEncoderWorkspace.allocate(2, device="cuda")
    encoder_states = _random(
        (2, config.max_source_positions, config.d_model), scale=0.1
    )
    expected = eager.preproject_cross_kv(encoder_states)
    global_cross = WhisperCrossArenas.allocate(
        config,
        4,
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
    )
    global_cross.keys.fill_(7)
    global_cross.values.fill_(-7)
    key_ptr = global_cross.keys.data_ptr()
    value_ptr = global_cross.values.data_ptr()
    batch_idx = torch.tensor([3, 1], device="cuda", dtype=torch.int64)

    actual = whisper_cross_kv(
        encoder_states,
        kernel_weights,
        workspace,
        global_cross,
        batch_idx,
    )

    _assert_close(actual.keys[:, 3], expected.keys[:, 0], cosine=0.9999)
    _assert_close(actual.keys[:, 1], expected.keys[:, 1], cosine=0.9999)
    _assert_close(actual.values[:, 3], expected.values[:, 0], cosine=0.9999)
    _assert_close(actual.values[:, 1], expected.values[:, 1], cosine=0.9999)
    assert torch.all(actual.keys[:, (0, 2)] == 7)
    assert torch.all(actual.values[:, (0, 2)] == -7)
    assert actual.keys.data_ptr() == key_ptr
    assert actual.values.data_ptr() == value_ptr

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = whisper_cross_kv(
            encoder_states,
            kernel_weights,
            workspace,
            global_cross,
            batch_idx,
        )
    graph.replay()
    torch.cuda.synchronize()
    assert captured.keys.data_ptr() == key_ptr
    assert captured.values.data_ptr() == value_ptr
    _assert_close(captured.keys[:, 3], expected.keys[:, 0], cosine=0.9999)
    _assert_close(captured.keys[:, 1], expected.keys[:, 1], cosine=0.9999)
    assert torch.all(captured.keys[:, (0, 2)] == 7)
