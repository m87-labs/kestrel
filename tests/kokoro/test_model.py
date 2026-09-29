from __future__ import annotations

import warnings
from pathlib import Path

import pytest
import torch

from kestrel.models.kokoro.albert import KokoroAlbert
from kestrel.models.kokoro.config import AlbertConfig, IstftNetConfig, KokoroConfig
from kestrel.models.kokoro.model import KokoroModel
from kestrel.models.kokoro.orchestrator import _split_phonemes
from kestrel.models.kokoro.weights import VoiceStore, _materialize_weight_norm


def _config() -> KokoroConfig:
    return KokoroConfig(
        vocab={"a": 43},
        n_token=178,
        hidden_dim=512,
        style_dim=128,
        n_layer=3,
        max_dur=50,
        n_mels=80,
        text_encoder_kernel_size=5,
        albert=AlbertConfig(178, 768, 12, 2048, 512, 12),
        istftnet=IstftNetConfig(
            upsample_kernel_sizes=(20, 12),
            upsample_rates=(10, 6),
            gen_istft_hop_size=5,
            gen_istft_n_fft=20,
            resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
            resblock_kernel_sizes=(3, 7, 11),
            upsample_initial_channel=512,
        ),
    )


def test_v1_model_layout_matches_published_checkpoint() -> None:
    with torch.device("meta"):
        model = KokoroModel(_config())
    state = model.state_dict()
    assert state["bert.encoder.albert_layer_groups.0.albert_layers.0.ffn.weight"].shape == (
        2048,
        768,
    )
    assert state["decoder.generator.conv_post.weight"].shape == (22, 128, 7)
    assert state["text_encoder.cnn.2.0.weight"].shape == (512, 512, 5)


def test_weight_norm_materialization_matches_pytorch() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        layer = torch.nn.utils.weight_norm(
            torch.nn.ConvTranspose1d(4, 6, 4, groups=2)
        )
    with torch.no_grad():
        layer.weight_g.uniform_(0.25, 1.75)
        layer.weight_v.normal_()
        layer(torch.randn(1, 4, 8))
    state = {
        f"layer.{name}": value.detach().clone()
        for name, value in layer.state_dict().items()
    }

    actual = _materialize_weight_norm(state)["layer.weight"]

    torch.testing.assert_close(actual, layer.weight, rtol=1e-6, atol=1e-7)


def test_phoneme_chunks_preserve_cjk_breaks_and_model_limit() -> None:
    phonemes = "a" * 509 + "。" + "b" * 511
    chunks = _split_phonemes(phonemes)

    assert tuple(map(len, chunks)) == (510, 510, 1)
    assert "".join(chunks) == phonemes


def test_phoneme_input_rejects_unknown_symbols() -> None:
    with torch.device("meta"):
        model = KokoroModel(_config())
    with pytest.raises(ValueError, match="unsupported Kokoro symbols"):
        model.encode_phonemes("a!")


def test_direct_albert_matches_transformers_reference() -> None:
    transformers = pytest.importorskip("transformers")
    config = AlbertConfig(
        vocab_size=19,
        hidden_size=16,
        num_attention_heads=4,
        intermediate_size=32,
        max_position_embeddings=16,
        num_hidden_layers=3,
        embedding_size=8,
    )
    torch.manual_seed(7)
    direct = KokoroAlbert(config).eval()
    reference_config = transformers.AlbertConfig(
        vocab_size=19,
        embedding_size=8,
        hidden_size=16,
        num_attention_heads=4,
        intermediate_size=32,
        max_position_embeddings=16,
        num_hidden_layers=3,
        num_hidden_groups=1,
        inner_group_num=1,
        hidden_act="gelu_new",
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        layer_norm_eps=1e-12,
    )
    reference = transformers.AlbertModel(
        reference_config, add_pooling_layer=False
    ).eval()
    reference.load_state_dict(direct.state_dict(), strict=True)
    input_ids = torch.tensor([[0, 4, 7, 2, 0], [0, 3, 1, 0, 0]])
    with torch.inference_mode():
        actual = direct(input_ids)
        expected = reference(input_ids=input_ids).last_hidden_state
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_voice_store_loads_exact_row_and_blends(tmp_path: Path) -> None:
    voices = tmp_path / "voices"
    voices.mkdir()
    first = torch.arange(4 * 256, dtype=torch.float32).reshape(4, 1, 256)
    second = first + 10
    torch.save(first, voices / "af_first.pt")
    torch.save(second, voices / "af_second.pt")
    store = VoiceStore(root=tmp_path)

    torch.testing.assert_close(store.style("af_first", 3), first[2])
    torch.testing.assert_close(
        store.style("af_first,af_second", 2), (first[1] + second[1]) / 2
    )
    with pytest.raises(ValueError, match="outside voice pack range"):
        store.style("af_first", 5)
    with pytest.raises(ValueError, match="comma-separated"):
        store.load("../secret")
