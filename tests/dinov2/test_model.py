from __future__ import annotations

from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from kestrel.models.dinov2.config import Dinov2Config
from kestrel.models.dinov2.model import Dinov2Layer, Dinov2Model, Dinov2PatchEmbeddings

from ._fixtures import MODEL_CONFIG


def _small_config() -> Dinov2Config:
    return replace(
        Dinov2Config.from_dict(MODEL_CONFIG),
        hidden_size=24,
        image_size=8,
        mlp_ratio=2,
        num_attention_heads=4,
        num_hidden_layers=2,
        patch_size=2,
    )


def _reference_layer(layer: Dinov2Layer, hidden_states: torch.Tensor) -> torch.Tensor:
    normed = F.layer_norm(
        hidden_states,
        layer.norm1.normalized_shape,
        layer.norm1.weight,
        layer.norm1.bias,
        layer.norm1.eps,
    )
    attention = layer.attention.attention
    batch, tokens, hidden = normed.shape
    heads = attention.num_attention_heads
    head_dim = attention.attention_head_size

    def project(module: torch.nn.Linear) -> torch.Tensor:
        projected = F.linear(normed, module.weight, module.bias)
        return projected.reshape(batch, tokens, heads, head_dim).transpose(1, 2)

    query = project(attention.query)
    key = project(attention.key)
    value = project(attention.value)
    probs = torch.softmax(query @ key.transpose(-2, -1) * attention.scaling, dim=-1)
    context = (probs @ value).transpose(1, 2).contiguous().reshape(batch, tokens, hidden)
    attention_output = F.linear(
        context,
        layer.attention.output.dense.weight,
        layer.attention.output.dense.bias,
    )
    hidden_states = hidden_states + attention_output * layer.layer_scale1.lambda1

    normed = F.layer_norm(
        hidden_states,
        layer.norm2.normalized_shape,
        layer.norm2.weight,
        layer.norm2.bias,
        layer.norm2.eps,
    )
    mlp = F.linear(normed, layer.mlp.fc1.weight, layer.mlp.fc1.bias)
    mlp = F.gelu(mlp, approximate="none")
    mlp = F.linear(mlp, layer.mlp.fc2.weight, layer.mlp.fc2.bias)
    return hidden_states + mlp * layer.layer_scale2.lambda1


def test_seeded_layer_matches_independent_inference_dataflow() -> None:
    torch.manual_seed(17)
    layer = Dinov2Layer(_small_config()).eval()
    hidden_states = torch.randn(2, 17, 24)
    expected = _reference_layer(layer, hidden_states)
    actual = layer(hidden_states)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_model_output_contract_and_state_dict_names() -> None:
    torch.manual_seed(19)
    model = Dinov2Model(_small_config()).eval()
    output = model(torch.randn(1, 3, 4, 6))
    assert output.last_hidden_state.shape == (1, 7, 24)
    assert output.pooler_output.shape == (1, 24)
    torch.testing.assert_close(output.pooler_output, output.last_hidden_state[:, 0, :])

    names = set(model.state_dict())
    assert "embeddings.mask_token" not in names
    assert "embeddings.patch_embeddings.projection.weight" in names
    assert "encoder.layer.1.attention.attention.query.weight" in names
    assert "encoder.layer.1.layer_scale2.lambda1" in names
    assert "layernorm.weight" in names


def test_position_interpolation_preserves_cls_and_resamples_grid() -> None:
    model = Dinov2Model(_small_config()).eval()
    embeddings = torch.empty(1, 7, 24)
    actual = model.embeddings.interpolate_pos_encoding(embeddings, 4, 6)
    table = model.embeddings.position_embeddings
    expected_grid = F.interpolate(
        table[:, 1:].reshape(1, 4, 4, 24).permute(0, 3, 1, 2).float(),
        size=(2, 3),
        mode="bicubic",
        align_corners=False,
    ).permute(0, 2, 3, 1).reshape(1, 6, 24)
    expected = torch.cat((table[:, :1], expected_grid), dim=1)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_position_interpolation_is_cached_under_inference() -> None:
    """Serving a fixed non-518 crop must not re-derive the bicubic table per forward;
    the cache returns the identical tensor (bitwise-neutral by construction) and a
    weight reload (in-place copy) invalidates it."""
    model = Dinov2Model(_small_config()).eval()
    embeddings = torch.empty(1, 7, 24)
    with torch.inference_mode():
        first = model.embeddings.interpolate_pos_encoding(embeddings, 4, 6)
        second = model.embeddings.interpolate_pos_encoding(embeddings, 4, 6)
    assert second is first
    assert len(model.embeddings._resampled_positions) == 1

    with torch.no_grad():
        model.embeddings.position_embeddings.copy_(
            model.embeddings.position_embeddings * 2.0
        )
    with torch.inference_mode():
        reloaded = model.embeddings.interpolate_pos_encoding(embeddings, 4, 6)
    assert reloaded is not first
    torch.testing.assert_close(reloaded, first * 2.0)

    with torch.enable_grad():
        grad_path = model.embeddings.interpolate_pos_encoding(embeddings, 4, 6)
    assert grad_path.grad_fn is not None  # never served from the detached cache


@pytest.mark.parametrize(
    ("shape", "message"),
    [
        ((3, 8, 8), "shape"),
        ((1, 1, 8, 8), "channels"),
        ((1, 3, 7, 8), "divisible"),
    ],
)
def test_patch_embed_validates_inputs(shape: tuple[int, ...], message: str) -> None:
    model = Dinov2Model(_small_config())
    with pytest.raises(ValueError, match=message):
        model.embeddings.patch_embeddings(torch.zeros(shape))


@pytest.mark.parametrize("image_size", (224, 518))
def test_nonoverlapping_patch_conv_equals_patchify_linear(image_size: int) -> None:
    """Kernel=stride=14 makes patch projection algebraically a patchify + linear."""
    torch.manual_seed(23)
    config = Dinov2Config.from_dict(MODEL_CONFIG)
    patch_embed = Dinov2PatchEmbeddings(config).eval()
    pixels = torch.randn(1, config.num_channels, image_size, image_size)

    actual = patch_embed(pixels)
    patch = config.patch_size
    grid = image_size // patch
    patchified = (
        pixels.unfold(2, patch, patch)
        .unfold(3, patch, patch)
        .permute(0, 2, 3, 1, 4, 5)
        .reshape(1, grid * grid, config.num_channels * patch * patch)
    )
    expected = F.linear(
        patchified,
        patch_embed.projection.weight.flatten(1),
        patch_embed.projection.bias,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)
