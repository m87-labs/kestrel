"""Minimal inference-only ALBERT used by Kokoro's duration encoder."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .config import AlbertConfig


class AlbertEmbeddings(nn.Module):
    def __init__(self, config: AlbertConfig) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.embedding_size, padding_idx=0
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.embedding_size
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.embedding_size
        )
        # Capitalization is checkpoint-defined by Hugging Face ALBERT.
        self.LayerNorm = nn.LayerNorm(
            config.embedding_size, eps=config.layer_norm_eps
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(
            input_ids.shape[1], device=input_ids.device
        ).unsqueeze(0)
        token_types = torch.zeros_like(input_ids)
        hidden = self.word_embeddings(input_ids)
        hidden = hidden + self.position_embeddings(positions)
        hidden = hidden + self.token_type_embeddings(token_types)
        return self.LayerNorm(hidden)


class AlbertAttention(nn.Module):
    def __init__(self, config: AlbertConfig) -> None:
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        if self.attention_head_size * self.num_attention_heads != config.hidden_size:
            raise ValueError("ALBERT hidden size must divide evenly into heads")
        self.query = nn.Linear(config.hidden_size, config.hidden_size)
        self.key = nn.Linear(config.hidden_size, config.hidden_size)
        self.value = nn.Linear(config.hidden_size, config.hidden_size)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def _heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, length, _ = tensor.shape
        return tensor.view(
            batch, length, self.num_attention_heads, self.attention_head_size
        ).transpose(1, 2)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        query = self._heads(self.query(hidden))
        key = self._heads(self.key(hidden))
        value = self._heads(self.value(hidden))
        context = F.scaled_dot_product_attention(query, key, value)
        context = context.transpose(1, 2).flatten(2)
        return self.LayerNorm(hidden + self.dense(context))


class AlbertLayer(nn.Module):
    def __init__(self, config: AlbertConfig) -> None:
        super().__init__()
        self.full_layer_layer_norm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.attention = AlbertAttention(config)
        self.ffn = nn.Linear(config.hidden_size, config.intermediate_size)
        self.ffn_output = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        attended = self.attention(hidden)
        feed_forward = self.ffn_output(F.gelu(self.ffn(attended), approximate="tanh"))
        return self.full_layer_layer_norm(attended + feed_forward)


class AlbertLayerGroup(nn.Module):
    def __init__(self, config: AlbertConfig) -> None:
        super().__init__()
        # Kokoro uses ALBERT's default one shared inner layer.
        self.albert_layers = nn.ModuleList([AlbertLayer(config)])


class AlbertEncoder(nn.Module):
    def __init__(self, config: AlbertConfig) -> None:
        super().__init__()
        self.embedding_hidden_mapping_in = nn.Linear(
            config.embedding_size, config.hidden_size
        )
        # Kokoro uses ALBERT's default one hidden group, shared for all 12 layers.
        self.albert_layer_groups = nn.ModuleList([AlbertLayerGroup(config)])


class KokoroAlbert(nn.Module):
    """Checkpoint-compatible subset of ``transformers.AlbertModel``."""

    def __init__(self, config: AlbertConfig) -> None:
        super().__init__()
        self.config = config
        self.embeddings = AlbertEmbeddings(config)
        self.encoder = AlbertEncoder(config)

    @property
    def device(self) -> torch.device:
        return self.embeddings.word_embeddings.weight.device

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("ALBERT input_ids must be [batch, sequence]")
        if input_ids.shape[1] > self.config.max_position_embeddings:
            raise ValueError("ALBERT input exceeds max_position_embeddings")
        hidden = self.encoder.embedding_hidden_mapping_in(
            self.embeddings(input_ids)
        )
        layer = self.encoder.albert_layer_groups[0].albert_layers[0]
        for _ in range(self.config.num_hidden_layers):
            hidden = layer(hidden)
        return hidden


__all__ = ["KokoroAlbert"]
