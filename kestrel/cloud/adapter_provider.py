"""LoRA adapter provider using the Moondream API."""

from __future__ import annotations

import io
import logging
from collections import OrderedDict
from typing import TYPE_CHECKING

import httpx
import torch

from kestrel.moondream.config import TextConfig
from kestrel.moondream.lora import LoRA, TextLoRA, TextLoRAConfig

if TYPE_CHECKING:
    pass

_LOGGER = logging.getLogger(__name__)


class AdapterLoadError(Exception):
    """Raised when adapter loading fails."""

    pass


class MoondreamAdapterProvider:
    """LoRA adapter provider using Moondream API.

    Implements the AdapterProvider protocol for kestrel's inference engine.

    Adapter ID format: "{finetune_id}@{step}" (e.g., "01HABC123@1000")
    """

    def __init__(
        self,
        text_config: TextConfig,
        *,
        api_key: str,
        api_base_url: str = "https://api.moondream.ai",
        max_lora_rank: int = 16,
        cache_size: int = 32,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        """Initialize the provider.

        Args:
            text_config: Text model configuration (needed to construct LoRA).
            api_key: Moondream API key for authentication.
            api_base_url: Base URL for the Moondream API.
            max_lora_rank: Maximum LoRA rank supported by this provider.
            cache_size: Maximum number of adapters to cache in memory.
            device: Device to load adapters onto.
            dtype: Data type for adapter weights.
        """
        self._text_config = text_config
        self._api_key = api_key
        self._api_base_url = api_base_url.rstrip("/")
        self._max_lora_rank = max_lora_rank
        self._cache_size = cache_size
        self._device = device
        self._dtype = dtype

        # LRU cache: OrderedDict maintains insertion order
        self._cache: OrderedDict[str, LoRA] = OrderedDict()

        # Reusable HTTP client
        self._client = httpx.Client(
            timeout=60.0,
            headers={"X-Moondream-Auth": api_key},
        )

    def config(self) -> dict:
        """Return workspace configuration.

        Required by AdapterProvider protocol.
        """
        return {"max_lora_rank": self._max_lora_rank}

    def get(self, adapter: str) -> LoRA:
        """Load and return LoRA adapter for the given adapter ID.

        Args:
            adapter: Adapter identifier in format "finetune_id@step"

        Returns:
            LoRA module ready for inference.

        Raises:
            AdapterLoadError: If loading fails.
            ValueError: If adapter ID format is invalid.
        """
        # Check cache
        if adapter in self._cache:
            # Move to end (most recently used)
            self._cache.move_to_end(adapter)
            return self._cache[adapter]

        # Parse adapter ID
        if "@" not in adapter:
            raise ValueError(
                f"Invalid adapter ID format: '{adapter}'. "
                "Expected format: 'finetune_id@step'"
            )
        finetune_id, step = adapter.split("@", 1)
        if not finetune_id or not step:
            raise ValueError(
                f"Invalid adapter ID format: '{adapter}'. "
                "Expected format: 'finetune_id@step'"
            )

        # Load adapter
        lora = self._load_adapter(finetune_id, step)

        # Add to cache with LRU eviction
        self._cache[adapter] = lora
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)  # Remove oldest

        return lora

    def _load_adapter(self, finetune_id: str, step: str) -> LoRA:
        """Download and construct LoRA adapter."""
        # Fetch presigned URL
        presigned_url = self._fetch_presigned_url(finetune_id, step)

        # Download checkpoint
        checkpoint = self._download_checkpoint(presigned_url)

        # Construct LoRA
        return self._construct_lora(checkpoint)

    def _fetch_presigned_url(self, finetune_id: str, step: str) -> str:
        """Fetch presigned download URL from Moondream API."""
        url = (
            f"{self._api_base_url}"
            f"/v1/tuning/finetunes/{finetune_id}/checkpoints/{step}/download"
        )

        try:
            response = self._client.get(url)

            if response.status_code == 404:
                raise AdapterLoadError(
                    f"Adapter not found: finetune_id={finetune_id}, step={step}"
                )

            response.raise_for_status()
            data = response.json()

            presigned_url = data.get("url")
            if not presigned_url:
                raise AdapterLoadError(
                    f"API response missing 'url' field for "
                    f"finetune_id={finetune_id}, step={step}"
                )

            return presigned_url

        except httpx.HTTPStatusError as e:
            raise AdapterLoadError(
                f"Failed to fetch presigned URL: {e.response.status_code}"
            ) from e
        except httpx.RequestError as e:
            raise AdapterLoadError(f"Network error fetching presigned URL: {e}") from e

    def _download_checkpoint(self, presigned_url: str) -> dict:
        """Download checkpoint from presigned URL."""
        try:
            # Use fresh request without auth headers for presigned URL
            response = httpx.get(presigned_url, timeout=60.0)
            response.raise_for_status()

            # Parse checkpoint
            buffer = io.BytesIO(response.content)
            checkpoint = torch.load(buffer, map_location="cpu", weights_only=True)

            if not isinstance(checkpoint, dict):
                raise AdapterLoadError("Invalid checkpoint format: expected dict")

            if "lora_state_dict" not in checkpoint:
                raise AdapterLoadError(
                    "Invalid checkpoint format: missing 'lora_state_dict'"
                )

            return checkpoint

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 403:
                raise AdapterLoadError(
                    "Presigned URL expired or invalid. Please retry."
                ) from e
            raise AdapterLoadError(
                f"Failed to download checkpoint: {e.response.status_code}"
            ) from e
        except httpx.RequestError as e:
            raise AdapterLoadError(f"Network error downloading checkpoint: {e}") from e
        except Exception as e:
            if isinstance(e, AdapterLoadError):
                raise
            raise AdapterLoadError(f"Failed to parse checkpoint: {e}") from e

    def _construct_lora(self, checkpoint: dict) -> LoRA:
        """Construct LoRA module from checkpoint."""
        state_dict = checkpoint["lora_state_dict"]

        # Detect rank from state dict
        rank = self._detect_rank(state_dict)

        if rank > self._max_lora_rank:
            raise AdapterLoadError(
                f"Adapter rank ({rank}) exceeds max_lora_rank ({self._max_lora_rank})"
            )

        # Create TextLoRA structure
        lora_config = TextLoRAConfig(rank=rank)
        text_lora = TextLoRA(
            text_config=self._text_config,
            lora_config=lora_config,
            dtype=self._dtype,
        )

        # Create LoRA wrapper
        lora = LoRA(text=text_lora, vision=None)

        # Load state dict
        lora.load_state_dict(state_dict, strict=False)
        lora.to(self._device)

        return lora

    def _detect_rank(self, state_dict: dict) -> int:
        """Detect LoRA rank from state dict tensors."""
        # Look for dense layer tensors to detect rank
        for key, tensor in state_dict.items():
            if "dense" in key and "up_a" in key and isinstance(tensor, torch.Tensor):
                # Dense up_a shape: [rank, d_model]
                return tensor.shape[0]

        # Fallback: check MoE layers
        for key, tensor in state_dict.items():
            if "moe" in key and "up_a" in key and isinstance(tensor, torch.Tensor):
                # MoE up_a shape: [num_experts, rank_per_expert, d_model]
                rank_per_expert = tensor.shape[1]
                moe_cfg = self._text_config.moe
                if moe_cfg:
                    return rank_per_expert * moe_cfg.experts_per_token
                return rank_per_expert

        raise AdapterLoadError("Could not detect LoRA rank from state dict")

    def close(self) -> None:
        """Close HTTP client."""
        self._client.close()
