import torch
from torch.compiler import disable as torch_compiler_disable

from .module import (
    FusedMoEConfig,
    _MoEExecutionContext,
    _MoEWorkspaces,
    _apply_single_lora,
    _make_execution_context,
    _prepare_moe_forward,
    _prepare_single_lora_routing,
    _run_base_down,
    _run_base_up,
    _run_moe_activation,
    _sum_moe_outputs,
    get_shared_moe_workspaces,
)


def _validate_optional_lora_pair(
    name: str,
    a: torch.Tensor | None,
    b: torch.Tensor | None,
) -> None:
    if (a is None) != (b is None):
        raise ValueError(f"{name} LoRA tensors must both be set or both be None")


class _DirectMoERunner:
    """Hidden tensor-level MoE runner."""

    def __init__(
        self,
        *,
        top_k: int,
        hidden_size: int,
        input_size: int,
        num_experts: int,
        config: FusedMoEConfig | None = None,
        tuned_configs: dict[int, dict[str, int] | None] | None = None,
        cute_config_available: dict[str, bool] | None = None,
    ) -> None:
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.num_experts = num_experts
        self.config = config or FusedMoEConfig()
        self._tuned_configs = tuned_configs if tuned_configs is not None else {}
        self._cute_config_available = (
            cute_config_available if cute_config_available is not None else {}
        )

    @property
    def _workspaces(self) -> _MoEWorkspaces:
        return get_shared_moe_workspaces()

    @torch_compiler_disable()
    def run(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        *,
        up_weight: torch.Tensor,
        down_weight: torch.Tensor,
        up_scale: torch.Tensor | None = None,
        down_scale: torch.Tensor | None = None,
        up_lora_a: torch.Tensor | None = None,
        up_lora_b: torch.Tensor | None = None,
        down_lora_a: torch.Tensor | None = None,
        down_lora_b: torch.Tensor | None = None,
        up_lora_scale: float = 1.0,
        down_lora_scale: float = 1.0,
        down_mul_routed_weight: bool = True,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        _validate_optional_lora_pair("up", up_lora_a, up_lora_b)
        _validate_optional_lora_pair("down", down_lora_a, down_lora_b)

        if hidden_states.size(0) == 0:
            if return_aux:
                return hidden_states, {
                    "up_out": hidden_states.new_empty((0, self.top_k, self.hidden_size * 2))
                }
            return hidden_states

        if hidden_states.dtype != torch.bfloat16:
            raise ValueError(
                "Hidden tensor-level MoE runner requires bfloat16 hidden states."
            )

        ctx: _MoEExecutionContext = _make_execution_context(self)
        prepared = _prepare_moe_forward(
            ctx,
            hidden_states,
            topk_weights,
            topk_ids,
            up_weight=up_weight,
            down_weight=down_weight,
            up_scale=up_scale,
            down_scale=down_scale,
            persistent_up_out=return_aux,
        )
        _run_base_up(ctx, prepared)

        lora_routing = None
        has_direct_lora = up_lora_a is not None or down_lora_a is not None
        if has_direct_lora:
            lora_routing = _prepare_single_lora_routing(ctx, prepared)

        if up_lora_a is not None:
            up_lora = torch.zeros_like(prepared.up_out)
            _apply_single_lora(
                ctx,
                x=prepared.hidden_states,
                topk_ids=prepared.topk_ids,
                topk_weights=prepared.topk_weights,
                output=up_lora,
                lora_a=up_lora_a,
                lora_b=up_lora_b,
                routing=lora_routing,
                lora_id=0,
                top_k=self.top_k,
                mul_routed_weight=False,
                shrink_config=self.config.lora_prefill_shrink,
                expand_config=self.config.lora_prefill_expand,
            )
            if up_lora_scale != 1.0:
                up_lora.mul_(up_lora_scale)
            prepared.up_out.add_(up_lora)

        down_in = _run_moe_activation(ctx, prepared)
        down_topk_weights = prepared.topk_weights.view(-1)
        if not down_mul_routed_weight:
            down_topk_weights = torch.ones_like(down_topk_weights)
        down_out = _run_base_down(
            ctx,
            prepared,
            down_in,
            topk_weights=down_topk_weights,
        )

        if down_lora_a is not None:
            down_lora = torch.zeros_like(down_out)
            _apply_single_lora(
                ctx,
                x=down_in,
                topk_ids=prepared.topk_ids,
                topk_weights=prepared.topk_weights,
                output=down_lora,
                lora_a=down_lora_a,
                lora_b=down_lora_b,
                routing=lora_routing,
                lora_id=0,
                top_k=1,
                mul_routed_weight=down_mul_routed_weight,
                shrink_config=self.config.lora_prefill_shrink,
                expand_config=self.config.lora_prefill_expand,
            )
            if down_lora_scale != 1.0:
                down_lora.mul_(down_lora_scale)
            down_out.add_(down_lora)

        down_out_gated = down_out
        if not down_mul_routed_weight:
            down_out_gated = down_out * prepared.topk_weights.unsqueeze(-1)

        fused = _sum_moe_outputs(
            ctx,
            prepared,
            down_out_gated,
            output=torch.empty(
                (prepared.num_tokens, ctx.input_size),
                device=prepared.hidden_states.device,
                dtype=prepared.hidden_states.dtype,
            ),
        )
        if return_aux:
            return fused, {"up_out": prepared.up_out}
        return fused


def _make_direct_moe_runner(
    *,
    top_k: int,
    hidden_size: int,
    input_size: int,
    num_experts: int,
    config: FusedMoEConfig | None = None,
) -> _DirectMoERunner:
    return _DirectMoERunner(
        top_k=top_k,
        hidden_size=hidden_size,
        input_size=input_size,
        num_experts=num_experts,
        config=config,
    )
