"""Model-owned fixed-reduction projections for explicitly native verification."""
from contextlib import contextmanager
from contextvars import ContextVar
import weakref
from torch import nn
from torch.nn import functional as F
from .verification_attention import VerificationAttention


class VerificationProjection:
    """Projection interface shared by dense layers and the MoE router."""
    _verification_scope = None

    def project(self, x, bias=None):
        scope = self._verification_scope
        if scope is not None and scope.active.get():
            if bias is not None:
                raise ValueError('native verification projections require bias-free weights')
            return self._verification_projection(x.contiguous())
        return F.linear(x, self.weight, bias)


class VerificationExperts:
    """Model hook for a prepared MoE numerical plan in native verification."""
    _verification_scope = None

    def verification_moe_handle(self, spec, device):
        scope = self._verification_scope
        if scope is None or not scope.active.get():
            return None
        handle = self._verification_experts
        if handle.spec != spec or handle.device != device:
            raise ValueError('verification experts changed their prepared spec/device')
        return handle


class VerificationLinear(nn.Linear, VerificationProjection):
    def forward(self, x):
        return self.project(x, self.bias)


class NativeVerificationProjections:
    """Own immutable weights and a task-local activation scope for one model.

    Prepare after loading/materialization and before graph capture. Other models
    have separate context variables and cannot observe this owner's activation.
    This owner's verification launches must be stream-ordered: graph cache
    entries share expert scratch and cannot execute concurrently. Outputs and
    recurrent-state ownership remain with their individual graph entries.
    """
    def __init__(self, modules, *, max_verification_tokens=None, attention_splits=None, attention_implementation="simt"):
        from kestrel_kernels.verification_dense import PreparedDense
        if attention_splits is not None and (type(attention_splits) is not int or attention_splits not in (1, 2, 4, 8)):
            raise ValueError("unsupported explicit verification attention split count")
        if attention_implementation not in ('simt', 'packed_mma', 'packed_mma_overlap') or (attention_implementation != 'simt' and attention_splits not in (2, 4, 8)):
            raise ValueError('unsupported explicit attention implementation/partition count')
        self.attention_implementation = attention_implementation
        self.attention_splits = attention_splits
        self.closed = False
        self.active = ContextVar(f'verification-projections-{id(self)}', default=False)
        owners = tuple(module for module in modules
                       if isinstance(module, (VerificationProjection, VerificationExperts))
                       or (attention_splits is not None and isinstance(module, VerificationAttention)))
        if any(module._verification_scope is not None for module in owners):
            raise ValueError('verification projections already belong to a runtime')
        # Validate complete inventory before publishing any owner binding.
        shared_experts = {}
        plans = []
        for module in owners:
            if isinstance(module, VerificationProjection):
                plan = PreparedDense(module.weight)
            elif isinstance(module, VerificationExperts):
                from kestrel_kernels.moe.api import MoeCapacity
                from kestrel_kernels.moe.padded_plan import PaddedMoePlan
                spec = module.verification_moe_spec()
                device = module.gate_up_proj.device
                key = spec, device
                if key not in shared_experts:
                    execution = PaddedMoePlan.from_generated(spec,
                        max_tokens=max_verification_tokens, device=device)
                    # Layers execute sequentially on the verification stream.
                    # One owner retains the workspace through all graph replays;
                    # another model/runtime receives an independent workspace.
                    shared_experts[key] = execution.prepare(
                        MoeCapacity(max_tokens=max_verification_tokens))
                plan = shared_experts[key]
            else:
                plan = None
            plans.append(plan)
        plans = tuple(plans)
        self.owners = tuple(weakref.ref(module) for module in owners)
        self.plans = plans
        for module, plan in zip(owners, plans, strict=True):
            if isinstance(module, VerificationProjection):
                module._verification_projection = plan
            elif isinstance(module, VerificationExperts):
                module._verification_experts = plan
            module._verification_scope = self

    @contextmanager
    def activate(self):
        if self.closed:
            raise RuntimeError('native verification projection owner is closed')
        token = self.active.set(True)
        try:
            yield
        finally:
            self.active.reset(token)

    def close(self):
        if self.active.get():
            raise RuntimeError('cannot close active native verification projections')
        if self.closed:
            return
        for reference in self.owners:
            module = reference()
            if module is not None and module._verification_scope is self:
                module._verification_scope = None
                if isinstance(module, VerificationProjection):
                    del module._verification_projection
                elif isinstance(module, VerificationExperts):
                    del module._verification_experts
        self.owners = ()
        self.plans = ()
        self.closed = True
