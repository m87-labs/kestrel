"""Model-owned fixed-reduction projections for explicitly native verification."""
from contextlib import contextmanager
from contextvars import ContextVar
import weakref
from torch import nn
from torch.nn import functional as F


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


class VerificationLinear(nn.Linear, VerificationProjection):
    def forward(self, x):
        return self.project(x, self.bias)


class NativeVerificationProjections:
    """Own immutable weights and a task-local activation scope for one model.

    Prepare after loading/materialization and before graph capture. Other models
    have separate context variables and cannot observe this owner's activation.
    """
    def __init__(self, modules):
        from kestrel_kernels.verification_dense import PreparedDense
        self.closed = False
        self.active = ContextVar(f'verification-projections-{id(self)}', default=False)
        owners = tuple(module for module in modules
                       if isinstance(module, VerificationProjection))
        if any(module._verification_scope is not None for module in owners):
            raise ValueError('verification projections already belong to a runtime')
        # Validate complete inventory before publishing any owner binding.
        plans = tuple(PreparedDense(module.weight) for module in owners)
        self.owners = tuple(weakref.ref(module) for module in owners)
        self.plans = plans
        for module, plan in zip(owners, plans, strict=True):
            module._verification_projection = plan
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
                del module._verification_projection
        self.owners = ()
        self.plans = ()
        self.closed = True
