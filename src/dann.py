"""
Domain-Adversarial Neural Network (DANN) components for compartment-based training.

Provides a gradient reversal layer, per-layer domain discriminators, and a container
module that computes the adversarial domain classification loss.
"""

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn import functional as F


class GradientReversalFunction(Function):
    """Reverses gradients in the backward pass, scaled by lambda."""

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.save_for_backward(lambda_)
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        (lambda_,) = ctx.saved_tensors
        return -lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    """Module wrapper around GradientReversalFunction."""

    def forward(self, x, lambda_):
        return GradientReversalFunction.apply(x, lambda_)


class DomainDiscriminator(nn.Module):
    """Small MLP that classifies mean-pooled layer representations into domains.

    Architecture: LayerNorm(n_embd) -> Linear(n_embd, hidden) -> GELU -> Linear(hidden, n_domains)
    """

    def __init__(self, n_embd: int, n_domains: int, hidden: int = 0):
        super().__init__()
        hidden = hidden if hidden > 0 else n_embd
        self.net = nn.Sequential(
            nn.LayerNorm(n_embd),
            nn.Linear(n_embd, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_domains),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DANNModule(nn.Module):
    """Container holding one DomainDiscriminator per selected transformer layer.

    forward(layer_outputs, domain_labels, lambda_):
        1. For each selected layer: mean-pool over T -> GRL -> discriminator -> cross_entropy
        2. Returns mean of per-layer losses (scalar)
    """

    def __init__(self, layer_indices: list[int], n_embd: int, n_domains: int, hidden: int = 0):
        super().__init__()
        self.grl = GradientReversalLayer()
        self.discriminators = nn.ModuleDict(
            {str(i): DomainDiscriminator(n_embd, n_domains, hidden) for i in layer_indices}
        )

    def forward(
        self,
        layer_outputs: dict[int, torch.Tensor],
        domain_labels: torch.Tensor,
        lambda_: float,
    ) -> torch.Tensor:
        lambda_t = torch.tensor(lambda_, device=domain_labels.device, dtype=torch.float32)
        losses = []
        for layer_idx_str, disc in self.discriminators.items():
            layer_idx = int(layer_idx_str)
            h = layer_outputs[layer_idx]  # (B, T, n_embd)
            pooled = h.mean(dim=1)  # (B, n_embd)
            reversed_ = self.grl(pooled, lambda_t)
            logits = disc(reversed_)  # (B, n_domains)
            loss = F.cross_entropy(logits, domain_labels)
            losses.append(loss)
        return torch.stack(losses).mean()


def parse_dann_layers(spec: str, n_layer: int) -> list[int]:
    """Parse comma-separated layer indices like '2,4,6' and validate bounds.

    Returns an empty list if spec is empty.
    """
    spec = spec.strip()
    if not spec:
        return []
    indices = []
    for part in spec.split(","):
        part = part.strip()
        idx = int(part)
        if idx < 0 or idx >= n_layer:
            raise ValueError(
                f"DANN layer index {idx} out of range [0, {n_layer})"
            )
        indices.append(idx)
    return sorted(set(indices))
