"""
Minimal experiment to test if transformer blocks can learn permutation undo/redo.

This isolates the permutation mechanism from language modeling:
- Single token input (no cross-token attention needed)
- Full transformer blocks but operating on sequence length 1
- Tests whether embeddings + N blocks can learn the bijection

Experiments:
A) Undo: (permuted_token, compartment) -> canonical_token
B) Redo: (canonical_token, compartment) -> permuted_token
"""

import argparse
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    vocab_size: int = 4096
    n_compartments: int = 4
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 1
    dropout: float = 0.0
    bias: bool = False


class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class SelfAttention(nn.Module):
    """Self-attention for single token (trivial but keeps architecture comparable)."""

    def __init__(self, config: Config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()  # T=1 for our case
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # Reshape for multi-head attention
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # Self-attention (trivial for T=1, but keeps the computation graph)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return self.dropout(x)


class Block(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = SelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class PermutationModel(nn.Module):
    """
    Model to learn permutation undo or redo.

    Input: token_id + compartment_id
    Output: target token_id (canonical or permuted depending on mode)
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # Token embedding
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        # Compartment embedding
        self.comp_emb = nn.Embedding(config.n_compartments, config.n_embd)

        # Transformer blocks
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)

        # Output head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)
        # Scale residual projections
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        n_params = sum(p.numel() for p in self.parameters())
        print(f"Model parameters: {n_params:,}")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, token_ids, compartment_ids, targets=None):
        """
        Args:
            token_ids: (B,) input token ids
            compartment_ids: (B,) compartment ids
            targets: (B,) target token ids (optional)
        """
        # Embed and add compartment info
        tok = self.tok_emb(token_ids)  # (B, n_embd)
        comp = self.comp_emb(compartment_ids)  # (B, n_embd)
        x = tok + comp

        # Add sequence dimension for transformer blocks
        x = x.unsqueeze(1)  # (B, 1, n_embd)

        # Forward through blocks
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)

        # Output logits
        logits = self.lm_head(x.squeeze(1))  # (B, vocab_size)

        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)

        return logits, loss


def generate_permutations(vocab_size: int, n_compartments: int, seed: int = 42) -> np.ndarray:
    """Generate random permutations for each compartment."""
    ss = np.random.SeedSequence(seed)
    child_seeds = ss.spawn(n_compartments)
    perms = np.empty((n_compartments, vocab_size), dtype=np.int64)
    for c, child_ss in enumerate(child_seeds):
        gen = np.random.Generator(np.random.PCG64(child_ss))
        perms[c] = gen.permutation(vocab_size)
    return perms


def generate_inverse_permutations(perms: np.ndarray) -> np.ndarray:
    """Generate inverse permutations from forward permutations."""
    n_compartments, vocab_size = perms.shape
    inv_perms = np.empty_like(perms)
    for c in range(n_compartments):
        inv_perms[c, perms[c]] = np.arange(vocab_size)
    return inv_perms


class PermutationDataset:
    """
    Dataset for permutation learning.

    Mode:
    - 'undo': input=permuted, target=canonical
    - 'redo': input=canonical, target=permuted
    """

    def __init__(
        self,
        vocab_size: int,
        n_compartments: int,
        perms: np.ndarray,
        inv_perms: np.ndarray,
        mode: str = "undo",
        seed: int = 0,
    ):
        self.vocab_size = vocab_size
        self.n_compartments = n_compartments
        self.perms = torch.from_numpy(perms)
        self.inv_perms = torch.from_numpy(inv_perms)
        self.mode = mode
        self.rng = np.random.Generator(np.random.PCG64(seed))

    def sample_batch(self, batch_size: int, device: torch.device):
        """Sample a batch of (input, compartment, target) tuples."""
        # Random canonical tokens
        canonical = self.rng.integers(0, self.vocab_size, size=batch_size)
        canonical = torch.from_numpy(canonical).long()

        # Random compartments
        compartments = self.rng.integers(0, self.n_compartments, size=batch_size)
        compartments = torch.from_numpy(compartments).long()

        # Apply permutation to get permuted tokens
        permuted = self.perms[compartments, canonical]

        if self.mode == "undo":
            # Input: permuted, Target: canonical
            inputs = permuted
            targets = canonical
        else:  # redo
            # Input: canonical, Target: permuted
            inputs = canonical
            targets = permuted

        return (
            inputs.to(device),
            compartments.to(device),
            targets.to(device),
        )


def train(
    config: Config,
    mode: str = "undo",
    n_steps: int = 10000,
    batch_size: int = 256,
    lr: float = 3e-4,
    eval_interval: int = 100,
    device: str = "cuda",
    seed: int = 42,
):
    """Train the permutation model."""
    torch.manual_seed(seed)
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Generate permutations
    perms = generate_permutations(config.vocab_size, config.n_compartments, seed)
    inv_perms = generate_inverse_permutations(perms)

    # Create dataset
    dataset = PermutationDataset(
        config.vocab_size, config.n_compartments, perms, inv_perms, mode, seed
    )

    # Create model
    model = PermutationModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    print(f"\nTraining {mode.upper()} direction")
    print(f"Config: vocab={config.vocab_size}, compartments={config.n_compartments}, "
          f"n_embd={config.n_embd}, n_layer={config.n_layer}, n_head={config.n_head}")
    print(f"Device: {device}")
    print("-" * 60)

    best_acc = 0.0
    for step in range(n_steps):
        model.train()
        inputs, compartments, targets = dataset.sample_batch(batch_size, device)

        logits, loss = model(inputs, compartments, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % eval_interval == 0 or step == n_steps - 1:
            model.eval()
            with torch.no_grad():
                # Larger eval batch for more stable metrics
                eval_inputs, eval_comps, eval_targets = dataset.sample_batch(
                    batch_size * 4, device
                )
                eval_logits, eval_loss = model(eval_inputs, eval_comps, eval_targets)
                preds = eval_logits.argmax(dim=-1)
                acc = (preds == eval_targets).float().mean().item()
                best_acc = max(best_acc, acc)

            print(f"Step {step:5d} | Loss: {eval_loss.item():.4f} | Acc: {acc*100:.2f}% | Best: {best_acc*100:.2f}%")

            # Early stopping if perfect
            if acc >= 0.9999:
                print(f"\nReached perfect accuracy at step {step}!")
                break

    return best_acc


def run_sweep(
    vocab_size: int = 4096,
    n_compartments: int = 4,
    n_embd: int = 128,
    n_head: int = 4,
    max_layers: int = 4,
    mode: str = "undo",
    n_steps: int = 20000,
    batch_size: int = 256,
    lr: float = 3e-4,
    device: str = "cuda",
    seed: int = 42,
):
    """Sweep over number of layers to find minimum needed."""
    print("=" * 60)
    print(f"SWEEP: {mode.upper()} direction")
    print(f"Vocab: {vocab_size}, Compartments: {n_compartments}")
    print(f"Embedding: {n_embd}, Heads: {n_head}")
    print("=" * 60)

    results = {}
    for n_layer in range(1, max_layers + 1):
        config = Config(
            vocab_size=vocab_size,
            n_compartments=n_compartments,
            n_embd=n_embd,
            n_head=n_head,
            n_layer=n_layer,
            dropout=0.0,
            bias=False,
        )
        best_acc = train(
            config,
            mode=mode,
            n_steps=n_steps,
            batch_size=batch_size,
            lr=lr,
            device=device,
            seed=seed,
        )
        results[n_layer] = best_acc
        print()

    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for n_layer, acc in results.items():
        status = "✓" if acc >= 0.9999 else "✗"
        print(f"  {n_layer} layer(s): {acc*100:.2f}% {status}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Permutation learning experiment")
    parser.add_argument("--mode", type=str, default="undo", choices=["undo", "redo", "both"],
                        help="Which direction to test")
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--n-compartments", type=int, default=4)
    parser.add_argument("--n-embd", type=int, default=128)
    parser.add_argument("--n-head", type=int, default=4)
    parser.add_argument("--max-layers", type=int, default=4)
    parser.add_argument("--n-steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.mode == "both":
        for mode in ["undo", "redo"]:
            run_sweep(
                vocab_size=args.vocab_size,
                n_compartments=args.n_compartments,
                n_embd=args.n_embd,
                n_head=args.n_head,
                max_layers=args.max_layers,
                mode=mode,
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                lr=args.lr,
                device=args.device,
                seed=args.seed,
            )
            print("\n" + "=" * 60 + "\n")
    else:
        run_sweep(
            vocab_size=args.vocab_size,
            n_compartments=args.n_compartments,
            n_embd=args.n_embd,
            n_head=args.n_head,
            max_layers=args.max_layers,
            mode=args.mode,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            seed=args.seed,
        )
