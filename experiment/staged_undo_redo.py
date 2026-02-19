"""
Staged training of undo (first) and redo (last) layers.

Stage 1: Train undo layer to match baseline's first layer output (already done in permutation_undo_repr.py)
Stage 2: Plug in trained undo layer, freeze middle layers, train redo layer with cross-entropy

Architecture:
- Undo layer (trainable in stage 1, optionally trainable in stage 2): permuted + compartment → canonical repr
- Middle layers (frozen, from baseline): layers 1-6
- Redo layer (trainable in stage 2): canonical repr → permuted token logits
"""

import argparse
import glob
import math
import os
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import GPT, Block, LayerNorm
from src.config.job_config import Model as ModelConfig


# -----------------------------------------------------------------------------
# Data loading (same as permutation_undo_repr.py)
# -----------------------------------------------------------------------------

def _load_data_shard(filename: str) -> np.ndarray:
    """Load tokenized data from a .bin file."""
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        assert header[0] == 20251013, "magic number mismatch"
        assert header[1] == 1, "unsupported version"
        ntok = header[2]
        tokens = np.frombuffer(f.read(), dtype=np.uint32)
    assert len(tokens) == ntok
    return tokens


class FinewebDataLoader:
    """Optimized data loader for fineweb tokenized data with pinned memory."""

    def __init__(
        self,
        data_pattern: str,
        batch_size: int,
        block_size: int,
        seed: int = 0,
        pin_memory: bool = True,
    ):
        self.batch_size = batch_size
        self.block_size = block_size
        self.rng = np.random.Generator(np.random.PCG64(seed))
        self._pin = pin_memory and torch.cuda.is_available()

        self.files = sorted(glob.glob(data_pattern))
        assert len(self.files) > 0, f"No files found matching {data_pattern}"
        print(f"FinewebDataLoader: found {len(self.files)} data shards")

        self.current_shard_idx = 0
        self.tokens = _load_data_shard(self.files[0])
        self.position = 0

        # Pre-allocate pinned memory buffer for faster H2D transfers
        if self._pin:
            self._batch_buf = torch.empty((batch_size, block_size), dtype=torch.long, pin_memory=True)
        else:
            self._batch_buf = torch.empty((batch_size, block_size), dtype=torch.long)

    def _advance_shard(self):
        self.current_shard_idx = (self.current_shard_idx + 1) % len(self.files)
        self.tokens = _load_data_shard(self.files[self.current_shard_idx])
        self.position = 0

    def next_batch(self) -> torch.Tensor:
        B, T = self.batch_size, self.block_size
        batch_np = self._batch_buf.numpy()

        for i in range(B):
            if self.position + T >= len(self.tokens):
                self._advance_shard()
            max_start = len(self.tokens) - T - 1
            start = self.rng.integers(0, max_start)
            batch_np[i] = self.tokens[start : start + T]

        return self._batch_buf


def generate_permutations(vocab_size: int, max_compartments: int, seed: int = 42) -> np.ndarray:
    """Generate random permutations for each compartment."""
    ss = np.random.SeedSequence(seed)
    child_seeds = ss.spawn(max_compartments)
    perms = np.empty((max_compartments, vocab_size), dtype=np.int64)
    for c, child_ss in enumerate(child_seeds):
        gen = np.random.Generator(np.random.PCG64(child_ss))
        perms[c] = gen.permutation(vocab_size)
    return perms


class PermutedBatchProcessor:
    """Takes canonical tokens and applies permutations with pinned memory."""

    def __init__(self, n_compartments: int, perms: np.ndarray, batch_size: int, block_size: int, seed: int = 0, pin_memory: bool = True):
        self.n_compartments = n_compartments
        self.perms = torch.from_numpy(perms)
        self.rng = np.random.Generator(np.random.PCG64(seed))
        self._pin = pin_memory and torch.cuda.is_available()

        # Pre-allocate buffers
        if self._pin:
            self._permuted_buf = torch.empty((batch_size, block_size), dtype=torch.long, pin_memory=True)
            self._comp_buf = torch.empty((batch_size, block_size), dtype=torch.long, pin_memory=True)
        else:
            self._permuted_buf = torch.empty((batch_size, block_size), dtype=torch.long)
            self._comp_buf = torch.empty((batch_size, block_size), dtype=torch.long)

    def process_batch(self, canonical: torch.Tensor, device: torch.device):
        """
        Returns:
            canonical: (B, T) original tokens (on device)
            permuted: (B, T) permuted tokens (on device)
            compartment_ids: (B, T) compartment for each position (on device)
        """
        B, T = canonical.shape
        compartments = self.rng.integers(0, self.n_compartments, size=B)
        compartments = torch.from_numpy(compartments).long()
        compartment_ids = compartments.unsqueeze(1).expand(-1, T).contiguous()

        # Apply permutation into pre-allocated buffer
        permuted = self.perms[compartment_ids, canonical]
        self._permuted_buf.copy_(permuted)
        self._comp_buf.copy_(compartment_ids)

        return (
            canonical.to(device, non_blocking=True),
            self._permuted_buf.to(device, non_blocking=True),
            self._comp_buf.to(device, non_blocking=True),
        )


# -----------------------------------------------------------------------------
# Model components
# -----------------------------------------------------------------------------

@dataclass
class Config:
    vocab_size: int = 151936
    n_compartments: int = 4
    max_compartments: int = 16
    n_embd: int = 256
    n_head: int = 4
    n_layer: int = 1
    block_size: int = 64
    dropout: float = 0.0
    bias: bool = False


class UndoLayer(nn.Module):
    """
    First layer that undoes permutation.
    Takes permuted tokens + compartment IDs, outputs representation in canonical space.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.tok_emb = nn.Embedding(config.vocab_size + 1, config.n_embd)  # +1 for translation token
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.comp_emb = nn.Embedding(config.max_compartments, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

        block_config = type('BlockConfig', (), {
            'n_embd': config.n_embd,
            'n_head': config.n_head,
            'dropout': config.dropout,
            'bias': config.bias,
        })()

        self.blocks = nn.ModuleList([Block(block_config) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, token_ids: torch.Tensor, compartment_ids: torch.Tensor) -> torch.Tensor:
        B, T = token_ids.shape
        device = token_ids.device

        pos = torch.arange(0, T, dtype=torch.long, device=device)
        tok = self.tok_emb(token_ids)
        pos = self.pos_emb(pos)
        comp = self.comp_emb(compartment_ids)

        x = self.drop(tok + pos + comp)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return x


class RedoLayer(nn.Module):
    """
    Last layer that redoes permutation.
    Takes canonical representation, outputs logits over permuted vocabulary.
    Uses compartment info to know which permutation to apply.
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # Compartment embedding to add before the redo block
        self.comp_emb = nn.Embedding(config.max_compartments, config.n_embd)

        block_config = type('BlockConfig', (), {
            'n_embd': config.n_embd,
            'n_head': config.n_head,
            'dropout': config.dropout,
            'bias': config.bias,
        })()

        self.blocks = nn.ModuleList([Block(block_config) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)

        # Output head predicts permuted tokens
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size + 1, bias=False)

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, compartment_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, n_embd) representation from middle layers
            compartment_ids: (B, T) compartment IDs
        Returns:
            logits: (B, T, vocab_size+1)
        """
        # Add compartment info
        comp = self.comp_emb(compartment_ids)
        x = x + comp

        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits


class StagedModel(nn.Module):
    """
    Full model with undo layer, frozen middle, and redo layer.
    """

    def __init__(
        self,
        undo_layer: UndoLayer,
        middle_layers: nn.ModuleList,
        redo_layer: RedoLayer,
        freeze_middle: bool = True,
    ):
        super().__init__()
        self.undo_layer = undo_layer
        self.middle_layers = middle_layers
        self.redo_layer = redo_layer

        if freeze_middle:
            for p in self.middle_layers.parameters():
                p.requires_grad = False

    def forward(
        self,
        permuted_tokens: torch.Tensor,
        compartment_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            permuted_tokens: (B, T) input token IDs (permuted)
            compartment_ids: (B, T) compartment IDs
            targets: (B, T) target token IDs (permuted, shifted by 1)
        """
        # Undo: permuted → canonical representation
        x = self.undo_layer(permuted_tokens, compartment_ids)

        # Middle: process in canonical space
        for block in self.middle_layers:
            x = block(x)

        # Redo: canonical → permuted logits
        logits = self.redo_layer(x, compartment_ids)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss


def load_baseline_model(checkpoint_path: str, config: ModelConfig, device: torch.device) -> GPT:
    """Load the baseline model."""
    model = GPT(config)
    state_dict = torch.load(checkpoint_path, map_location=device)

    unwanted_prefix = "_orig_mod."
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_undo_layer_from_checkpoint(checkpoint_path: str, config: Config, device: torch.device) -> UndoLayer:
    """Load a trained undo layer."""
    undo = UndoLayer(config)
    state_dict = torch.load(checkpoint_path, map_location=device)
    undo.load_state_dict(state_dict)
    undo.to(device)
    return undo


class PerCompartmentEvaluator:
    """Evaluates loss per compartment and computes baseline floor."""

    def __init__(
        self,
        baseline_model: GPT,
        data_loader: FinewebDataLoader,
        perms: np.ndarray,
        n_compartments: int,
        batch_size: int,
        device: torch.device,
        autocast_ctx,
    ):
        self.baseline_model = baseline_model
        self.data_loader = data_loader
        self.perms_torch = torch.from_numpy(perms)
        self.n_compartments = n_compartments
        self.batch_size = batch_size
        self.device = device
        self.autocast_ctx = autocast_ctx

    def get_baseline_loss(self, n_batches: int = 10) -> float:
        """
        Compute baseline model's loss on compartment 0 (canonical) data.
        This is the floor we're trying to match.
        """
        self.baseline_model.eval()
        total_loss = 0.0

        with torch.no_grad(), self.autocast_ctx:
            for _ in range(n_batches):
                canonical = self.data_loader.next_batch().to(self.device)

                # Baseline uses compartment 0 and its permutation
                B, T = canonical.shape
                comp_ids = torch.zeros((B, T), dtype=torch.long, device=self.device)

                # Apply compartment 0's permutation
                permuted = self.perms_torch[comp_ids.cpu(), canonical.cpu()].to(self.device)

                # Targets: next permuted token
                targets = torch.empty_like(permuted)
                targets[:, :-1] = permuted[:, 1:]
                targets[:, -1] = -1

                # Run through baseline
                logits, loss = self.baseline_model(permuted, targets, compartment_ids=comp_ids)
                total_loss += loss.item()

        return total_loss / n_batches

    def get_per_compartment_loss(
        self,
        model: nn.Module,
        n_batches_per_compartment: int = 5,
    ) -> dict[int, float]:
        """
        Compute loss for each compartment separately.
        """
        model.eval()
        losses = {c: 0.0 for c in range(self.n_compartments)}

        with torch.no_grad(), self.autocast_ctx:
            for comp_id in range(self.n_compartments):
                for _ in range(n_batches_per_compartment):
                    canonical = self.data_loader.next_batch()
                    B, T = canonical.shape

                    # Force this compartment for all samples
                    comp_ids = torch.full((B, T), comp_id, dtype=torch.long)

                    # Apply this compartment's permutation
                    permuted = self.perms_torch[comp_ids, canonical].to(self.device)
                    comp_ids = comp_ids.to(self.device)

                    # Targets
                    targets = torch.empty_like(permuted)
                    targets[:, :-1] = permuted[:, 1:]
                    targets[:, -1] = -1

                    _, loss = model(permuted, comp_ids, targets)
                    losses[comp_id] += loss.item()

                losses[comp_id] /= n_batches_per_compartment

        return losses


def train_redo(
    baseline_checkpoint: str,
    undo_checkpoint: str,
    data_pattern: str,
    n_compartments: int = 4,
    n_redo_layers: int = 1,
    train_undo: bool = True,
    n_steps: int = 10000,
    batch_size: int = 128,
    gradient_accumulation_steps: int = 16,
    lr: float = 3e-4,
    eval_interval: int = 100,
    device: str = "cuda",
    seed: int = 42,
    plot_output: Optional[str] = None,
    use_compile: bool = True,
):
    """Train the redo layer (and optionally fine-tune undo layer)."""
    torch.manual_seed(seed)
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    use_amp = device.type == "cuda"
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else nullcontext()

    # Config
    config = Config(
        vocab_size=151936,
        n_compartments=n_compartments,
        max_compartments=16,
        n_embd=256,
        n_head=4,
        n_layer=1,  # For undo/redo layers
        block_size=64,
        dropout=0.0,
        bias=False,
    )

    # Load baseline model to extract middle layers
    baseline_config = ModelConfig(
        n_layer=8,
        n_head=4,
        n_embd=256,
        block_size=64,
        vocab_size=151936 + 1,
        weight_tying=False,
        bias=False,
        dropout=0.0,
        embedding_vocab_size=151936 + 1,
        shared_token_embeddings=False,
        use_compartment_embeddings=True,
        base_vocab_size=151936,
        max_compartments=16,
        translation_token_id=151936,
    )

    print(f"Loading baseline model from {baseline_checkpoint}")
    baseline_model = load_baseline_model(baseline_checkpoint, baseline_config, device)

    # Extract middle layers (layers 1-6, i.e., indices 1 to n_layer-2)
    # Baseline has 8 layers (0-7), we want layers 1-6
    middle_layers = nn.ModuleList([
        baseline_model.transformer.h[i] for i in range(1, 7)
    ])
    print(f"Extracted {len(middle_layers)} middle layers from baseline")

    # Load trained undo layer
    print(f"Loading undo layer from {undo_checkpoint}")
    undo_layer = load_undo_layer_from_checkpoint(undo_checkpoint, config, device)

    # Create redo layer
    redo_config = Config(
        vocab_size=151936,
        n_compartments=n_compartments,
        max_compartments=16,
        n_embd=256,
        n_head=4,
        n_layer=n_redo_layers,
        block_size=64,
        dropout=0.0,
        bias=False,
    )
    redo_layer = RedoLayer(redo_config).to(device)

    # Build staged model
    staged_model = StagedModel(
        undo_layer=undo_layer,
        middle_layers=middle_layers,
        redo_layer=redo_layer,
        freeze_middle=True,
    )
    staged_model.to(device)

    # Compile if requested
    if use_compile and hasattr(torch, 'compile'):
        print("Compiling model with torch.compile()...")
        staged_model = torch.compile(staged_model)

    # Set up optimizer - only train undo (if enabled) and redo
    params_to_train = list(redo_layer.parameters())
    if train_undo:
        params_to_train.extend(undo_layer.parameters())
        print("Training both undo and redo layers")
    else:
        for p in undo_layer.parameters():
            p.requires_grad = False
        print("Training only redo layer (undo frozen)")

    optimizer = torch.optim.AdamW(params_to_train, lr=lr)

    # Data loading
    perms = generate_permutations(151936, 16, seed=64)

    data_loader = FinewebDataLoader(
        data_pattern=data_pattern,
        batch_size=batch_size,
        block_size=64,
        seed=seed,
    )

    batch_processor = PermutedBatchProcessor(
        n_compartments=n_compartments,
        perms=perms,
        batch_size=batch_size,
        block_size=64,
        seed=seed + 1000,
    )

    # Set up per-compartment evaluator
    eval_data_loader = FinewebDataLoader(
        data_pattern=data_pattern,
        batch_size=batch_size,
        block_size=64,
        seed=seed + 5000,  # Different seed for eval
    )

    evaluator = PerCompartmentEvaluator(
        baseline_model=baseline_model,
        data_loader=eval_data_loader,
        perms=perms,
        n_compartments=n_compartments,
        batch_size=batch_size,
        device=device,
        autocast_ctx=autocast_ctx,
    )

    # Compute baseline floor
    print("Computing baseline floor...")
    baseline_floor = evaluator.get_baseline_loss(n_batches=20)
    print(f"Baseline floor (compartment 0): {baseline_floor:.4f}")

    effective_batch_size = batch_size * gradient_accumulation_steps
    print(f"\nTraining REDO layer")
    print(f"Compartments: {n_compartments}, Redo layers: {n_redo_layers}")
    print(f"Micro batch: {batch_size}, Grad accum: {gradient_accumulation_steps}, Effective batch: {effective_batch_size}")
    print(f"Device: {device}, Compile: {use_compile}")
    print("-" * 60)

    # Track metrics for plotting
    history = {
        'steps': [],
        'overall_loss': [],
        'per_compartment_loss': {c: [] for c in range(n_compartments)},
        'baseline_floor': baseline_floor,
    }

    best_loss = float('inf')
    for step in range(n_steps):
        staged_model.train()
        optimizer.zero_grad()

        # Gradient accumulation loop
        accum_loss = 0.0
        for micro_step in range(gradient_accumulation_steps):
            # Get batch
            canonical_batch = data_loader.next_batch()
            canonical, permuted, compartment_ids = batch_processor.process_batch(
                canonical_batch, device
            )

            # Targets: next permuted token
            targets = torch.empty_like(permuted)
            targets[:, :-1] = permuted[:, 1:]
            targets[:, -1] = -1  # ignore last position

            with autocast_ctx:
                logits, loss = staged_model(permuted, compartment_ids, targets)
                # Scale loss for gradient accumulation
                loss = loss / gradient_accumulation_steps

            loss.backward()
            accum_loss += loss.item()

        optimizer.step()

        if step % eval_interval == 0 or step == n_steps - 1:
            staged_model.eval()

            # Overall eval
            with torch.no_grad(), autocast_ctx:
                eval_canonical_batch = eval_data_loader.next_batch()
                eval_canonical, eval_permuted, eval_comp_ids = batch_processor.process_batch(
                    eval_canonical_batch, device
                )
                eval_targets = torch.empty_like(eval_permuted)
                eval_targets[:, :-1] = eval_permuted[:, 1:]
                eval_targets[:, -1] = -1

                eval_logits, eval_loss = staged_model(eval_permuted, eval_comp_ids, eval_targets)
                eval_loss_val = eval_loss.item()

                # Compute accuracy (excluding ignored positions)
                preds = eval_logits.argmax(dim=-1)
                mask = eval_targets != -1
                acc = (preds[mask] == eval_targets[mask]).float().mean().item()

                best_loss = min(best_loss, eval_loss_val)

            # Per-compartment eval
            per_comp_losses = evaluator.get_per_compartment_loss(staged_model, n_batches_per_compartment=3)

            # Log
            history['steps'].append(step)
            history['overall_loss'].append(eval_loss_val)
            for c, loss_c in per_comp_losses.items():
                history['per_compartment_loss'][c].append(loss_c)

            # Print
            comp_str = " | ".join([f"C{c}: {per_comp_losses[c]:.4f}" for c in range(n_compartments)])
            print(f"Step {step:5d} | Loss: {eval_loss_val:.4f} | Acc: {acc*100:.2f}% | {comp_str} | Floor: {baseline_floor:.4f}")

    # Save plot data and generate plot
    if plot_output:
        import json
        import matplotlib.pyplot as plt

        # Save data
        data_path = plot_output.replace('.png', '.json')
        with open(data_path, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"Saved plot data to {data_path}")

        # Generate plot
        plt.figure(figsize=(10, 6))

        # Plot per-compartment losses
        colors = plt.cm.tab10(np.linspace(0, 1, n_compartments))
        for c in range(n_compartments):
            plt.plot(
                history['steps'],
                history['per_compartment_loss'][c],
                label=f'Compartment {c}',
                color=colors[c],
                linewidth=2,
            )

        # Plot baseline floor
        plt.axhline(
            y=baseline_floor,
            color='black',
            linestyle='--',
            linewidth=2,
            label=f'Baseline floor ({baseline_floor:.4f})',
        )

        plt.xlabel('Step')
        plt.ylabel('Loss (Cross-Entropy)')
        plt.title(f'Per-Compartment Validation Loss ({n_compartments} compartments)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig(plot_output, dpi=150)
        print(f"Saved plot to {plot_output}")
        plt.close()

    return best_loss, staged_model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Staged undo/redo training")
    parser.add_argument(
        "--baseline-checkpoint",
        type=str,
        default="/mnt/pccfs2/backed_up/vin/dev/translation-compression/out/translation-compression/dw-2-1-dry-runs/2025-10-27T18-46-17Z__1-domain-baseline-fineweb-8-256-prototype__fb319b5c__s64__8000b93__a48a5009/checkpoints/step-1000000/model.pt",
    )
    parser.add_argument(
        "--undo-checkpoint",
        type=str,
        required=True,
        help="Path to trained undo layer checkpoint",
    )
    parser.add_argument(
        "--data-pattern",
        type=str,
        default="data/fineweb350B-dedup-suffix-31/fineweb350b-dedup_train_*.bin",
    )
    parser.add_argument("--n-compartments", type=int, default=4)
    parser.add_argument("--n-redo-layers", type=int, default=1)
    parser.add_argument("--train-undo", action="store_true",
                        help="Also fine-tune the undo layer during redo training")
    parser.add_argument("--n-steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Micro batch size per gradient accumulation step")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16,
                        help="Number of micro-batches to accumulate before optimizer step")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot-output", type=str, default=None,
                        help="Path to save plot (e.g., results.png). Also saves .json with data.")
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile() for debugging")
    args = parser.parse_args()

    train_redo(
        baseline_checkpoint=args.baseline_checkpoint,
        undo_checkpoint=args.undo_checkpoint,
        data_pattern=args.data_pattern,
        n_compartments=args.n_compartments,
        n_redo_layers=args.n_redo_layers,
        train_undo=args.train_undo,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
        plot_output=args.plot_output,
        use_compile=not args.no_compile,
    )
