"""
Representation-matching experiment for permutation undo.

Tests whether N transformer blocks can learn to map permuted tokens + compartment
embeddings into the same representation space as the baseline model's first layer.

Input: permuted token sequence + compartment IDs
Target: first-layer output of baseline model on canonical (unpermuted) tokens
Loss: MSE between predicted and target representations

Uses real tokenized fineweb data, not random tokens.
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
# Data loading (adapted from train.py)
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
    """
    Simple data loader for fineweb tokenized data.
    Returns batches of canonical (unpermuted) token sequences.
    """

    def __init__(
        self,
        data_pattern: str,
        batch_size: int,
        block_size: int,
        seed: int = 0,
    ):
        self.batch_size = batch_size
        self.block_size = block_size
        self.rng = np.random.Generator(np.random.PCG64(seed))

        # Find all data shards
        self.files = sorted(glob.glob(data_pattern))
        assert len(self.files) > 0, f"No files found matching {data_pattern}"
        print(f"FinewebDataLoader: found {len(self.files)} data shards")

        # Load first shard
        self.current_shard_idx = 0
        self.tokens = _load_data_shard(self.files[0])
        self.position = 0

    def _advance_shard(self):
        """Move to next data shard."""
        self.current_shard_idx = (self.current_shard_idx + 1) % len(self.files)
        self.tokens = _load_data_shard(self.files[self.current_shard_idx])
        self.position = 0

    def next_batch(self) -> torch.Tensor:
        """
        Get next batch of canonical token sequences.

        Returns:
            (B, T) tensor of token IDs
        """
        B, T = self.batch_size, self.block_size
        batch = np.empty((B, T), dtype=np.int64)

        for i in range(B):
            # Check if we need to advance shard
            if self.position + T >= len(self.tokens):
                self._advance_shard()

            # Sample a random position in current shard
            max_start = len(self.tokens) - T - 1
            start = self.rng.integers(0, max_start)
            batch[i] = self.tokens[start : start + T]

        return torch.from_numpy(batch).long()


@dataclass
class UndoConfig:
    """Config for the undo model."""
    vocab_size: int = 151936
    n_compartments: int = 4
    max_compartments: int = 16
    n_embd: int = 256
    n_head: int = 4
    n_layer: int = 1  # Number of transformer blocks in undo model
    block_size: int = 64
    dropout: float = 0.0
    bias: bool = False


class UndoModel(nn.Module):
    """
    Model to undo permutation and produce baseline-compatible representations.

    Takes permuted tokens + compartment IDs, outputs representations that should
    match the baseline model's first-layer output on canonical tokens.
    """

    def __init__(self, config: UndoConfig):
        super().__init__()
        self.config = config

        # Token embedding (for permuted tokens)
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        # Position embedding
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        # Compartment embedding
        self.comp_emb = nn.Embedding(config.max_compartments, config.n_embd)

        self.drop = nn.Dropout(config.dropout)

        # Create a minimal config for the Block class
        block_config = type('BlockConfig', (), {
            'n_embd': config.n_embd,
            'n_head': config.n_head,
            'dropout': config.dropout,
            'bias': config.bias,
        })()

        # Transformer blocks
        self.blocks = nn.ModuleList([Block(block_config) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)

        # Initialize weights
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        n_params = sum(p.numel() for p in self.parameters())
        print(f"UndoModel parameters: {n_params:,}")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, token_ids: torch.Tensor, compartment_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            token_ids: (B, T) permuted token IDs
            compartment_ids: (B, T) compartment IDs for each position

        Returns:
            (B, T, n_embd) representation that should match baseline first-layer output
        """
        B, T = token_ids.shape
        device = token_ids.device

        pos = torch.arange(0, T, dtype=torch.long, device=device)

        tok = self.tok_emb(token_ids)  # (B, T, n_embd)
        pos = self.pos_emb(pos)  # (T, n_embd)
        comp = self.comp_emb(compartment_ids)  # (B, T, n_embd)

        x = self.drop(tok + pos + comp)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        return x


def load_baseline_model(checkpoint_path: str, config: ModelConfig, device: torch.device) -> GPT:
    """Load and freeze the baseline model."""
    model = GPT(config)
    state_dict = torch.load(checkpoint_path, map_location=device)

    # Handle potential _orig_mod. prefix from torch.compile
    unwanted_prefix = "_orig_mod."
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Freeze all parameters
    for p in model.parameters():
        p.requires_grad = False

    return model


def get_baseline_layer_output(
    model: GPT,
    token_ids: torch.Tensor,
    compartment_ids: Optional[torch.Tensor] = None,
    target_layer: int = 0,
) -> torch.Tensor:
    """
    Run tokens through baseline model and return output after specified transformer block.

    Args:
        model: The baseline GPT model
        token_ids: (B, T) token IDs
        compartment_ids: (B, T) compartment IDs (optional)
        target_layer: Which layer's output to return (0 = first block, 1 = second, etc.)
    """
    device = token_ids.device
    B, T = token_ids.shape

    pos = torch.arange(0, T, dtype=torch.long, device=device)

    # Get embeddings (same as model.forward but we stop early)
    if model.shared_token_embeddings:
        base_vocab = model.base_vocab_size
        trans_id = model.translation_token_id
        emb_vocab_last = model.embedding_vocab_size - 1
        idx_mod = torch.remainder(token_ids, base_vocab)
        idx_mod = torch.where(
            token_ids == trans_id,
            torch.full_like(token_ids, emb_vocab_last),
            idx_mod,
        )
        tok_emb = model.transformer.wte(idx_mod)
    else:
        tok_emb = model.transformer.wte(token_ids)

    pos_emb = model.transformer.wpe(pos)
    x = tok_emb + pos_emb

    if model.use_compartment_embeddings and compartment_ids is not None:
        comp_emb = model.comp_emb(compartment_ids)
        x = x + comp_emb

    x = model.transformer.drop(x)

    # Run through blocks up to and including target_layer
    for i in range(target_layer + 1):
        x = model.transformer.h[i](x)

    return x


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
    """
    Takes canonical tokens from data loader and applies permutations.
    """

    def __init__(
        self,
        n_compartments: int,
        perms: np.ndarray,
        seed: int = 0,
    ):
        self.n_compartments = n_compartments
        self.perms = torch.from_numpy(perms)
        self.rng = np.random.Generator(np.random.PCG64(seed))

    def process_batch(self, canonical: torch.Tensor, device: torch.device):
        """
        Apply permutations to canonical tokens.

        Args:
            canonical: (B, T) canonical token IDs from data loader

        Returns:
            canonical_tokens: (B, T) - original unpermuted tokens (on device)
            permuted_tokens: (B, T) - tokens after applying compartment permutation
            compartment_ids: (B, T) - which compartment each sequence belongs to
        """
        B, T = canonical.shape

        # Random compartment per sequence (same compartment for whole sequence)
        compartments = self.rng.integers(0, self.n_compartments, size=B)
        compartments = torch.from_numpy(compartments).long()

        # Expand compartments to (B, T)
        compartment_ids = compartments.unsqueeze(1).expand(-1, T).contiguous()

        # Apply permutation: permuted[b, t] = perms[compartment[b], canonical[b, t]]
        permuted = self.perms[compartment_ids, canonical]

        return (
            canonical.to(device),
            permuted.to(device),
            compartment_ids.to(device),
        )


def train(
    baseline_checkpoint: str,
    data_pattern: str,
    n_compartments: int = 4,
    n_undo_layers: int = 1,
    target_layer: int = 0,
    n_steps: int = 10000,
    batch_size: int = 64,
    lr: float = 3e-4,
    eval_interval: int = 100,
    device: str = "cuda",
    seed: int = 42,
    save_checkpoint: Optional[str] = None,
):
    """Train the undo model to match baseline layer representations."""
    torch.manual_seed(seed)
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Use autocast for flash attention compatibility
    use_amp = device.type == "cuda"
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else nullcontext()

    # Baseline model config (8-256 fineweb)
    baseline_config = ModelConfig(
        n_layer=8,
        n_head=4,
        n_embd=256,
        block_size=64,
        vocab_size=151936,  # Will be overwritten to composite vocab
        weight_tying=False,
        bias=False,
        dropout=0.0,
        # Advanced options for compartment support
        embedding_vocab_size=151936 + 1,  # base + translation token
        shared_token_embeddings=False,
        use_compartment_embeddings=True,
        base_vocab_size=151936,
        max_compartments=16,
        translation_token_id=151936,
    )
    # The actual vocab size for the baseline (with permutation mode)
    baseline_config = ModelConfig(
        **{**baseline_config.__dict__, 'vocab_size': 151936 + 1}
    )

    print(f"Loading baseline model from {baseline_checkpoint}")
    baseline_model = load_baseline_model(baseline_checkpoint, baseline_config, device)
    print(f"Baseline model loaded ({sum(p.numel() for p in baseline_model.parameters()):,} params, frozen)")

    # Undo model config
    undo_config = UndoConfig(
        vocab_size=151936 + 1,  # Same as baseline
        n_compartments=n_compartments,
        max_compartments=16,
        n_embd=256,
        n_head=4,
        n_layer=n_undo_layers,
        block_size=64,
        dropout=0.0,
        bias=False,
    )

    undo_model = UndoModel(undo_config).to(device)
    optimizer = torch.optim.AdamW(undo_model.parameters(), lr=lr)

    # Generate permutations (same seed as training would use)
    perms = generate_permutations(151936, 16, seed=64)  # Using training seed

    # Data loader for real fineweb data
    data_loader = FinewebDataLoader(
        data_pattern=data_pattern,
        batch_size=batch_size,
        block_size=64,
        seed=seed,
    )

    # Processor to apply permutations
    batch_processor = PermutedBatchProcessor(
        n_compartments=n_compartments,
        perms=perms,
        seed=seed + 1000,  # Different seed for compartment assignment
    )

    print(f"\nTraining UNDO model")
    print(f"Compartments: {n_compartments}, Undo layers: {n_undo_layers}, Target layer: {target_layer}")
    print(f"Device: {device}")
    print("-" * 60)

    best_loss = float('inf')
    for step in range(n_steps):
        undo_model.train()

        # Get canonical tokens from real data
        canonical_batch = data_loader.next_batch()

        # Apply permutations
        canonical, permuted, compartment_ids = batch_processor.process_batch(
            canonical_batch, device
        )

        # Get baseline layer output on canonical tokens (target)
        # Note: baseline was trained with compartment 0, so we pass zeros
        with torch.no_grad(), autocast_ctx:
            baseline_comp_ids = torch.zeros_like(compartment_ids)
            target_repr = get_baseline_layer_output(
                baseline_model, canonical, baseline_comp_ids, target_layer
            )

        # Get undo model output on permuted tokens
        with autocast_ctx:
            pred_repr = undo_model(permuted, compartment_ids)

            # MSE loss (compute in autocast context for consistency)
            loss = F.mse_loss(pred_repr, target_repr)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % eval_interval == 0 or step == n_steps - 1:
            undo_model.eval()
            with torch.no_grad(), autocast_ctx:
                # Eval on fresh batch
                eval_canonical_batch = data_loader.next_batch()
                eval_canonical, eval_permuted, eval_comp_ids = batch_processor.process_batch(
                    eval_canonical_batch, device
                )
                eval_baseline_comp = torch.zeros_like(eval_comp_ids)
                eval_target = get_baseline_layer_output(
                    baseline_model, eval_canonical, eval_baseline_comp, target_layer
                )
                eval_pred = undo_model(eval_permuted, eval_comp_ids)
                eval_loss = F.mse_loss(eval_pred, eval_target).item()

                # Also compute cosine similarity as a metric
                # Flatten to (B*T, n_embd) and compute mean cosine sim
                pred_flat = eval_pred.view(-1, eval_pred.size(-1))
                target_flat = eval_target.view(-1, eval_target.size(-1))
                cos_sim = F.cosine_similarity(pred_flat, target_flat, dim=-1).mean().item()

                best_loss = min(best_loss, eval_loss)

            print(f"Step {step:5d} | MSE: {eval_loss:.6f} | CosSim: {cos_sim:.4f} | Best MSE: {best_loss:.6f}")

    # Save checkpoint if requested
    if save_checkpoint:
        os.makedirs(os.path.dirname(save_checkpoint) or ".", exist_ok=True)
        torch.save(undo_model.state_dict(), save_checkpoint)
        print(f"Saved undo model to {save_checkpoint}")

    return best_loss, undo_model


def run_sweep(
    baseline_checkpoint: str,
    data_pattern: str,
    n_compartments: int = 4,
    max_layers: int = 3,
    target_layer: int = 0,
    n_steps: int = 20000,
    batch_size: int = 64,
    lr: float = 3e-4,
    device: str = "cuda",
    seed: int = 42,
    save_dir: Optional[str] = None,
):
    """Sweep over number of undo layers."""
    print("=" * 60)
    print(f"SWEEP: Representation Undo")
    print(f"Compartments: {n_compartments}")
    print(f"Target baseline layer: {target_layer}")
    print(f"Baseline: {baseline_checkpoint}")
    print(f"Data: {data_pattern}")
    print("=" * 60)

    results = {}
    for n_layer in range(1, max_layers + 1):
        print(f"\n{'='*60}")
        print(f"Testing {n_layer} undo layer(s)")
        print("=" * 60)

        save_checkpoint = None
        if save_dir:
            save_checkpoint = os.path.join(
                save_dir, f"undo_layer_{n_layer}L_target{target_layer}_comp{n_compartments}.pt"
            )

        best_loss, _ = train(
            baseline_checkpoint=baseline_checkpoint,
            data_pattern=data_pattern,
            n_compartments=n_compartments,
            n_undo_layers=n_layer,
            target_layer=target_layer,
            n_steps=n_steps,
            batch_size=batch_size,
            lr=lr,
            device=device,
            seed=seed,
            save_checkpoint=save_checkpoint,
        )
        results[n_layer] = best_loss
        print()

    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for n_layer, loss in results.items():
        print(f"  {n_layer} layer(s): MSE = {loss:.6f}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Representation undo experiment")
    parser.add_argument(
        "--baseline-checkpoint",
        type=str,
        default="/mnt/pccfs2/backed_up/vin/dev/translation-compression/out/translation-compression/dw-2-1-dry-runs/2025-10-27T18-46-17Z__1-domain-baseline-fineweb-8-256-prototype__fb319b5c__s64__8000b93__a48a5009/checkpoints/step-1000000/model.pt",
        help="Path to baseline model checkpoint",
    )
    parser.add_argument(
        "--data-pattern",
        type=str,
        default="data/fineweb350B-dedup-suffix-31/fineweb350b-dedup_train_*.bin",
        help="Glob pattern for tokenized data files",
    )
    parser.add_argument("--n-compartments", type=int, default=4)
    parser.add_argument("--max-layers", type=int, default=3)
    parser.add_argument("--target-layer", type=int, default=0,
                        help="Which baseline layer's output to match (0 = first block)")
    parser.add_argument("--n-steps", type=int, default=20000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Directory to save trained undo layer checkpoints")
    args = parser.parse_args()

    run_sweep(
        baseline_checkpoint=args.baseline_checkpoint,
        data_pattern=args.data_pattern,
        n_compartments=args.n_compartments,
        max_layers=args.max_layers,
        target_layer=args.target_layer,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
        save_dir=args.save_dir,
    )
