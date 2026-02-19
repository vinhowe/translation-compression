"""
Fine-tune only first and last layers of baseline model on compartmentalized data.

Freezes middle layers (1 to n_layer-2) and trains:
- Layer 0 (first transformer block)
- Layer n_layer-1 (last transformer block)
- Token embeddings (wte)
- Position embeddings (wpe)
- Compartment embeddings (comp_emb)
- Final layer norm (ln_f)
- LM head

This tests if first/last layers can learn to undo/redo permutations while
the frozen middle layers process in a "canonical" representation space.
"""

import argparse
import glob
import os
import sys
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import GPT
from src.config.job_config import Model as ModelConfig


# -----------------------------------------------------------------------------
# Data loading
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
    """Data loader for fineweb tokenized data with pinned memory."""

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
    """Takes canonical tokens and applies permutations on GPU."""

    def __init__(
        self,
        n_compartments: int,
        perms: np.ndarray,
        batch_size: int,
        block_size: int,
        device: torch.device,
        seed: int = 0,
    ):
        self.n_compartments = n_compartments
        # Keep permutations on GPU for fast indexing
        self.perms = torch.from_numpy(perms).to(device)
        self.rng = np.random.Generator(np.random.PCG64(seed))
        self.device = device

    def process_batch(self, canonical: torch.Tensor):
        """Apply random compartment permutation to each sample (on GPU)."""
        B, T = canonical.shape
        # Generate compartment assignments
        compartments = self.rng.integers(0, self.n_compartments, size=B)
        compartments = torch.from_numpy(compartments).long().to(self.device)
        compartment_ids = compartments.unsqueeze(1).expand(-1, T)

        # Move canonical to GPU first, then do permutation indexing on GPU
        canonical_gpu = canonical.to(self.device, non_blocking=True)
        permuted = self.perms[compartment_ids, canonical_gpu]

        return permuted, compartment_ids


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

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
    return model


def freeze_middle_layers(model: GPT, n_first_layers: int = 1, n_last_layers: int = 1):
    """
    Freeze middle transformer blocks, keeping first N and last M trainable.

    Args:
        model: The GPT model
        n_first_layers: Number of layers at the start to keep trainable (default 1)
        n_last_layers: Number of layers at the end to keep trainable (default 1)
    """
    n_layers = len(model.transformer.h)
    frozen_count = 0
    trainable_count = 0

    trainable_indices = set(range(n_first_layers)) | set(range(n_layers - n_last_layers, n_layers))
    frozen_indices = set(range(n_layers)) - trainable_indices

    for i, block in enumerate(model.transformer.h):
        if i in trainable_indices:
            for p in block.parameters():
                p.requires_grad = True
                trainable_count += p.numel()
        else:
            for p in block.parameters():
                p.requires_grad = False
                frozen_count += p.numel()

    print(f"Frozen layers: {sorted(frozen_indices)} ({frozen_count:,} params)")
    print(f"Trainable layers: {sorted(trainable_indices)} ({trainable_count:,} params)")

    # Embeddings, ln_f, lm_head are all trainable
    embed_params = 0
    for name in ['wte', 'wpe', 'drop']:
        if hasattr(model.transformer, name):
            for p in model.transformer[name].parameters():
                p.requires_grad = True
                embed_params += p.numel()

    if model.comp_emb is not None:
        for p in model.comp_emb.parameters():
            p.requires_grad = True
            embed_params += p.numel()

    for p in model.transformer.ln_f.parameters():
        p.requires_grad = True
        embed_params += p.numel()

    for p in model.lm_head.parameters():
        p.requires_grad = True
        embed_params += p.numel()

    print(f"Trainable embeddings/head: {embed_params:,} params")

    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total trainable: {total_trainable:,} / {total_params:,} ({100*total_trainable/total_params:.1f}%)")


def train(
    baseline_checkpoint: str,
    data_pattern: str,
    n_compartments: int = 2,
    n_first_layers: int = 1,
    n_last_layers: int = 1,
    n_steps: int = 500000,
    batch_size: int = 512,
    gradient_accumulation_steps: int = 4,
    lr: float = 2e-5,
    warmup_iters: int = 1000,
    weight_decay: float = 0.0,
    grad_clip: float = 1.0,
    log_interval: int = 10,
    eval_interval: int = 500,
    save_interval: int = 50000,
    save_dir: str = "experiment/finetune_checkpoints",
    device: str = "cuda",
    seed: int = 64,
    use_compile: bool = True,
):
    """Fine-tune first and last layers on compartmentalized data."""
    torch.manual_seed(seed)
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    use_amp = device.type == "cuda"
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else nullcontext()

    # Model config (matching baseline)
    config = ModelConfig(
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

    # Load baseline model
    print(f"Loading baseline model from {baseline_checkpoint}")
    model = load_baseline_model(baseline_checkpoint, config, device)

    # Freeze middle layers
    print("\nFreezing middle layers...")
    freeze_middle_layers(model, n_first_layers=n_first_layers, n_last_layers=n_last_layers)

    # Compile if requested
    if use_compile and hasattr(torch, 'compile'):
        print("\nCompiling model with torch.compile()...")
        model = torch.compile(model)

    # Get raw model for configure_optimizers (handles compiled model)
    raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model

    # Optimizer using model's configure_optimizers for proper weight decay handling
    # This separates decay (2D+ params) from no-decay (bias, LayerNorm)
    optimizer = raw_model.configure_optimizers(
        weight_decay=weight_decay,
        learning_rate=lr,
        betas=(0.9, 0.999),
        device_type=device.type,
    )

    # LR scheduler helper (warmup then constant)
    def get_lr(it):
        if it < warmup_iters:
            return lr * (it + 1) / (warmup_iters + 1)
        return lr

    # Data
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
        device=device,
        seed=seed + 1000,
    )

    # Eval data loader (separate seed)
    eval_data_loader = FinewebDataLoader(
        data_pattern=data_pattern,
        batch_size=batch_size,
        block_size=64,
        seed=seed + 5000,
    )

    eval_batch_processor = PermutedBatchProcessor(
        n_compartments=n_compartments,
        perms=perms,
        batch_size=batch_size,
        block_size=64,
        device=device,
        seed=seed + 6000,
    )

    # Create save dir
    os.makedirs(save_dir, exist_ok=True)

    effective_batch_size = batch_size * gradient_accumulation_steps
    tokens_per_step = effective_batch_size * 64  # block_size=64
    print(f"\nTraining first/last layers")
    print(f"Compartments: {n_compartments}")
    print(f"Micro batch: {batch_size}, Grad accum: {gradient_accumulation_steps}, Effective batch: {effective_batch_size}")
    print(f"Tokens per step: {tokens_per_step:,}")
    print(f"LR: {lr}, Warmup: {warmup_iters}, Weight decay: {weight_decay}, Grad clip: {grad_clip}")
    print(f"Device: {device}, Compile: {use_compile}")
    print(f"Steps: {n_steps}, Eval interval: {eval_interval}, Save interval: {save_interval}")
    print("-" * 60)

    best_loss = float('inf')
    tokens_seen = 0

    for step in range(n_steps):
        # Set learning rate for this step (warmup schedule)
        current_lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        model.train()
        optimizer.zero_grad(set_to_none=True)

        # Gradient accumulation
        accum_loss = 0.0
        for _ in range(gradient_accumulation_steps):
            canonical_batch = data_loader.next_batch()
            permuted, compartment_ids = batch_processor.process_batch(canonical_batch)

            # Targets: next permuted token
            targets = torch.empty_like(permuted)
            targets[:, :-1] = permuted[:, 1:]
            targets[:, -1] = -1

            with autocast_ctx:
                logits, loss = model(permuted, targets, compartment_ids=compartment_ids)
                loss = loss / gradient_accumulation_steps

            loss.backward()
            accum_loss += loss.item()
            tokens_seen += permuted.numel()

        # Gradient clipping
        if grad_clip != 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        # Quick loss log
        if step % log_interval == 0:
            tokens_m = tokens_seen / 1e6
            print(f"Step {step:6d} | Train loss: {accum_loss:.4f} | LR: {current_lr:.2e} | Tok: {tokens_m:.1f}M")

        # Full eval
        if step % eval_interval == 0 or step == n_steps - 1:
            model.eval()

            eval_losses = []
            per_comp_losses = {c: [] for c in range(n_compartments)}

            with torch.no_grad(), autocast_ctx:
                # Overall eval
                for _ in range(10):
                    eval_canonical = eval_data_loader.next_batch()
                    eval_permuted, eval_comp_ids = eval_batch_processor.process_batch(eval_canonical)

                    eval_targets = torch.empty_like(eval_permuted)
                    eval_targets[:, :-1] = eval_permuted[:, 1:]
                    eval_targets[:, -1] = -1

                    _, eval_loss = model(eval_permuted, eval_targets, compartment_ids=eval_comp_ids)
                    eval_losses.append(eval_loss.item())

                # Per-compartment eval (use perms already on GPU from batch_processor)
                perms_gpu = batch_processor.perms
                for comp_id in range(n_compartments):
                    for _ in range(5):
                        eval_canonical = eval_data_loader.next_batch().to(device)
                        B, T = eval_canonical.shape

                        comp_ids = torch.full((B, T), comp_id, dtype=torch.long, device=device)
                        eval_permuted = perms_gpu[comp_ids, eval_canonical]

                        eval_targets = torch.empty_like(eval_permuted)
                        eval_targets[:, :-1] = eval_permuted[:, 1:]
                        eval_targets[:, -1] = -1

                        _, eval_loss = model(eval_permuted, eval_targets, compartment_ids=comp_ids)
                        per_comp_losses[comp_id].append(eval_loss.item())

            avg_loss = sum(eval_losses) / len(eval_losses)
            best_loss = min(best_loss, avg_loss)

            comp_str = " | ".join([f"C{c}: {sum(per_comp_losses[c])/len(per_comp_losses[c]):.4f}"
                                   for c in range(n_compartments)])

            tokens_m = tokens_seen / 1e6
            print(f"[EVAL] Step {step:6d} | Val loss: {avg_loss:.4f} | Best: {best_loss:.4f} | {comp_str}")

        # Save checkpoint
        if save_interval > 0 and (step > 0 and step % save_interval == 0):
            ckpt_path = os.path.join(save_dir, f"model_step_{step}.pt")
            # Get the underlying model if compiled
            model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model
            torch.save(model_to_save.state_dict(), ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

    # Final save
    final_path = os.path.join(save_dir, f"model_final.pt")
    model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model
    torch.save(model_to_save.state_dict(), final_path)
    print(f"Saved final model to {final_path}")

    return best_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune first/last layers on compartmentalized data")
    parser.add_argument(
        "--baseline-checkpoint",
        type=str,
        default="/mnt/pccfs2/backed_up/vin/dev/translation-compression/out/translation-compression/dw-2-1-dry-runs/2025-10-27T18-46-17Z__1-domain-baseline-fineweb-8-256-prototype__fb319b5c__s64__8000b93__a48a5009/checkpoints/step-1000000/model.pt",
    )
    parser.add_argument(
        "--data-pattern",
        type=str,
        default="data/fineweb350B-dedup-suffix-31/fineweb350b-dedup_train_*.bin",
    )
    parser.add_argument("--n-compartments", type=int, default=2)
    parser.add_argument("--n-first-layers", type=int, default=1,
                        help="Number of layers at the start to keep trainable")
    parser.add_argument("--n-last-layers", type=int, default=1,
                        help="Number of layers at the end to keep trainable")
    parser.add_argument("--n-steps", type=int, default=500000)
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Micro batch size per gradient accumulation step")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4,
                        help="Number of micro-batches to accumulate")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup-iters", type=int, default=1000)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--save-interval", type=int, default=50000,
                        help="Save checkpoint every N steps (0 to disable)")
    parser.add_argument("--save-dir", type=str, default="experiment/finetune_checkpoints")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=64)
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile()")
    args = parser.parse_args()

    train(
        baseline_checkpoint=args.baseline_checkpoint,
        data_pattern=args.data_pattern,
        n_compartments=args.n_compartments,
        n_first_layers=args.n_first_layers,
        n_last_layers=args.n_last_layers,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lr=args.lr,
        warmup_iters=args.warmup_iters,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        save_dir=args.save_dir,
        device=args.device,
        seed=args.seed,
        use_compile=not args.no_compile,
    )
