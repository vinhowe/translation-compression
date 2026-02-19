"""Quick profiling of the finetune_first_last.py training loop."""

import os
import sys
import time
import glob

import numpy as np
import torch
from contextlib import nullcontext

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import GPT
from src.config.job_config import Model as ModelConfig


# Minimal data loading
def _load_data_shard(filename: str) -> np.ndarray:
    with open(filename, "rb") as f:
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        assert header[0] == 20251013
        ntok = header[2]
        tokens = np.frombuffer(f.read(), dtype=np.uint32)
    return tokens


class SimpleDataLoader:
    def __init__(self, data_pattern, batch_size, block_size):
        self.batch_size = batch_size
        self.block_size = block_size
        self.files = sorted(glob.glob(data_pattern))
        self.tokens = _load_data_shard(self.files[0])
        self.pos = 0
        self.rng = np.random.default_rng(42)

    def next_batch(self):
        B, T = self.batch_size, self.block_size
        batch = np.empty((B, T), dtype=np.int64)
        for i in range(B):
            start = self.rng.integers(0, len(self.tokens) - T - 1)
            batch[i] = self.tokens[start:start + T]
        return torch.from_numpy(batch)


def generate_permutations(vocab_size, max_compartments, seed=64):
    ss = np.random.SeedSequence(seed)
    child_seeds = ss.spawn(max_compartments)
    perms = np.empty((max_compartments, vocab_size), dtype=np.int64)
    for c, child_ss in enumerate(child_seeds):
        gen = np.random.Generator(np.random.PCG64(child_ss))
        perms[c] = gen.permutation(vocab_size)
    return perms


def main():
    device = torch.device("cuda")

    # Config
    config = ModelConfig(
        n_layer=8, n_head=4, n_embd=256, block_size=64,
        vocab_size=151936 + 1, weight_tying=False, bias=False, dropout=0.0,
        embedding_vocab_size=151936 + 1, shared_token_embeddings=False,
        use_compartment_embeddings=True, base_vocab_size=151936,
        max_compartments=16, translation_token_id=151936,
    )

    checkpoint_path = "/mnt/pccfs2/backed_up/vin/dev/translation-compression/out/translation-compression/dw-2-1-dry-runs/2025-10-27T18-46-17Z__1-domain-baseline-fineweb-8-256-prototype__fb319b5c__s64__8000b93__a48a5009/checkpoints/step-1000000/model.pt"

    print("Loading model...")
    model = GPT(config)
    state_dict = torch.load(checkpoint_path, map_location=device)
    for k in list(state_dict.keys()):
        if k.startswith("_orig_mod."):
            state_dict[k[len("_orig_mod."):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.to(device)

    # Freeze middle layers
    for i, block in enumerate(model.transformer.h):
        if i not in [0, 7]:
            for p in block.parameters():
                p.requires_grad = False

    # Compile model
    print("Compiling model...")
    model = torch.compile(model)

    print("Setting up data...")
    data_pattern = "data/fineweb350B-dedup-suffix-31/fineweb350b-dedup_train_*.bin"
    batch_size = 512
    block_size = 64
    grad_accum = 4

    loader = SimpleDataLoader(data_pattern, batch_size, block_size)
    # Keep perms on GPU!
    perms = torch.from_numpy(generate_permutations(151936, 16, seed=64)).to(device)

    optimizer = model.configure_optimizers(0.0, 2e-5, (0.9, 0.999), "cuda")
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    n_compartments = 2

    # Warmup (more iterations for torch.compile)
    print("Warmup (includes compile)...")
    for _ in range(5):
        canonical = loader.next_batch().to(device)
        comps = torch.randint(0, n_compartments, (batch_size,), device=device)
        comp_ids = comps.unsqueeze(1).expand(-1, block_size)
        permuted = perms[comp_ids, canonical]
        targets = torch.empty_like(permuted)
        targets[:, :-1] = permuted[:, 1:]
        targets[:, -1] = -1
        with autocast_ctx:
            _, loss = model(permuted, targets, compartment_ids=comp_ids)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    torch.cuda.synchronize()

    # Profile with torch profiler
    print("\nProfiling with torch.profiler...")

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            for _ in range(grad_accum):
                canonical = loader.next_batch().to(device)
                comps = torch.randint(0, n_compartments, (batch_size,), device=device)
                comp_ids = comps.unsqueeze(1).expand(-1, block_size)
                permuted = perms[comp_ids, canonical]
                targets = torch.empty_like(permuted)
                targets[:, :-1] = permuted[:, 1:]
                targets[:, -1] = -1
                with autocast_ctx:
                    _, loss = model(permuted, targets, compartment_ids=comp_ids)
                    loss = loss / grad_accum
                loss.backward()
            optimizer.step()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))

    # Also do simple timing
    print("\n" + "="*60)
    print("Simple timing breakdown:")
    print("="*60)

    torch.cuda.synchronize()

    # Time individual components
    n_iters = 5

    # Data loading + transfer
    t0 = time.perf_counter()
    for _ in range(n_iters * grad_accum):
        canonical = loader.next_batch().to(device)
    torch.cuda.synchronize()
    data_time = (time.perf_counter() - t0) / n_iters
    print(f"Data loading + transfer (per step, {grad_accum} batches): {data_time*1000:.1f}ms")

    # Permutation (on GPU now)
    t0 = time.perf_counter()
    for _ in range(n_iters * grad_accum):
        canonical = loader.next_batch().to(device)
        comps = torch.randint(0, n_compartments, (batch_size,), device=device)
        comp_ids = comps.unsqueeze(1).expand(-1, block_size)
        permuted = perms[comp_ids, canonical]
    torch.cuda.synchronize()
    perm_time = (time.perf_counter() - t0) / n_iters - data_time
    print(f"Permutation on GPU (per step): {perm_time*1000:.1f}ms")

    # Forward pass only
    canonical = loader.next_batch().to(device)
    comps = torch.randint(0, n_compartments, (batch_size,), device=device)
    comp_ids = comps.unsqueeze(1).expand(-1, block_size)
    permuted = perms[comp_ids, canonical]
    targets = torch.empty_like(permuted)
    targets[:, :-1] = permuted[:, 1:]
    targets[:, -1] = -1

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters * grad_accum):
        with autocast_ctx:
            _, loss = model(permuted, targets, compartment_ids=comp_ids)
    torch.cuda.synchronize()
    fwd_time = (time.perf_counter() - t0) / n_iters
    print(f"Forward pass (per step, {grad_accum} batches): {fwd_time*1000:.1f}ms")

    # Backward pass
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        optimizer.zero_grad(set_to_none=True)
        for _ in range(grad_accum):
            with autocast_ctx:
                _, loss = model(permuted, targets, compartment_ids=comp_ids)
                loss = loss / grad_accum
            loss.backward()
    torch.cuda.synchronize()
    bwd_time = (time.perf_counter() - t0) / n_iters - fwd_time
    print(f"Backward pass (per step): {bwd_time*1000:.1f}ms")

    # Optimizer step
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        optimizer.step()
    torch.cuda.synchronize()
    opt_time = (time.perf_counter() - t0) / n_iters
    print(f"Optimizer step (per step): {opt_time*1000:.1f}ms")

    # Full step
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        optimizer.zero_grad(set_to_none=True)
        for _ in range(grad_accum):
            canonical = loader.next_batch().to(device)
            comps = torch.randint(0, n_compartments, (batch_size,), device=device)
            comp_ids = comps.unsqueeze(1).expand(-1, block_size)
            permuted = perms[comp_ids, canonical]
            targets = torch.empty_like(permuted)
            targets[:, :-1] = permuted[:, 1:]
            targets[:, -1] = -1
            with autocast_ctx:
                _, loss = model(permuted, targets, compartment_ids=comp_ids)
                loss = loss / grad_accum
            loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    full_time = (time.perf_counter() - t0) / n_iters
    print(f"\nFull step time: {full_time*1000:.1f}ms")
    print(f"Steps per second: {1/full_time:.2f}")
    print(f"Tokens per second: {batch_size * block_size * grad_accum / full_time:,.0f}")


if __name__ == "__main__":
    main()
