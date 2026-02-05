"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

Configuration is read from `config/job_config.py` dataclasses via `ConfigManager`.

To run on a single GPU, example:
$ python train.py --training.batch_size=32 --system.compile=False

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import json
import os
import shutil
import time
import uuid
import glob
from contextlib import nullcontext
from dataclasses import asdict, replace
from typing import Any, Optional, cast

import numpy as np
import torch
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from src.config.job_config import JobConfig, Model
from src.config.manager import ConfigManager
from src.experiment import append_to_experiment_log, cfg_hash, run_dirs, slug, write_meta
from src.model import GPT
from src.data import UniformBatchDataLoader, UniformCompartmentDataLoader
from src.assignments import write_assignments
from src.config.presets import apply_size_tier, apply_bpe16384_batch_config
from src.weights import compute_weights_map
import struct
import threading
from queue import Queue


def check_duplicate_run(
    config: JobConfig, wandb_project: str, wandb_group: Optional[str] = None
) -> Optional[str]:
    """Check if a completed run with the same config already exists in wandb.

    Returns the run ID if a duplicate is found, None otherwise.
    Only checks on master process (rank 0).
    """
    if int(os.environ.get("RANK", 0)) != 0:
        return None

    try:
        import wandb

        api = wandb.Api()

        # Extract meaningful config fields for comparison (exclude logging/system settings)
        config_dict = asdict(config)

        # For data section, only compare fields that affect training
        # uniform_seed only matters for uniform data source, not pretokenized
        data_compare = config_dict["data"].copy()
        if config_dict["data"].get("source") == "pretokenized":
            data_compare.pop("uniform_seed", None)

        compare_fields = {
            "model": config_dict["model"],
            "training": config_dict["training"],
            "experiment": config_dict["experiment"],
            "data": data_compare,
            "optimizer": config_dict["optimizer"],
        }

        # Build filters - only query finished runs, optionally within a group
        filters = {"state": "finished"}
        if wandb_group:
            filters["group"] = wandb_group

        # Query completed runs in the project
        runs = api.runs(
            wandb_project,
            filters=filters,
            per_page=1000,
        )

        for run in runs:
            run_config = run.config
            # Compare the meaningful fields
            match = True
            for section, expected in compare_fields.items():
                run_section = run_config.get(section, {})

                # For data section with pretokenized source, ignore uniform_seed in existing run too
                if section == "data" and expected.get("source") == "pretokenized":
                    run_section = run_section.copy() if run_section else {}
                    run_section.pop("uniform_seed", None)

                if run_section != expected:
                    match = False
                    break

            if match:
                return run.id

    except Exception as e:
        # Don't fail training if wandb API is unavailable
        print(f"[dedupe] Warning: Could not check for duplicates: {e}")

    return None


# Limit CPU threads to avoid oversubscription when running multiple processes
# Default to 4 threads; override with OMP_NUM_THREADS env var
_num_threads = int(os.environ.get("OMP_NUM_THREADS", "4"))
torch.set_num_threads(_num_threads)


def print0(*args, **kwargs):
    # modified print that only prints from the master process
    # if this is not a distributed run, it's just a print
    if int(os.environ.get("RANK", 0)) == 0:
        print(*args, **kwargs)


def _peek_data_shard(filename):
    # only reads the header, returns header data
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
    if header[0] != 20251013:
        print("ERROR: magic number mismatch in the data .bin file!")
        print("---> HINT: Are you passing in a correct file with --input_bin?")
        print(
            "---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README"
        )
        exit(1)
    assert header[1] == 1, "unsupported version"
    ntok = header[2]  # number of tokens (claimed)
    return ntok  # for now just return the number of tokens


def _load_data_shard(filename):
    with open(filename, "rb") as f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        assert header[0] == 20251013, "magic number mismatch in the data .bin file"
        assert header[1] == 1, "unsupported version"
        ntok = header[2]  # number of tokens (claimed)
        # the rest of it are tokens, stored as uint32
        tokens = np.frombuffer(f.read(), dtype=np.uint32)
    print(len(tokens), ntok)
    assert len(tokens) == ntok, "number of tokens read does not match header?"
    return tokens


class DistributedDataLoader:
    def __init__(self, filename_pattern, B, T, process_rank, num_processes):
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, (
            f"did not find any files that match the pattern {filename_pattern}"
        )

        # load and validate all data shards, count number of tokens in total
        ntok_total = 0
        for fname in self.files:
            shard_ntok = _peek_data_shard(fname)
            assert shard_ntok >= num_processes * B * T + 1
            ntok_total += shard_ntok
        self.ntok_total = ntok_total
        print0(
            f"DataLoader: total number of tokens: {ntok_total:,} across {len(self.files)} files"
        )

        # kick things off
        self.current_shard = None
        self.reset()

    def reset(self):
        # we're being a bit clever here: if we already had shard 0 loaded,
        # then don't do the work to reload it, just reset the pointer
        if self.current_shard != 0:
            self.current_shard = 0
            self.tokens = _load_data_shard(self.files[self.current_shard])
        self.current_position = self.process_rank * self.B * self.T

    def advance(self):  # advance to next data shard
        self.current_shard = (self.current_shard + 1) % len(self.files)  # pyright: ignore[reportOptionalOperand]
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def state_dict(self) -> dict:
        return {
            "current_shard": self.current_shard,
            "current_position": self.current_position,
        }

    def load_state_dict(self, state: dict) -> None:
        shard = state["current_shard"]
        if shard != self.current_shard:
            self.current_shard = shard
            self.tokens = _load_data_shard(self.files[self.current_shard])
        self.current_position = state["current_position"]

    def next_batch(self):
        B = self.B
        T = self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)  # targets
        # advance the start pointer in current shard
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds advance the shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()
        return x, y


class AssignmentsDataLoader:
    def __init__(
        self,
        assignments_file: str,
        filename_pattern: str,
        B: int,
        T: int,
        process_rank: int,
        num_processes: int,
        base_vocab_size: int,
        max_compartments: int,
        n_compartments: int,
        permute_tokens: bool = False,
        permutations_path: Optional[str] = None,
        permute_inputs: bool = True,
        pin_memory: bool = True,
    ) -> None:
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.B = B
        self.T = T
        self.base_vocab_size = base_vocab_size
        self.max_compartments = max_compartments
        self.n_compartments = n_compartments
        self.permute_tokens = permute_tokens
        self.permute_inputs = permute_inputs
        # translation token id differs by mode; uses n_compartments to match model vocab size
        self.translation_token_id = (
            base_vocab_size if permute_tokens else base_vocab_size * n_compartments
        )
        # Optionally load permutations array of shape [max_compartments, base_vocab]
        self._permutations: Optional[np.ndarray]
        if self.permute_tokens:
            if permutations_path is None or not os.path.exists(permutations_path):
                raise FileNotFoundError(
                    f"Permutations file not found: {permutations_path}"
                )
            perms = np.load(permutations_path)
            if perms.dtype != np.int64 and perms.dtype != np.int32:
                perms = perms.astype(np.int64)
            rows, cols = perms.shape
            if cols != base_vocab_size:
                raise ValueError(
                    f"permutations.npy base vocab mismatch: {cols} != {base_vocab_size}"
                )
            if rows < max_compartments:
                raise ValueError(
                    f"permutations.npy compartments {rows} < required {max_compartments}"
                )
            if rows > max_compartments:
                perms = perms[:max_compartments]
            self._permutations = perms
        else:
            self._permutations = None

        # Whether to return pinned-memory CPU tensors for faster H2D copies
        self._pin_memory = bool(pin_memory and torch.cuda.is_available())

        # Load assignments header and records
        with open(assignments_file, "rb") as f:
            header = f.read(32)
            magic, version, rec_size, flags, num_compartments, num_records, seed = (
                struct.unpack("<8sBBHIQQ", header)
            )
            assert magic == b"TCASSIGN", "assignments magic mismatch"
            assert version == 1 and rec_size == 8, (
                "unsupported assignments version/record size"
            )
            assert num_compartments == max_compartments, (
                f"assignments num_compartments {num_compartments} != expected {max_compartments}"
            )
            self._records = np.frombuffer(f.read(), dtype=np.uint64)
        self.num_records = int(self._records.shape[0])

        # Prepare token shards
        self.files = sorted(glob.glob(filename_pattern))
        assert len(self.files) > 0, (
            f"did not find any files that match the pattern {filename_pattern}"
        )
        self.current_shard = 0
        self.tokens = _load_data_shard(self.files[self.current_shard])
        self.token_pos = 0
        # offset per rank
        if len(self.tokens) > (self.T + 2):
            self.token_pos = (self.process_rank * self.T) % (
                len(self.tokens) - (self.T + 2)
            )
        else:
            self.token_pos = 0

        # assignment index start (strided by world size)
        self.assignment_idx = self.process_rank % max(1, self.num_records)

        # Pre-allocate pinned memory tensors to avoid expensive allocation per batch
        # Pinned memory allocation is ~200ms, so reusing saves significant time
        if self._pin_memory:
            self._x_buf = torch.empty((B, T), dtype=torch.long, pin_memory=True)
            self._y_buf = torch.empty((B, T), dtype=torch.long, pin_memory=True)
            self._cids_buf = torch.empty((B, T), dtype=torch.long, pin_memory=True)
        else:
            self._x_buf = None
            self._y_buf = None
            self._cids_buf = None

    def _load_shard(self, idx: int) -> None:
        new_idx = idx % len(self.files)
        if new_idx != self.current_shard:
            self.current_shard = new_idx
            self.tokens = _load_data_shard(self.files[self.current_shard])
        # Always reset token position when (re)loading or cycling shards,
        # including the single-shard case where new_idx == current_shard.
        self.token_pos = 0

    def _advance_shard(self) -> None:
        self._load_shard((self.current_shard + 1) % len(self.files))

    def _read_tokens(self, n: int) -> np.ndarray:
        out = np.empty(n, dtype=np.int32)
        filled = 0
        while filled < n:
            remaining = len(self.tokens) - self.token_pos
            if remaining <= 1:  # keep one token gap safety
                self._advance_shard()
                continue
            can_take = min(n - filled, remaining)
            out[filled : filled + can_take] = self.tokens[
                self.token_pos : self.token_pos + can_take
            ].astype(np.int32)
            self.token_pos += can_take
            filled += can_take
            if self.token_pos >= len(self.tokens):
                self._advance_shard()
        return out

    @staticmethod
    def _decode_record(word: np.uint64) -> tuple[int, int, int]:
        w = int(word)
        kind = w & 0xFF
        src = (w >> 16) & 0xFFFF
        dst = (w >> 32) & 0xFFFF
        return kind, src, dst

    def state_dict(self) -> dict:
        return {
            "current_shard": self.current_shard,
            "token_pos": self.token_pos,
            "assignment_idx": self.assignment_idx,
        }

    def load_state_dict(self, state: dict) -> None:
        shard = state["current_shard"]
        if shard != self.current_shard:
            self._load_shard(shard)
        self.token_pos = state["token_pos"]
        self.assignment_idx = state["assignment_idx"]

    def reset(self) -> None:
        self._load_shard(0)
        self.assignment_idx = self.process_rank % max(1, self.num_records)

    def next_batch(self):
        B = self.B
        T = self.T
        half = T // 2
        assert T % 2 == 0 and half >= 2, "block_size must be even and >= 4"

        # Reuse pre-allocated pinned-memory tensors if available (avoids ~200ms alloc overhead)
        if self._x_buf is not None:
            x_t = self._x_buf
            y_t = self._y_buf
            cids_t = self._cids_buf
        else:
            x_t = torch.empty((B, T), dtype=torch.long)
            y_t = torch.empty((B, T), dtype=torch.long)
            cids_t = torch.empty((B, T), dtype=torch.long)

        x_np = x_t.numpy()
        y_np = y_t.numpy()
        cid_np = cids_t.numpy()

        # Vectorized decode of B assignment records (strided by num_processes)
        rec_indices = (
            self.assignment_idx + self.num_processes * np.arange(B, dtype=np.int64)
        ) % max(1, self.num_records)
        words = self._records[rec_indices]
        w = words.astype(np.uint64, copy=False)
        kinds = (w & np.uint64(0xFF)).astype(np.int64, copy=False)
        srcs = ((w >> np.uint64(16)) & np.uint64(0xFFFF)).astype(np.int64, copy=False)
        dsts = ((w >> np.uint64(32)) & np.uint64(0xFFFF)).astype(np.int64, copy=False)

        is_trans = kinds == 1
        idx_trans = np.nonzero(is_trans)[0]
        idx_comp = np.nonzero(~is_trans)[0]

        # Preserve per-example token consumption order: compute sizes and a single contiguous slice
        size_per_b = np.where(is_trans, half, T).astype(np.int64, copy=False)
        if B == 0:
            return x_t, y_t, cids_t
        starts = np.zeros(B, dtype=np.int64)
        if B > 1:
            starts[1:] = np.cumsum(size_per_b[:-1], dtype=np.int64)
        total_needed = int(starts[-1] + size_per_b[-1])
        tokens_batch = self._read_tokens(total_needed).astype(np.int64, copy=False)

        # Translation group: gather half-length segments in b-order
        if idx_trans.size > 0:
            m = int(idx_trans.size)
            base_idx_half = np.arange(half, dtype=np.int64)[None, :]
            trans_starts = starts[idx_trans][:, None]
            samples = tokens_batch[trans_starts + base_idx_half]

            # Prepare input/output seq and cids for translation rows
            seq_in = np.empty((m, T), dtype=np.int64)
            seq_out = np.empty((m, T), dtype=np.int64)
            cid_tr = np.empty((m, T), dtype=np.int64)

            # Set translation token positions and cids (same for input/output)
            seq_in[:, 0] = self.translation_token_id
            seq_in[:, half] = self.translation_token_id
            seq_out[:, 0] = self.translation_token_id
            seq_out[:, half] = self.translation_token_id
            cid_tr[:, 0] = srcs[idx_trans]
            cid_tr[:, half] = dsts[idx_trans]

            if self.permute_tokens:
                # Direct fancy indexing - avoids loading full rows (2000x faster)
                # Old approach loaded entire rows (~311MB), new approach loads only needed elements (~64KB)
                perms = cast(np.ndarray, self._permutations)
                src_comp = srcs[idx_trans][:, None]  # [m, 1]
                dst_comp = dsts[idx_trans][:, None]  # [m, 1]
                token_slice = samples[:, : half - 1]  # [m, half-1]
                perm_src = perms[
                    np.broadcast_to(src_comp, token_slice.shape), token_slice
                ]
                perm_dst = perms[
                    np.broadcast_to(dst_comp, token_slice.shape), token_slice
                ]
                # Inputs: optionally permuted
                if self.permute_inputs:
                    seq_in[:, 1:half] = perm_src
                    seq_in[:, half + 1 :] = perm_dst
                else:
                    seq_in[:, 1:half] = samples[:, : half - 1]
                    seq_in[:, half + 1 :] = samples[:, : half - 1]
                # Targets: always in permuted space
                seq_out[:, 1:half] = perm_src
                seq_out[:, half + 1 :] = perm_dst
            else:
                base = self.base_vocab_size
                src_offsets = srcs[idx_trans] * base
                dst_offsets = dsts[idx_trans] * base
                seq_in[:, 1:half] = samples[:, : half - 1] + src_offsets[:, None]
                seq_in[:, half + 1 :] = samples[:, : half - 1] + dst_offsets[:, None]
                # Without permutation, inputs and outputs are identical
                seq_out[:, 1:half] = seq_in[:, 1:half]
                seq_out[:, half + 1 :] = seq_in[:, half + 1 :]

            # Fill cids across spans
            cid_tr[:, 1:half] = srcs[idx_trans][:, None]
            cid_tr[:, half + 1 :] = dsts[idx_trans][:, None]

            # Targets: next-token; last is ignored
            y_tr = np.empty_like(seq_out)
            y_tr[:, :-1] = seq_out[:, 1:]
            y_tr[:, -1] = -1

            # Scatter into batch slots
            x_np[idx_trans] = seq_in
            y_np[idx_trans] = y_tr
            cid_np[idx_trans] = cid_tr

        # Compartment group: gather T-length segments in b-order
        if idx_comp.size > 0:
            base_idx_T = np.arange(T, dtype=np.int64)[None, :]
            comp_starts = starts[idx_comp][:, None]
            samples = tokens_batch[comp_starts + base_idx_T]

            if self.permute_tokens:
                # Direct fancy indexing - avoids loading full rows (2000x faster)
                perms = cast(np.ndarray, self._permutations)
                src_comp = srcs[idx_comp][:, None]  # [n, 1]
                perm_samples = perms[np.broadcast_to(src_comp, samples.shape), samples]
                # Inputs: optionally permuted
                if self.permute_inputs:
                    x_comp = perm_samples
                else:
                    x_comp = samples
                # Targets: always in permuted space
                y_comp = np.empty_like(perm_samples)
                y_comp[:, :-1] = perm_samples[:, 1:]
                y_comp[:, -1] = -1
            else:
                base = self.base_vocab_size
                src_offsets = srcs[idx_comp] * base
                x_comp = samples + src_offsets[:, None]
                y_comp = np.empty_like(x_comp)
                y_comp[:, :-1] = x_comp[:, 1:]
                y_comp[:, -1] = -1

            x_np[idx_comp] = x_comp
            y_np[idx_comp] = y_comp
            cid_np[idx_comp, :] = srcs[idx_comp][:, None]

        # advance assignment pointer for next batch
        self.assignment_idx = (
            self.assignment_idx + self.num_processes * B
        ) % self.num_records

        # Optional runtime invariants for debugging. Enable with TC_DEBUG_LOADER=1
        if os.environ.get("TC_DEBUG_LOADER", "") == "1":
            assert (
                x_np.shape == (B, T) and y_np.shape == (B, T) and cid_np.shape == (B, T)
            )
            assert (y_np[:, -1] == -1).all()
            if (
                B > 0
                and T > 1
                and not (self.permute_tokens and not self.permute_inputs)
            ):
                assert (y_np[:, :-1] == x_np[:, 1:]).all()
            if idx_trans.size > 0:
                assert (x_np[idx_trans, 0] == self.translation_token_id).all()
                assert (x_np[idx_trans, half] == self.translation_token_id).all()
            # Basic bounds checking of vocab id space
            if self.permute_tokens:
                # Permuted mode uses base vocab ids and a single translation token at base_vocab_size
                assert (x_np >= 0).all()
                assert (x_np < (self.base_vocab_size + 1)).all()
            else:
                max_vocab = self.base_vocab_size * self.max_compartments + 1
                assert (x_np >= 0).all() and (x_np < max_vocab).all()

        return x_t, y_t, cids_t


STORAGE_ROOT = os.environ.get(
    "TC_STORAGE_ROOT", "/mnt/pccfs2/backed_up/vin/dev/translation-compression"
)


def main(config: JobConfig) -> None:
    # Apply size tier overrides (if provided) and mirror assignment_seed to training.seed
    config = apply_size_tier(config)
    # Auto-configure batch/grad_accum for bpe16384 vocab if not explicitly set
    config = apply_bpe16384_batch_config(config)
    config = replace(
        config,
        experiment=replace(config.experiment, assignment_seed=config.training.seed),
    )
    # If uniform_seed is 0 (default), inherit from training.seed for simpler sweep configs
    if config.data.uniform_seed == 0:
        config = replace(
            config,
            data=replace(config.data, uniform_seed=config.training.seed),
        )
    # -----------------------------------------------------------------------------
    # Unpack config into local variables (matches original script expectations)
    # wandb logging
    wandb_log = config.logging.wandb_log
    wandb_project = config.logging.wandb_project
    wandb_run_name = config.logging.wandb_run_name
    wandb_group = config.logging.wandb_group
    wandb_notes = config.logging.wandb_notes

    # Check for duplicate completed runs before expensive setup
    if wandb_log:
        dup_run_id = check_duplicate_run(config, wandb_project, wandb_group)
        if dup_run_id:
            print0(
                f"[dedupe] Skipping: found completed run with same config: {dup_run_id}"
            )
            print0(
                f"[dedupe] View at: https://wandb.ai/pccl/{wandb_project}/runs/{dup_run_id}"
            )
            return

    # I/O: structured run directory under hardcoded storage root
    run_id = os.environ.get("RUN_ID", uuid.uuid4().hex[:8])
    out_root = os.path.join(STORAGE_ROOT, "out")
    if os.environ.get("OUT_DIR"):
        out_dir = os.environ["OUT_DIR"]
        project_dir = os.path.join(out_root, slug(wandb_project))
        group_dir = os.path.join(project_dir, slug(wandb_group or "default"))
    else:
        project_dir, group_dir, out_dir = run_dirs(
            out_root,
            wandb_project,
            wandb_group,
            wandb_run_name,
            config,
            run_id,
        )
    eval_interval = config.training.eval_interval
    log_interval = config.training.log_interval
    eval_iters = config.training.eval_iters
    eval_only = config.training.eval_only
    always_save_checkpoint = config.training.always_save_checkpoint
    init_from = config.init.init_from
    # data
    train_bin = config.data.train_bin
    val_bin = config.data.val_bin
    gradient_accumulation_steps = config.training.gradient_accumulation_steps
    batch_size = config.training.batch_size
    block_size = config.model.block_size
    # model
    dropout = config.model.dropout
    # adamw optimizer
    learning_rate = config.optimizer.learning_rate
    max_iters = config.training.max_iters
    weight_decay = config.optimizer.weight_decay
    beta1 = config.optimizer.beta1
    beta2 = config.optimizer.beta2
    grad_clip = config.optimizer.grad_clip
    # learning rate decay settings
    # decay_lr = config.lr.decay_lr
    warmup_iters = config.lr.warmup_iters
    # lr_decay_iters = config.lr.lr_decay_iters
    # min_lr = config.lr.min_lr
    # DDP settings
    backend = config.distributed.backend
    # system
    device = config.system.device
    if config.system.dtype == "auto":
        dtype = (
            "bfloat16"
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else "float16"
        )
    else:
        dtype = config.system.dtype
    compile = config.system.compile
    # -----------------------------------------------------------------------------

    # various inits, derived attributes, I/O setup
    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    ddp_rank = None
    if ddp:
        init_process_group(backend=backend)
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        ddp_world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = (
            ddp_rank == 0
        )  # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank  # each process gets a different seed
        # world_size number of processes will be training simultaneously, so we can scale
        # down the desired gradient accumulation iterations per process proportionally
        assert gradient_accumulation_steps % ddp_world_size == 0
        gradient_accumulation_steps //= ddp_world_size
    else:
        # if not ddp, we are running on a single gpu, and one process
        master_process = True
        seed_offset = 0
        ddp_world_size = 1
    tokens_per_iter = (
        gradient_accumulation_steps * ddp_world_size * batch_size * block_size
    )
    print(
        f"tokens per iteration will be: gradient_accumulation_steps={gradient_accumulation_steps} * ddp_world_size={ddp_world_size} * batch_size={batch_size} * block_size={block_size} = {tokens_per_iter:,}"
    )
    # Effective global batch size across all processes and gradient accumulation
    effective_batch_size = gradient_accumulation_steps * ddp_world_size * batch_size

    # Restrict checkpointing to these specific global steps only (reverse engineering
    # Morris figure)
    checkpoint_steps = {
        100,
        850,
        3500,
        7000,
        14000,
        29000,
        60000,
        120000,
        240000,
        500000,
        1000000,
    }

    assignments_path = os.path.join(out_dir, "assignments.bin")
    permutations_path = os.path.join(out_dir, "permutations.npy")
    training_seed = config.training.seed + seed_offset
    # Determine data source
    use_pretokenized = config.data.source == "pretokenized"

    # If using pretokenized data, compute deterministic cache paths for assignments/permutations
    if use_pretokenized:
        # Cache directory under the same hardcoded storage root
        cache_root = os.path.join(STORAGE_ROOT, "cache")
        exp = config.experiment
        if exp.max_compartments is None:
            raise ValueError("experiment.max_compartments is required")
        max_compartments_int = cast(int, exp.max_compartments)
        total_examples = config.training.max_iters * effective_batch_size

        # Format float safely for filenames
        def _fmt_float(x: float) -> str:
            s = f"{x:.6g}".rstrip("0").rstrip(".")
            return s.replace(".", "p") if "." in s else s

        # Build description from inputs to assignments creation
        assignments_desc = (
            f"n{exp.n_compartments}_t{_fmt_float(max(0.0, float(exp.translation_ratio)))}_"
            f"m{exp.translation_ratio_mode}_"
            f"sc{exp.compartment_scaling}_total{int(total_examples)}_"
            f"maxc{max_compartments_int}_seed{int(training_seed)}"
        )
        # Point assignments_path to cached file
        assignments_path = os.path.join(
            cache_root, f"assignments_{assignments_desc}.bin"
        )

        # If permuting tokens per compartment, also compute cached permutations path
        if exp.permute_tokens_per_compartment:
            if config.model.vocab_size is not None:
                base_vocab_int = cast(int, config.model.vocab_size)
                perms_desc = f"basev{base_vocab_int}_maxc{max_compartments_int}_seed{int(training_seed)}"
                permutations_path = os.path.join(
                    cache_root, f"permutations_{perms_desc}.npy"
                )

    if master_process:
        os.makedirs(os.path.join(out_dir, "checkpoints"), exist_ok=True)
        write_meta(out_dir, config)
        append_to_experiment_log(project_dir, group_dir, out_dir, config)
        # Generate assignments/permutations only for pretokenized source
        if use_pretokenized:
            exp = config.experiment
            try:
                if exp.max_compartments is None:
                    raise ValueError("experiment.max_compartments is required")
                max_compartments_int = cast(int, exp.max_compartments)
                total_examples = config.training.max_iters * effective_batch_size
                # Ensure cache directory exists
                cache_root = os.path.dirname(assignments_path)
                os.makedirs(cache_root, exist_ok=True)
                # Assignments: write only if not already cached, with simple cross-process locking
                if not os.path.exists(assignments_path):
                    lock_path = assignments_path + ".lock"
                    tmp_path = assignments_path + f".tmp.{os.getpid()}"
                    acquired = False
                    while True:
                        try:
                            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
                            os.close(fd)
                            acquired = True
                            break
                        except FileExistsError:
                            # Another process is writing; wait until file appears
                            if os.path.exists(assignments_path):
                                break
                            time.sleep(0.1)
                    if acquired:
                        try:
                            write_assignments(
                                tmp_path,
                                weights_map=compute_weights_map(
                                    n=exp.n_compartments,
                                    t=max(0.0, float(exp.translation_ratio)),
                                    scaling=exp.compartment_scaling,
                                    mode=exp.translation_ratio_mode,
                                ),
                                total_examples=total_examples,
                                max_compartments=max_compartments_int,
                                seed=training_seed,
                                no_shuffle=False,
                            )
                            os.replace(tmp_path, assignments_path)
                            print0(
                                f"Wrote assignments to {assignments_path} with total_examples={total_examples:,}"
                            )
                        finally:
                            try:
                                if os.path.exists(tmp_path):
                                    os.remove(tmp_path)
                            except Exception:
                                pass
                            try:
                                os.remove(lock_path)
                            except Exception:
                                pass
                    else:
                        print0(f"Using cached assignments at {assignments_path}")
                else:
                    print0(f"Using cached assignments at {assignments_path}")
                # If enabled, also create deterministic per-compartment permutations of base tokens
                if exp.permute_tokens_per_compartment:
                    if config.model.vocab_size is None:
                        raise ValueError(
                            "model.vocab_size (base) must be set to create permutations"
                        )
                    base_vocab_int = cast(int, config.model.vocab_size)
                    # Permutations: write only if not already cached, with simple cross-process locking
                    if not os.path.exists(permutations_path):
                        lock_path = permutations_path + ".lock"
                        tmp_path = permutations_path + f".tmp.{os.getpid()}"
                        acquired = False
                        while True:
                            try:
                                fd = os.open(
                                    lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR
                                )
                                os.close(fd)
                                acquired = True
                                break
                            except FileExistsError:
                                # Another process is writing; wait until file appears
                                if os.path.exists(permutations_path):
                                    break
                                time.sleep(0.1)
                        if acquired:
                            try:
                                # Use SeedSequence to spawn deterministic child RNGs per compartment
                                ss = np.random.SeedSequence(
                                    int(training_seed) & 0xFFFFFFFFFFFFFFFF
                                )
                                child_seeds = ss.spawn(max_compartments_int)
                                # Allocate permutations for all compartments up to max_compartments
                                perms = np.empty(
                                    (int(max_compartments_int), base_vocab_int),
                                    dtype=np.int64,
                                )
                                for c, child_ss in enumerate(child_seeds):
                                    gen = np.random.Generator(np.random.PCG64(child_ss))
                                    perms[c] = gen.permutation(base_vocab_int).astype(
                                        np.int64
                                    )
                                # Write using a file handle to avoid numpy appending an extra .npy
                                with open(tmp_path, "wb") as f:
                                    np.save(f, perms)
                                    f.flush()
                                    os.fsync(f.fileno())
                                os.replace(tmp_path, permutations_path)
                                print0(
                                    f"Wrote per-compartment permutations to {permutations_path} with shape {perms.shape}"
                                )
                            finally:
                                try:
                                    if os.path.exists(tmp_path):
                                        os.remove(tmp_path)
                                except Exception:
                                    pass
                                try:
                                    os.remove(lock_path)
                                except Exception:
                                    pass
                        else:
                            print0(f"Using cached permutations at {permutations_path}")
                    else:
                        print0(f"Using cached permutations at {permutations_path}")
            except Exception as e:
                print0(f"Failed generating assignments: {e}")
                raise
    # Ensure all processes wait for assignments/permutations if using pretokenized data
    if ddp and use_pretokenized:
        torch.distributed.barrier()
    torch.manual_seed(training_seed)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    device_type = (
        "cuda" if "cuda" in device else "cpu"
    )  # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]
    ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.autocast(device_type=device_type, dtype=ptdtype)
    )

    # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
    iter_num = 0
    best_val_loss = 1e9

    # model init
    model: torch.nn.Module
    checkpoint: Optional[dict[str, Any]] = None
    # Compute composite vocabulary
    exp_cfg = config.experiment
    if exp_cfg.max_compartments is None:
        raise ValueError("experiment.max_compartments is required")
    base_vocab = config.model.vocab_size
    if base_vocab is None:
        raise ValueError(
            "model.vocab_size (base) must be set for composite vocab computation"
        )
    # If permuting per compartment, vocabulary is base_vocab + 1 (only translation token)
    # Otherwise, it's base_vocab * n_compartments + 1 (offset scheme)
    # Note: we use n_compartments (not max_compartments) to size the model efficiently
    composite_vocab = (
        base_vocab + 1
        if exp_cfg.permute_tokens_per_compartment
        else base_vocab * exp_cfg.n_compartments + 1
    )
    # Translation token id differs by mode
    translation_token_id_cfg = (
        base_vocab
        if exp_cfg.permute_tokens_per_compartment
        else base_vocab * exp_cfg.n_compartments
    )
    # Auto-resume from latest checkpoint if present; else init from scratch
    ckpt_dir = os.path.join(out_dir, "checkpoints", "latest")
    dataloader_state: Optional[dict] = None
    if os.path.islink(ckpt_dir) or os.path.isdir(ckpt_dir):
        model_ckpt = os.path.join(ckpt_dir, "model.pt")
        trainer_state_path = os.path.join(ckpt_dir, "trainer_state.json")
        if os.path.exists(model_ckpt) and os.path.exists(trainer_state_path):
            print(f"Resuming training from checkpoint: {ckpt_dir}")
            with open(trainer_state_path, "r") as f:
                trainer_state = json.load(f)
            iter_num = trainer_state["iter_num"]
            best_val_loss = trainer_state.get("best_val_loss", 1e9)
            # Load model state
            model_state_dict = torch.load(model_ckpt, map_location=device)
            # convert torch.compile state dict back to regular state dict
            unwanted_prefix = "_orig_mod."
            for k, v in list(model_state_dict.items()):
                if k.startswith(unwanted_prefix):
                    model_state_dict[k[len(unwanted_prefix) :]] = model_state_dict.pop(k)
            checkpoint = {"model_state_dict": model_state_dict}
            # Load optimizer state if present
            opt_ckpt = os.path.join(ckpt_dir, "optimizer.pt")
            if os.path.exists(opt_ckpt):
                checkpoint["optimizer"] = torch.load(opt_ckpt, map_location=device)
            # Load dataloader state if present
            dl_ckpt = os.path.join(ckpt_dir, "dataloader.pt")
            if os.path.exists(dl_ckpt):
                dataloader_state = torch.load(dl_ckpt, map_location="cpu")
    if checkpoint is not None:
        # Build model config matching the current config (not from checkpoint)
        vocab = composite_vocab
        gptconf = Model(
            **{
                **asdict(config.model),
                "vocab_size": vocab,
                "embedding_vocab_size": (
                    (base_vocab + 1)
                    if config.experiment.shared_token_embeddings
                    else vocab
                ),
                "shared_token_embeddings": bool(
                    config.experiment.shared_token_embeddings
                ),
                "use_compartment_embeddings": bool(
                    config.experiment.use_compartment_embeddings
                ),
                "copy_compartment_embeddings": (
                    False
                    if exp_cfg.permute_tokens_per_compartment
                    else bool(config.experiment.copy_compartment_embeddings)
                ),
                "copy_compartment_lm_head": (
                    False
                    if exp_cfg.permute_tokens_per_compartment
                    else bool(config.experiment.copy_compartment_lm_head)
                ),
                "base_vocab_size": base_vocab,
                "max_compartments": exp_cfg.n_compartments,
                "translation_token_id": translation_token_id_cfg,
                "weight_tying": (
                    False
                    if config.experiment.shared_token_embeddings
                    else config.model.weight_tying
                ),
            }
        )
        model = GPT(gptconf)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        # init a new model from scratch
        print("Initializing a new model from scratch")
        # determine the vocab size we'll use for from-scratch training
        vocab = composite_vocab
        gptconf = Model(
            **{
                **asdict(config.model),
                "vocab_size": vocab,
                # pass advanced options to model
                "embedding_vocab_size": (
                    (base_vocab + 1)
                    if config.experiment.shared_token_embeddings
                    else vocab
                ),
                "shared_token_embeddings": bool(
                    config.experiment.shared_token_embeddings
                ),
                "use_compartment_embeddings": bool(
                    config.experiment.use_compartment_embeddings
                ),
                # When permuting per-compartment, copying compartment weights is a no-op and shape-incompatible
                "copy_compartment_embeddings": (
                    False
                    if exp_cfg.permute_tokens_per_compartment
                    else bool(config.experiment.copy_compartment_embeddings)
                ),
                "copy_compartment_lm_head": (
                    False
                    if exp_cfg.permute_tokens_per_compartment
                    else bool(config.experiment.copy_compartment_lm_head)
                ),
                "base_vocab_size": base_vocab,
                "max_compartments": exp_cfg.n_compartments,  # Use n_compartments for model sizing
                "translation_token_id": translation_token_id_cfg,
                # disable weight tying if shared embeddings are used (validated elsewhere)
                "weight_tying": (
                    False
                    if config.experiment.shared_token_embeddings
                    else config.model.weight_tying
                ),
            }
        )
        model = GPT(gptconf)
    # crop down the model block size if desired, using model surgery
    if isinstance(model, GPT) and block_size < model.config.block_size:
        model.crop_block_size(block_size)
    model.to(device)

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.GradScaler(enabled=(dtype == "float16"))

    # optimizer
    optimizer = model.configure_optimizers(
        weight_decay, learning_rate, (beta1, beta2), device_type
    )
    if checkpoint is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])  # pyright: ignore[reportArgumentType]
    checkpoint = None  # free up memory

    # compile the model
    if compile:
        print("compiling the model... (takes a ~minute)")
        # model = cast(torch.nn.Module, torch.compile(model, mode="max-autotune"))  # requires PyTorch 2.0
        model = cast(torch.nn.Module, torch.compile(model))  # requires PyTorch 2.0

    # wrap model into DDP container
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])  # pyright: ignore[reportPossiblyUnboundVariable]

    # Data loaders: pretokenized assignments-based vs uniform synthetic
    if use_pretokenized:
        # Custom assignments-driven dataloader that transforms tokens per assignments.bin
        train_loader = AssignmentsDataLoader(
            assignments_path,
            train_bin,
            batch_size,
            block_size,
            ddp_rank or 0,
            ddp_world_size,
            base_vocab,
            cast(int, exp_cfg.max_compartments),
            exp_cfg.n_compartments,
            permute_tokens=exp_cfg.permute_tokens_per_compartment,
            permutations_path=(
                permutations_path if exp_cfg.permute_tokens_per_compartment else None
            ),
            permute_inputs=exp_cfg.permute_input_tokens_per_compartment,
        )
        val_loader = None
        if val_bin:
            val_loader = AssignmentsDataLoader(
                assignments_path,
                val_bin,
                batch_size,
                block_size,
                ddp_rank or 0,
                ddp_world_size,
                base_vocab,
                cast(int, exp_cfg.max_compartments),
                exp_cfg.n_compartments,
                permute_tokens=exp_cfg.permute_tokens_per_compartment,
                permutations_path=(
                    permutations_path
                    if exp_cfg.permute_tokens_per_compartment
                    else None
                ),
                permute_inputs=exp_cfg.permute_input_tokens_per_compartment,
            )
    else:
        # Uniform synthetic stream
        # Check if we need compartment/translation support
        needs_compartments = exp_cfg.n_compartments > 1 or exp_cfg.translation_ratio > 0
        if needs_compartments:
            # Generate permutations in-memory if needed
            uniform_perms = None
            if exp_cfg.permute_tokens_per_compartment:
                max_c = cast(int, exp_cfg.max_compartments)
                ss = np.random.SeedSequence(int(training_seed) & 0xFFFFFFFFFFFFFFFF)
                child_seeds = ss.spawn(max_c)
                uniform_perms = np.empty((max_c, base_vocab), dtype=np.int64)
                for c, child_ss in enumerate(child_seeds):
                    gen = np.random.Generator(np.random.PCG64(child_ss))
                    uniform_perms[c] = gen.permutation(base_vocab).astype(np.int64)
                print0(
                    f"Generated in-memory permutations with shape {uniform_perms.shape}"
                )

            train_loader = UniformCompartmentDataLoader(
                B=batch_size,
                T=block_size,
                base_vocab_size=base_vocab,
                seed=config.data.uniform_seed + training_seed,
                n_compartments=exp_cfg.n_compartments,
                max_compartments=cast(int, exp_cfg.max_compartments),
                translation_ratio=exp_cfg.translation_ratio,
                translation_ratio_mode=exp_cfg.translation_ratio_mode,
                compartment_scaling=exp_cfg.compartment_scaling,
                process_rank=ddp_rank or 0,
                num_processes=ddp_world_size,
                permute_tokens=exp_cfg.permute_tokens_per_compartment,
                permutations=uniform_perms,
                permute_inputs=exp_cfg.permute_input_tokens_per_compartment,
                pin_memory=True,
            )
        else:
            # Simple single-compartment uniform data (original behavior)
            train_loader = UniformBatchDataLoader(
                B=batch_size,
                T=block_size,
                vocab_size=composite_vocab,
                seed=config.data.uniform_seed + training_seed,
                process_rank=ddp_rank or 0,
                num_processes=ddp_world_size,
                return_compartment_ids=bool(exp_cfg.use_compartment_embeddings),
                pin_memory=True,
            )
        val_loader = None

    # Restore dataloader state if resuming from checkpoint
    if dataloader_state is not None:
        print0(f"Restoring dataloader state from checkpoint (resuming at iter {iter_num})")
        train_loader.load_state_dict(dataloader_state["train"])
        if val_loader is not None and "val" in dataloader_state:
            val_loader.load_state_dict(dataloader_state["val"])

    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def estimate_loss():
        if val_loader is None:
            return None
        out = {}
        model.eval()
        for split in ["train", "val"]:
            # Reset the validation data loader to ensure deterministic eval on the same
            # data
            if split == "val":
                val_loader.reset()
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                batch = val_loader.next_batch()
                if isinstance(batch, tuple) and len(batch) == 3:
                    X, Y, Cval = batch
                else:
                    X, Y = batch  # type: ignore[misc]
                    Cval = None
                X, Y = X.to(device, non_blocking=True), Y.to(device, non_blocking=True)
                Cval = Cval.to(device, non_blocking=True) if Cval is not None else None
                with ctx:
                    # mark cudagraph step begin if available to avoid output overwrite
                    torch.compiler.cudagraph_mark_step_begin()
                    logits, loss = model(X, Y, compartment_ids=Cval)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    # # learning rate decay scheduler (cosine with warmup)
    # def get_lr(it):
    #     # 1) linear warmup for warmup_iters steps
    #     if it < warmup_iters:
    #         return learning_rate * (it + 1) / (warmup_iters + 1)
    #     # 2) if it > lr_decay_iters, return min learning rate
    #     if it > lr_decay_iters:
    #         return min_lr
    #     # 3) in between, use cosine decay down to min learning rate
    #     decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    #     assert 0 <= decay_ratio <= 1
    #     coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    #     return min_lr + coeff * (learning_rate - min_lr)

    # learning rate decay scheduler (warmup and no decay)
    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < warmup_iters:
            return learning_rate * (it + 1) / (warmup_iters + 1)
        # 2) if it > lr_decay_iters, return target learning rate
        return learning_rate

    # logging
    wandb_buffer_enabled = config.logging.wandb_buffer
    wandb_log_buffer: list[tuple[dict, int]] = []  # list of (metrics_dict, step)

    def wandb_log_or_buffer(metrics: dict, step: int) -> None:
        """Log to wandb directly, or buffer if wandb_buffer is enabled."""
        if wandb_buffer_enabled:
            wandb_log_buffer.append((metrics, step))
        else:
            import wandb
            wandb.log(metrics, step=step)

    def wandb_flush_buffer() -> None:
        """Flush all buffered wandb log entries."""
        if not wandb_log_buffer:
            return
        import wandb
        for metrics, step in wandb_log_buffer:
            wandb.log(metrics, step=step)
        wandb_log_buffer.clear()

    wandb_run: Optional[Any] = None
    if wandb_log and master_process:
        import wandb

        name_value: Optional[str] = None
        if isinstance(wandb_run_name, str):
            _nm = wandb_run_name.strip()
            if _nm and _nm.lower() != "sweep":
                name_value = wandb_run_name

        wandb_run = wandb.init(
            project=wandb_project,
            group=wandb_group,
            notes=wandb_notes,
            config={
                "cfg_hash": cfg_hash(config),
                "run_id": run_id,
                "out_dir": out_dir,
                **asdict(config),
            },
            dir=out_dir,
            id=run_id,
            resume="allow",
            name=name_value,
        )

    # training loop
    batch0 = train_loader.next_batch()  # fetch the very first batch
    if isinstance(batch0, tuple) and len(batch0) == 3:
        X, Y, C = batch0
    else:
        X, Y = batch0  # type: ignore[misc]
        C = None
    X, Y = X.to(device, non_blocking=True), Y.to(device, non_blocking=True)
    C = C.to(device, non_blocking=True) if C is not None else None
    t0 = time.time()
    local_iter_num = 0  # number of iterations in the lifetime of this process
    raw_model: GPT = (
        cast(GPT, model.module) if ddp else cast(GPT, model)
    )  # unwrap DDP container if needed
    running_mfu = -1.0
    while True:
        # determine and set the learning rate for this iteration
        # lr = get_lr(iter_num) if decay_lr else learning_rate
        lr = get_lr(iter_num)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % eval_interval == 0 and master_process:
            losses = None
            if eval_iters > 0 and val_loader is not None:
                losses = cast(dict[str, torch.Tensor], estimate_loss())
                print(
                    f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
                )
                if wandb_log and master_process:
                    wandb_log_or_buffer(
                        {
                            "iter": iter_num,
                            "train/loss": losses["train"],
                            "val/loss": losses["val"],
                            "lr": lr,
                            "mfu": running_mfu * 100,  # convert to percentage
                        },
                        step=iter_num,
                    )
            if losses is not None and losses["val"] < best_val_loss:
                best_val_loss = losses["val"]

        # Save checkpoints only at explicitly specified steps
        if (
            master_process
            and (iter_num % eval_interval == 0 or iter_num in checkpoint_steps)
            and iter_num > 0
        ):
            ck_root = os.path.join(out_dir, "checkpoints")
            step_dir = os.path.join(ck_root, f"step-{iter_num:06d}")
            os.makedirs(step_dir, exist_ok=True)
            print(f"saving checkpoint to {step_dir}")
            # Write all checkpoint files before updating the latest symlink,
            # so a SIGKILL can't leave latest pointing at an incomplete checkpoint.
            torch.save(raw_model.state_dict(), os.path.join(step_dir, "model.pt"))
            torch.save(optimizer.state_dict(), os.path.join(step_dir, "optimizer.pt"))
            # Save dataloader state for resumption
            dl_state = {"train": train_loader.state_dict()}
            if val_loader is not None:
                dl_state["val"] = val_loader.state_dict()
            torch.save(dl_state, os.path.join(step_dir, "dataloader.pt"))
            with open(os.path.join(step_dir, "trainer_state.json"), "w") as f:
                json.dump(
                    {
                        "iter_num": iter_num,
                        "best_val_loss": float(best_val_loss),
                    },
                    f,
                )
            # Atomically update latest symlink (after all files are written)
            latest = os.path.join(ck_root, "latest")
            tmp_link = latest + f".tmp.{os.getpid()}"
            os.symlink(os.path.basename(step_dir), tmp_link)
            os.replace(tmp_link, latest)
            # Flush buffered wandb logs now that checkpoint is durable
            if wandb_log and master_process and wandb_buffer_enabled:
                wandb_flush_buffer()

        if iter_num == 0 and eval_only:
            break

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        last_loss: Optional[torch.Tensor] = None
        for micro_step in range(gradient_accumulation_steps):
            if ddp:
                # in DDP training we only need to sync gradients at the last micro step.
                # the official way to do this is with model.no_sync() context manager, but
                # I really dislike that this bloats the code and forces us to repeat code
                # looking at the source of that context manager, it just toggles this variable
                cast(DDP, model).require_backward_grad_sync = (
                    micro_step == gradient_accumulation_steps - 1
                )
            with ctx:
                torch.compiler.cudagraph_mark_step_begin()
                logits, loss = model(X, Y, compartment_ids=C)
                loss = (
                    loss / gradient_accumulation_steps
                )  # scale the loss to account for gradient accumulation
            last_loss = loss
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            batch = train_loader.next_batch()
            if isinstance(batch, tuple) and len(batch) == 3:
                X, Y, C = batch
            else:
                X, Y = batch  # type: ignore[misc]
                C = None
            X, Y = X.to(device, non_blocking=True), Y.to(device, non_blocking=True)
            C = C.to(device, non_blocking=True) if C is not None else None
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()
        # clip the gradient
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % log_interval == 0 and master_process:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
            lossf = cast(torch.Tensor, last_loss).item() * gradient_accumulation_steps
            if local_iter_num >= 5:  # let the training loop settle a bit
                mfu = raw_model.estimate_mfu(
                    batch_size * gradient_accumulation_steps, dt
                )
                running_mfu = (
                    mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
                )
            print(
                f"iter {iter_num}: loss {lossf:.4f}, time {dt * 1000:.2f}ms, mfu {running_mfu * 100:.2f}%"
            )
            if wandb_log and master_process:
                wandb_log_or_buffer(
                    {
                        "iter": iter_num,
                        "train/loss": lossf,
                        "lr": lr,
                        "mfu": running_mfu * 100,
                        "time_ms": dt * 1000,
                    },
                    step=iter_num,
                )
        iter_num += 1
        local_iter_num += 1

        # termination conditions
        if iter_num > max_iters:
            break

    # Flush any remaining buffered wandb logs at normal termination
    if wandb_log and master_process and wandb_buffer_enabled:
        wandb_flush_buffer()

    if ddp:
        destroy_process_group()


if __name__ == "__main__":
    config_manager = ConfigManager()
    config = config_manager.parse_args()
    main(config)
