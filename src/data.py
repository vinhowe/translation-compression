from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import torch


@dataclass
class UniformDataset:
    vocab_size: int
    seq_len: int
    size: int
    seed: int

    def __len__(self):
        return self.size

    def __getitem__(self, idx: int) -> torch.Tensor:
        gen = torch.Generator()
        gen.manual_seed(self.seed + idx)
        return torch.randint(
            low=0, high=self.vocab_size, size=(self.seq_len,), generator=gen
        )


@dataclass
class UniformBatchDataLoader:
    B: int
    T: int
    vocab_size: int
    seed: int
    process_rank: int = 0
    num_processes: int = 1
    return_compartment_ids: bool = False
    pin_memory: bool = True

    def __post_init__(self) -> None:
        self._pin = bool(self.pin_memory and torch.cuda.is_available())
        self._gen = torch.Generator()
        # ensure different stream per rank; deterministic across runs
        self._gen.manual_seed(int(self.seed) + int(self.process_rank))

    def reset(self) -> None:
        # re-seed to initial state for deterministic epoch resets if needed
        self._gen.manual_seed(int(self.seed) + int(self.process_rank))

    def next_batch(self):
        x = torch.randint(
            0,
            int(self.vocab_size),
            (int(self.B), int(self.T)),
            generator=self._gen,
            dtype=torch.long,
            pin_memory=self._pin,
        )
        y = torch.empty_like(x)
        if self.T > 0:
            y[:, :-1] = x[:, 1:]
            y[:, -1] = -1
        else:
            y.fill_(-1)
        if self.return_compartment_ids:
            # single compartment id 0 by default for all tokens
            cids = torch.zeros(
                (int(self.B), int(self.T)), dtype=torch.long, pin_memory=self._pin
            )
            return x, y, cids
        return x, y


@dataclass
class UniformCompartmentDataLoader:
    """Uniform random data loader with compartment and translation support.

    Generates random tokens and applies compartment/translation structure
    matching the behavior of AssignmentsDataLoader but without needing
    pretokenized data files.
    """

    B: int
    T: int
    base_vocab_size: int
    seed: int
    n_compartments: int
    max_compartments: int
    translation_ratio: float = 0.0
    translation_ratio_mode: Literal["compartment", "absolute"] = "compartment"
    compartment_scaling: Literal["equal", "unequal"] = "equal"
    process_rank: int = 0
    num_processes: int = 1
    permute_tokens: bool = True
    permutations: Optional[np.ndarray] = None
    permute_inputs: bool = True
    pin_memory: bool = True

    def __post_init__(self) -> None:
        self._pin = bool(self.pin_memory and torch.cuda.is_available())

        # Compute weights for compartments and translations
        from src.weights import compute_weights_map

        self._weights_map = compute_weights_map(
            n=self.n_compartments,
            t=self.translation_ratio,
            scaling=self.compartment_scaling,
            mode=self.translation_ratio_mode,
        )

        # Build category list and probability array for sampling
        self._categories: list[tuple[str, int, int]] = []  # (kind, src, dst)
        probs: list[float] = []
        for key, prob in sorted(self._weights_map.items()):
            if ">" in key:
                left, right = key.split(">", 1)
                self._categories.append(("T", int(left), int(right)))
            else:
                self._categories.append(("C", int(key), 0))
            probs.append(prob)
        self._probs = np.array(probs, dtype=np.float64)
        self._probs /= self._probs.sum()  # normalize

        # Translation token id
        self.translation_token_id = (
            self.base_vocab_size
            if self.permute_tokens
            else self.base_vocab_size * self.max_compartments
        )

        # Initialize RNGs with different seeds for categories vs tokens
        # This ensures category sampling and token generation are independent streams
        self._cat_rng = np.random.Generator(
            np.random.PCG64(int(self.seed) + int(self.process_rank))
        )
        self._tok_gen = torch.Generator()
        # Use a different offset for token generation to ensure independence
        self._tok_gen.manual_seed(int(self.seed) + int(self.process_rank) + 1_000_000)

        # Store permutations if provided
        self._permutations = self.permutations

    def reset(self) -> None:
        """Re-seed RNGs to initial state for deterministic resets."""
        self._cat_rng = np.random.Generator(
            np.random.PCG64(int(self.seed) + int(self.process_rank))
        )
        self._tok_gen.manual_seed(int(self.seed) + int(self.process_rank) + 1_000_000)

    def next_batch(self):
        B = self.B
        T = self.T
        half = T // 2
        assert T % 2 == 0 and half >= 2, "block_size must be even and >= 4"

        # Allocate output tensors
        x_t = torch.empty((B, T), dtype=torch.long, pin_memory=self._pin)
        y_t = torch.empty((B, T), dtype=torch.long, pin_memory=self._pin)
        cids_t = torch.empty((B, T), dtype=torch.long, pin_memory=self._pin)

        x_np = x_t.numpy()
        y_np = y_t.numpy()
        cid_np = cids_t.numpy()

        # Sample categories for each batch item
        cat_indices = self._cat_rng.choice(len(self._categories), size=B, p=self._probs)

        # Decode categories
        kinds = np.array([self._categories[i][0] for i in cat_indices])
        srcs = np.array([self._categories[i][1] for i in cat_indices], dtype=np.int64)
        dsts = np.array([self._categories[i][2] for i in cat_indices], dtype=np.int64)

        is_trans = kinds == "T"
        idx_trans = np.nonzero(is_trans)[0]
        idx_comp = np.nonzero(~is_trans)[0]

        # Generate random base tokens for all items
        # Translation items need half tokens, compartment items need T tokens
        size_per_b = np.where(is_trans, half, T).astype(np.int64)
        total_needed = int(size_per_b.sum())

        tokens_flat = torch.randint(
            0,
            self.base_vocab_size,
            (total_needed,),
            generator=self._tok_gen,
            dtype=torch.long,
        ).numpy()

        # Compute start indices for each batch item's tokens
        starts = np.zeros(B, dtype=np.int64)
        if B > 1:
            starts[1:] = np.cumsum(size_per_b[:-1], dtype=np.int64)

        # Process translation examples
        if idx_trans.size > 0:
            m = int(idx_trans.size)
            base_idx_half = np.arange(half, dtype=np.int64)[None, :]
            trans_starts = starts[idx_trans][:, None]
            samples = tokens_flat[trans_starts + base_idx_half]

            # Prepare input/output seq and cids for translation rows
            seq_in = np.empty((m, T), dtype=np.int64)
            seq_out = np.empty((m, T), dtype=np.int64)
            cid_tr = np.empty((m, T), dtype=np.int64)

            # Set translation token positions
            seq_in[:, 0] = self.translation_token_id
            seq_in[:, half] = self.translation_token_id
            seq_out[:, 0] = self.translation_token_id
            seq_out[:, half] = self.translation_token_id
            cid_tr[:, 0] = srcs[idx_trans]
            cid_tr[:, half] = dsts[idx_trans]

            if self.permute_tokens and self._permutations is not None:
                # Per-row permutations
                src_maps = self._permutations[srcs[idx_trans]]
                dst_maps = self._permutations[dsts[idx_trans]]
                # Map tokens row-wise
                perm_src = np.take_along_axis(src_maps, samples[:, : half - 1], axis=1)
                perm_dst = np.take_along_axis(dst_maps, samples[:, : half - 1], axis=1)
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

        # Process compartment examples
        if idx_comp.size > 0:
            base_idx_T = np.arange(T, dtype=np.int64)[None, :]
            comp_starts = starts[idx_comp][:, None]
            samples = tokens_flat[comp_starts + base_idx_T]

            if self.permute_tokens and self._permutations is not None:
                src_maps = self._permutations[srcs[idx_comp]]
                perm_samples = np.take_along_axis(src_maps, samples, axis=1)
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

        return x_t, y_t, cids_t
