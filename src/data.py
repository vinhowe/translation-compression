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

    def state_dict(self) -> dict:
        return {"gen_state": self._gen.get_state()}

    def load_state_dict(self, state: dict) -> None:
        self._gen.set_state(state["gen_state"])

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
    tied_token_mask: Optional[np.ndarray] = None
    pin_memory: bool = True

    def __post_init__(self) -> None:
        self._pin = bool(self.pin_memory and torch.cuda.is_available())
        B, T = self.B, self.T
        half = T // 2

        # Compute weights for compartments and translations
        from src.weights import compute_weights_map

        self._weights_map = compute_weights_map(
            n=self.n_compartments,
            t=self.translation_ratio,
            scaling=self.compartment_scaling,
            mode=self.translation_ratio_mode,
        )

        # Build category arrays for vectorized sampling (avoid string comparisons)
        # kind: 0 = compartment, 1 = translation
        cat_kinds: list[int] = []
        cat_srcs: list[int] = []
        cat_dsts: list[int] = []
        probs: list[float] = []
        for key, prob in sorted(self._weights_map.items()):
            if ">" in key:
                left, right = key.split(">", 1)
                cat_kinds.append(1)  # translation
                cat_srcs.append(int(left))
                cat_dsts.append(int(right))
            else:
                cat_kinds.append(0)  # compartment
                cat_srcs.append(int(key))
                cat_dsts.append(0)
            probs.append(prob)

        self._cat_kinds = np.array(cat_kinds, dtype=np.int64)
        self._cat_srcs = np.array(cat_srcs, dtype=np.int64)
        self._cat_dsts = np.array(cat_dsts, dtype=np.int64)
        self._probs = np.array(probs, dtype=np.float64)
        self._probs /= self._probs.sum()  # normalize
        self._num_categories = len(cat_kinds)

        # Translation token id - uses n_compartments to match model vocab size
        self.translation_token_id = (
            self.base_vocab_size
            if self.permute_tokens
            else self.base_vocab_size * self.n_compartments
        )

        # Initialize RNGs with different seeds for categories vs tokens
        self._cat_rng = np.random.Generator(
            np.random.PCG64(int(self.seed) + int(self.process_rank))
        )
        self._tok_gen = torch.Generator()
        self._tok_gen.manual_seed(int(self.seed) + int(self.process_rank) + 1_000_000)

        # Store tied token mask as torch tensor for offset-mode tying
        if self.tied_token_mask is not None:
            self._tied_mask_torch = torch.from_numpy(self.tied_token_mask).bool()
        else:
            self._tied_mask_torch = None

        # Store permutations as torch tensor for faster indexing
        if self.permutations is not None:
            self._perm_torch = torch.from_numpy(self.permutations).long()
            if self._pin:
                self._perm_torch = self._perm_torch.pin_memory()
        else:
            self._perm_torch = None

        # Pre-allocate output buffers (reused every batch)
        self._x = torch.empty((B, T), dtype=torch.long, pin_memory=self._pin)
        self._y = torch.empty((B, T), dtype=torch.long, pin_memory=self._pin)
        self._cids = torch.empty((B, T), dtype=torch.long, pin_memory=self._pin)

        # Pre-allocate work buffers for translation rows (worst case: all translation)
        self._seq_in = torch.empty((B, T), dtype=torch.long, pin_memory=self._pin)
        self._seq_out = torch.empty((B, T), dtype=torch.long, pin_memory=self._pin)

        # Pre-compute index tensors that don't change
        self._half = half
        self._arange_half = torch.arange(half, dtype=torch.long)
        self._arange_T = torch.arange(T, dtype=torch.long)
        self._arange_B = torch.arange(B, dtype=torch.long)

        # Pre-allocate token buffer (worst case: B * T tokens)
        self._all_tokens = torch.empty(B * T, dtype=torch.long)

        # Pre-allocate numpy buffers for vectorized operations
        self._kinds = np.empty(B, dtype=np.int64)
        self._srcs_np = np.empty(B, dtype=np.int64)
        self._dsts_np = np.empty(B, dtype=np.int64)

    def state_dict(self) -> dict:
        return {
            "cat_rng_state": self._cat_rng.bit_generator.state,
            "tok_gen_state": self._tok_gen.get_state(),
        }

    def load_state_dict(self, state: dict) -> None:
        self._cat_rng.bit_generator.state = state["cat_rng_state"]
        self._tok_gen.set_state(state["tok_gen_state"])

    def reset(self) -> None:
        """Re-seed RNGs to initial state for deterministic resets."""
        self._cat_rng = np.random.Generator(
            np.random.PCG64(int(self.seed) + int(self.process_rank))
        )
        self._tok_gen.manual_seed(int(self.seed) + int(self.process_rank) + 1_000_000)

    def next_batch(self):
        B, T, half = self.B, self.T, self._half

        # Sample categories for each batch item
        cat_indices = self._cat_rng.choice(self._num_categories, size=B, p=self._probs)

        # Vectorized category decode using pre-allocated buffers
        np.take(self._cat_kinds, cat_indices, out=self._kinds)
        np.take(self._cat_srcs, cat_indices, out=self._srcs_np)
        np.take(self._cat_dsts, cat_indices, out=self._dsts_np)

        is_trans = self._kinds == 1
        n_trans = is_trans.sum()
        n_comp = B - n_trans

        # Convert to torch (views, no copy)
        srcs_t = torch.from_numpy(self._srcs_np)
        dsts_t = torch.from_numpy(self._dsts_np)

        # Generate random tokens into pre-allocated buffer
        torch.randint(
            0, self.base_vocab_size, (B * T,),
            generator=self._tok_gen, dtype=torch.long,
            out=self._all_tokens
        )
        all_tokens = self._all_tokens

        # Get references to pre-allocated buffers
        x = self._x
        y = self._y
        cids = self._cids

        if n_trans > 0 and n_comp > 0:
            # Mixed batch - process both types
            idx_trans = torch.from_numpy(np.nonzero(is_trans)[0])
            idx_comp = torch.from_numpy(np.nonzero(~is_trans)[0])

            # Process translation rows
            self._fill_translation_rows(
                x, y, cids, idx_trans, srcs_t, dsts_t, all_tokens, n_trans
            )

            # Process compartment rows
            self._fill_compartment_rows(
                x, y, cids, idx_comp, srcs_t, all_tokens, n_trans, n_comp
            )

        elif n_trans > 0:
            # All translation
            idx_trans = self._arange_B
            self._fill_translation_rows(
                x, y, cids, idx_trans, srcs_t, dsts_t, all_tokens, n_trans
            )

        else:
            # All compartment
            idx_comp = self._arange_B
            self._fill_compartment_rows(
                x, y, cids, idx_comp, srcs_t, all_tokens, 0, n_comp
            )

        return x, y, cids

    def _fill_translation_rows(
        self, x, y, cids, idx_trans, srcs_t, dsts_t, all_tokens, m
    ):
        """Fill translation rows into output tensors."""
        half, T = self._half, self.T
        trans_token = self.translation_token_id

        # Get source/dest compartments for translation rows
        src_c = srcs_t[idx_trans]
        dst_c = dsts_t[idx_trans]

        # Extract tokens for translation rows (half tokens each, starting at offset 0)
        # tokens shape: (m, half)
        tokens = all_tokens[: m * half].view(m, half)
        content = tokens[:, : half - 1]  # actual content (excluding last position)

        # Apply permutations if needed
        if self.permute_tokens and self._perm_torch is not None:
            # Gather permutation maps for each row's src/dst compartment
            perm_src = self._perm_torch[src_c]  # (m, vocab)
            perm_dst = self._perm_torch[dst_c]  # (m, vocab)
            # Apply permutation: perm[content] using gather
            content_src = torch.gather(perm_src, 1, content)
            content_dst = torch.gather(perm_dst, 1, content)
        else:
            # Offset-based (no permutation)
            base = self.base_vocab_size
            if self._tied_mask_torch is not None:
                tied_slice = self._tied_mask_torch[content]  # bool [m, half-1]
                src_off = torch.where(tied_slice, 0, (src_c * base).unsqueeze(1))
                dst_off = torch.where(tied_slice, 0, (dst_c * base).unsqueeze(1))
            else:
                src_off = (src_c * base).unsqueeze(1)
                dst_off = (dst_c * base).unsqueeze(1)
            content_src = content + src_off
            content_dst = content + dst_off

        # Build x for translation rows
        # Layout: [TRANS, src_content..., TRANS, dst_content...]
        x_trans = x[idx_trans]
        x_trans[:, 0] = trans_token
        x_trans[:, half] = trans_token
        if self.permute_inputs:
            x_trans[:, 1:half] = content_src
            x_trans[:, half + 1:] = content_dst
        else:
            x_trans[:, 1:half] = content
            x_trans[:, half + 1:] = content

        # Build y (next token prediction targets)
        # y[t] = x[t+1] for autoregressive, with permuted targets
        y_trans = y[idx_trans]
        y_trans[:, 0] = content_src[:, 0]  # after first TRANS
        y_trans[:, 1:half - 1] = content_src[:, 1:]
        y_trans[:, half - 1] = trans_token  # predict second TRANS
        y_trans[:, half] = content_dst[:, 0]  # after second TRANS
        y_trans[:, half + 1:-1] = content_dst[:, 1:]
        y_trans[:, -1] = -1  # ignore last position

        # Build cids
        cids_trans = cids[idx_trans]
        cids_trans[:, :half] = src_c.unsqueeze(1)
        cids_trans[:, half:] = dst_c.unsqueeze(1)

    def _fill_compartment_rows(
        self, x, y, cids, idx_comp, srcs_t, all_tokens, token_offset, m
    ):
        """Fill compartment rows into output tensors."""
        T = self.T
        half = self._half

        # Get source compartments for compartment rows
        src_c = srcs_t[idx_comp]

        # Extract tokens for compartment rows
        # Start after translation tokens (token_offset = n_trans * half)
        start = token_offset * half
        tokens = all_tokens[start: start + m * T].view(m, T)

        # Apply permutations if needed
        if self.permute_tokens and self._perm_torch is not None:
            perm_src = self._perm_torch[src_c]  # (m, vocab)
            perm_tokens = torch.gather(perm_src, 1, tokens)
            if self.permute_inputs:
                x_comp = perm_tokens
            else:
                x_comp = tokens
            # Targets always in permuted space
            y_comp = torch.empty_like(perm_tokens)
            y_comp[:, :-1] = perm_tokens[:, 1:]
            y_comp[:, -1] = -1
        else:
            base = self.base_vocab_size
            if self._tied_mask_torch is not None:
                tied_slice = self._tied_mask_torch[tokens]  # bool [m, T]
                src_off = torch.where(tied_slice, 0, (src_c * base).unsqueeze(1))
            else:
                src_off = (src_c * base).unsqueeze(1)
            x_comp = tokens + src_off
            y_comp = torch.empty_like(x_comp)
            y_comp[:, :-1] = x_comp[:, 1:]
            y_comp[:, -1] = -1

        x[idx_comp] = x_comp
        y[idx_comp] = y_comp
        cids[idx_comp] = src_c.unsqueeze(1).expand(-1, T)
