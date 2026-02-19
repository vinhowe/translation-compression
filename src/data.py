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
    tying_remap: Optional[np.ndarray] = None
    translation_token_id_override: Optional[int] = None
    pin_memory: bool = True
    # Translation mode: "standard" or "interleaved"
    translation_mode: Literal["standard", "interleaved"] = "standard"
    # Chunk size for interleaved mode
    translation_chunk_size: int = 4

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
        if self.translation_token_id_override is not None:
            self.translation_token_id = self.translation_token_id_override
        else:
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

        # Store tying remap as torch tensor for compact vocab layout
        if self.tying_remap is not None:
            self._remap_torch = torch.from_numpy(self.tying_remap).long()
        else:
            self._remap_torch = None

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
        if self.translation_mode == "interleaved":
            self._fill_translation_rows_interleaved(
                x, y, cids, idx_trans, srcs_t, dsts_t, all_tokens, m
            )
        else:
            self._fill_translation_rows_standard(
                x, y, cids, idx_trans, srcs_t, dsts_t, all_tokens, m
            )

    def _fill_translation_rows_standard(
        self, x, y, cids, idx_trans, srcs_t, dsts_t, all_tokens, m
    ):
        """Fill translation rows with standard format: [TRANS][src][TRANS][dst]."""
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
            if self._remap_torch is not None:
                # Compact remap: remap[comp, base_token] -> compact_id
                content_src = self._remap_torch[src_c.unsqueeze(1).expand_as(content), content]
                content_dst = self._remap_torch[dst_c.unsqueeze(1).expand_as(content), content]
            else:
                base = self.base_vocab_size
                content_src = content + (src_c * base).unsqueeze(1)
                content_dst = content + (dst_c * base).unsqueeze(1)

        # Build x for translation rows
        # Layout: [TRANS, src_content..., TRANS, dst_content...]
        x_trans = torch.empty((m, T), dtype=x.dtype, device=x.device)
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
        y_trans = torch.empty((m, T), dtype=y.dtype, device=y.device)
        y_trans[:, 0] = content_src[:, 0]  # after first TRANS
        y_trans[:, 1:half - 1] = content_src[:, 1:]
        y_trans[:, half - 1] = trans_token  # predict second TRANS
        y_trans[:, half] = content_dst[:, 0]  # after second TRANS
        y_trans[:, half + 1:-1] = content_dst[:, 1:]
        y_trans[:, -1] = -1  # ignore last position

        # Build cids
        cids_trans = torch.empty((m, T), dtype=cids.dtype, device=cids.device)
        cids_trans[:, :half] = src_c.unsqueeze(1)
        cids_trans[:, half:] = dst_c.unsqueeze(1)

        # Write back to output tensors
        x[idx_trans] = x_trans
        y[idx_trans] = y_trans
        cids[idx_trans] = cids_trans

    def _fill_translation_rows_interleaved(
        self, x, y, cids, idx_trans, srcs_t, dsts_t, all_tokens, m
    ):
        """Fill translation rows with interleaved format: [TRANS][src_chunk][dst_chunk]..."""
        T = self.T
        chunk = self.translation_chunk_size
        trans_token = self.translation_token_id

        # Get source/dest compartments for translation rows
        src_c = srcs_t[idx_trans]
        dst_c = dsts_t[idx_trans]

        # Calculate how many complete chunk pairs fit after TRANS token
        # Layout: [TRANS][chunk_src][chunk_dst][chunk_src][chunk_dst]...
        content_positions = T - 1  # positions after TRANS
        pair_size = 2 * chunk
        n_complete_pairs = content_positions // pair_size
        remaining = content_positions - n_complete_pairs * pair_size

        # Handle partial final chunks
        # remaining positions split as: min(remaining, chunk) for src, rest for dst
        partial_src = min(remaining, chunk)
        partial_dst = remaining - partial_src

        # Total content tokens needed (for complete pairs + partial)
        n_content_tokens = n_complete_pairs * chunk + max(partial_src, partial_dst)

        # Extract tokens for content (same content, different permutations)
        tokens = all_tokens[: m * n_content_tokens].view(m, n_content_tokens)

        # Apply permutations if needed
        if self.permute_tokens and self._perm_torch is not None:
            perm_src = self._perm_torch[src_c]  # (m, vocab)
            perm_dst = self._perm_torch[dst_c]  # (m, vocab)
            content_src = torch.gather(perm_src, 1, tokens)
            content_dst = torch.gather(perm_dst, 1, tokens)
        else:
            base = self.base_vocab_size
            content_src = tokens + (src_c * base).unsqueeze(1)
            content_dst = tokens + (dst_c * base).unsqueeze(1)

        # Build x for translation rows
        x_trans = torch.zeros((m, T), dtype=x.dtype, device=x.device)
        x_trans[:, 0] = trans_token

        # Fill complete interleaved chunks
        for p in range(n_complete_pairs):
            src_start = p * chunk
            src_end = src_start + chunk
            x_src_start = 1 + p * pair_size
            x_dst_start = x_src_start + chunk
            if self.permute_inputs:
                x_trans[:, x_src_start:x_src_start + chunk] = content_src[:, src_start:src_end]
                x_trans[:, x_dst_start:x_dst_start + chunk] = content_dst[:, src_start:src_end]
            else:
                x_trans[:, x_src_start:x_src_start + chunk] = tokens[:, src_start:src_end]
                x_trans[:, x_dst_start:x_dst_start + chunk] = tokens[:, src_start:src_end]

        # Fill partial final chunks if any
        if partial_src > 0:
            partial_content_start = n_complete_pairs * chunk
            x_partial_src_start = 1 + n_complete_pairs * pair_size
            if self.permute_inputs:
                x_trans[:, x_partial_src_start:x_partial_src_start + partial_src] = \
                    content_src[:, partial_content_start:partial_content_start + partial_src]
            else:
                x_trans[:, x_partial_src_start:x_partial_src_start + partial_src] = \
                    tokens[:, partial_content_start:partial_content_start + partial_src]

            if partial_dst > 0:
                x_partial_dst_start = x_partial_src_start + partial_src
                if self.permute_inputs:
                    x_trans[:, x_partial_dst_start:x_partial_dst_start + partial_dst] = \
                        content_dst[:, partial_content_start:partial_content_start + partial_dst]
                else:
                    x_trans[:, x_partial_dst_start:x_partial_dst_start + partial_dst] = \
                        tokens[:, partial_content_start:partial_content_start + partial_dst]

        # Build y (next token prediction targets)
        # y[t] = x[t+1] for autoregressive
        y_trans = torch.full((m, T), -1, dtype=y.dtype, device=y.device)

        # y[0] predicts x[1] (first src token after TRANS)
        y_trans[:, 0] = content_src[:, 0]

        # Fill y for complete pairs
        for p in range(n_complete_pairs):
            src_start = p * chunk
            x_src_start = 1 + p * pair_size
            x_dst_start = x_src_start + chunk

            # Within src chunk: predict next src token, last src predicts first dst
            for i in range(chunk - 1):
                y_trans[:, x_src_start + i] = content_src[:, src_start + i + 1]
            y_trans[:, x_src_start + chunk - 1] = content_dst[:, src_start]

            # Within dst chunk: predict next dst token
            for i in range(chunk - 1):
                y_trans[:, x_dst_start + i] = content_dst[:, src_start + i + 1]

            # Last dst token: predict first of next chunk (src or partial)
            if p < n_complete_pairs - 1:
                next_src_start = (p + 1) * chunk
                y_trans[:, x_dst_start + chunk - 1] = content_src[:, next_src_start]
            elif partial_src > 0:
                # Predict first token of partial src chunk
                y_trans[:, x_dst_start + chunk - 1] = content_src[:, n_complete_pairs * chunk]

        # Fill y for partial chunks
        if partial_src > 0:
            partial_content_start = n_complete_pairs * chunk
            x_partial_src_start = 1 + n_complete_pairs * pair_size

            # Within partial src chunk
            for i in range(partial_src - 1):
                y_trans[:, x_partial_src_start + i] = content_src[:, partial_content_start + i + 1]

            if partial_dst > 0:
                # Last partial src predicts first partial dst
                y_trans[:, x_partial_src_start + partial_src - 1] = content_dst[:, partial_content_start]

                # Within partial dst chunk
                x_partial_dst_start = x_partial_src_start + partial_src
                for i in range(partial_dst - 1):
                    y_trans[:, x_partial_dst_start + i] = content_dst[:, partial_content_start + i + 1]
                # Last position is -1 (already set)

        # Build cids - alternating src/dst for each chunk
        cids_trans = torch.zeros((m, T), dtype=cids.dtype, device=cids.device)
        cids_trans[:, 0] = src_c  # TRANS token gets src compartment id

        for p in range(n_complete_pairs):
            x_src_start = 1 + p * pair_size
            x_dst_start = x_src_start + chunk
            cids_trans[:, x_src_start:x_src_start + chunk] = src_c.unsqueeze(1)
            cids_trans[:, x_dst_start:x_dst_start + chunk] = dst_c.unsqueeze(1)

        # Partial chunks cids
        if partial_src > 0:
            x_partial_src_start = 1 + n_complete_pairs * pair_size
            cids_trans[:, x_partial_src_start:x_partial_src_start + partial_src] = src_c.unsqueeze(1)
            if partial_dst > 0:
                x_partial_dst_start = x_partial_src_start + partial_src
                cids_trans[:, x_partial_dst_start:x_partial_dst_start + partial_dst] = dst_c.unsqueeze(1)

        # Write back to output tensors
        x[idx_trans] = x_trans
        y[idx_trans] = y_trans
        cids[idx_trans] = cids_trans

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
            if self._remap_torch is not None:
                x_comp = self._remap_torch[src_c.unsqueeze(1).expand_as(tokens), tokens]
            else:
                base = self.base_vocab_size
                x_comp = tokens + (src_c * base).unsqueeze(1)
            y_comp = torch.empty_like(x_comp)
            y_comp[:, :-1] = x_comp[:, 1:]
            y_comp[:, -1] = -1

        x[idx_comp] = x_comp
        y[idx_comp] = y_comp
        cids[idx_comp] = src_c.unsqueeze(1).expand(-1, T)
