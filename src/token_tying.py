"""Token tying: share a subset of tokens across compartments.

Provides utilities to compute token frequencies from data shards,
determine which tokens should be tied (shared) across compartments,
and modify permutation tables to enforce tying constraints.
"""

import glob
import numpy as np


def compute_token_frequencies(
    file_pattern: str, vocab_size: int, max_shards: int = 1
) -> np.ndarray:
    """Scan the first `max_shards` sorted data shards and count per-token occurrences.

    Args:
        file_pattern: Glob pattern for .bin data shards.
        vocab_size: Base vocabulary size (determines bincount range).
        max_shards: Number of shards to read (default 1).

    Returns:
        uint64 array of shape [vocab_size] with per-token counts.
    """
    files = sorted(glob.glob(file_pattern))
    if not files:
        raise FileNotFoundError(f"No files match pattern: {file_pattern}")

    files = files[:max_shards]
    counts = np.zeros(vocab_size, dtype=np.uint64)

    for fname in files:
        with open(fname, "rb") as f:
            header = np.frombuffer(f.read(256 * 4), dtype=np.int32)
            assert header[0] == 20251013, "magic number mismatch in data .bin file"
            assert header[1] == 1, "unsupported version"
            tokens = np.frombuffer(f.read(), dtype=np.uint32)
        shard_counts = np.bincount(tokens, minlength=vocab_size).astype(np.uint64)
        # Truncate if tokens exceed vocab_size (shouldn't happen, but be safe)
        counts += shard_counts[:vocab_size]

    return counts


def compute_tied_mask(
    freqs: np.ndarray, mode: str, target: float
) -> np.ndarray:
    """Compute a boolean mask indicating which tokens are tied (shared).

    The mask is True for tokens that should be identical across compartments.
    We tie tokens until the *untied* fraction of total mass is <= target.

    Args:
        freqs: uint64 array of shape [vocab_size] with per-token frequency counts.
        mode: "top_k" to tie most frequent tokens first, "bottom_k" to tie least frequent first.
        target: Target fraction of token mass that remains untied. 0 = all tied, 1 = none tied.

    Returns:
        Boolean array of shape [vocab_size] — True = tied.
    """
    total_mass = freqs.sum()
    if total_mass == 0:
        return np.zeros(len(freqs), dtype=bool)

    # target=1 means nothing is tied
    if target >= 1.0:
        return np.zeros(len(freqs), dtype=bool)
    # target=0 means everything is tied
    if target <= 0.0:
        return np.ones(len(freqs), dtype=bool)

    if mode == "top_k":
        # Sort by frequency descending — tie the most frequent first
        order = np.argsort(freqs)[::-1]
    elif mode == "bottom_k":
        # Sort by frequency ascending — tie the least frequent first
        order = np.argsort(freqs)
    else:
        raise ValueError(f"Unknown token tying mode: {mode!r}")

    tied_mask = np.zeros(len(freqs), dtype=bool)
    tied_mass = np.uint64(0)

    for idx in order:
        tied_mass += freqs[idx]
        tied_mask[idx] = True
        # Check: is the remaining untied mass <= target fraction?
        untied_mass = total_mass - tied_mass
        if untied_mass / total_mass <= target:
            break

    return tied_mask


def apply_tying_to_permutations(
    perms: np.ndarray, tied_mask: np.ndarray
) -> np.ndarray:
    """Modify permutation table so tied tokens map to identity.

    For tied positions, all compartments map token i -> i (identity).
    For untied positions, each compartment's permutation is a valid
    permutation of the untied token set only.

    Args:
        perms: int64 array of shape [n_compartments, vocab_size] — original permutations.
        tied_mask: bool array of shape [vocab_size] — True = tied.

    Returns:
        Modified permutations array (new copy).
    """
    n_compartments, vocab_size = perms.shape
    new_perms = np.empty_like(perms)

    tied_indices = np.nonzero(tied_mask)[0]
    untied_indices = np.nonzero(~tied_mask)[0]

    for c in range(n_compartments):
        # Tied tokens: identity
        new_perms[c, tied_indices] = tied_indices

        # Untied tokens: the original permutation maps untied positions
        # to some set of outputs. We need to ensure untied inputs map
        # only to untied outputs, forming a valid permutation of the untied set.
        # Extract where the original perm sends untied positions
        orig_outputs = perms[c, untied_indices]
        # These outputs may include tied positions; we need to re-permute
        # so that untied inputs map to untied outputs only.
        # Use the relative ordering from the original permutation to
        # create a consistent permutation of untied_indices.
        # Sort orig_outputs to get a ranking, then use that ranking
        # to index into the sorted untied_indices.
        rank = np.argsort(np.argsort(orig_outputs))
        sorted_untied = np.sort(untied_indices)
        new_perms[c, untied_indices] = sorted_untied[rank]

    return new_perms
