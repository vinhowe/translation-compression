import math
import os
import random
import struct
from dataclasses import dataclass
from typing import List

from tqdm.auto import tqdm
import numpy as np


MAGIC = b"TCASSIGN"  # 8 bytes
VERSION = 1
RECORD_SIZE = 8  # bytes
HEADER_SIZE = 32  # bytes
HEADER_STRUCT = struct.Struct("<8sBBHIQQ")


@dataclass(frozen=True)
class Category:
    kind: str  # 'C' or 'T'
    src: int
    dst: int

    def key(self) -> tuple[int, int, int]:
        return (0 if self.kind == "C" else 1, self.src, self.dst)


def _largest_remainder_allocations(
    weights: List[tuple[Category, float]], total: int
) -> List[tuple[Category, int]]:
    if total < 0:
        raise ValueError("Total must be non-negative")
    sum_w = sum(w for _, w in weights)
    if total == 0:
        return [(cat, 0) for cat, _ in weights]
    if sum_w <= 0:
        raise ValueError("Sum of weights must be > 0 when total > 0")

    floors: List[int] = []
    fracs: List[float] = []
    for cat, w in weights:
        exact = (total * w) / sum_w
        floor_v = int(math.floor(exact))
        frac = exact - floor_v
        floors.append(floor_v)
        fracs.append(frac)

    allocated = sum(floors)
    remainder = total - allocated
    order = [cat.key() for cat, _ in weights]
    idxs = list(range(len(weights)))
    idxs.sort(key=lambda i: (-fracs[i], order[i]))

    counts = floors[:]
    for i in range(remainder):
        counts[idxs[i]] += 1

    return [(weights[i][0], counts[i]) for i in range(len(weights))]


def _encode_record_u64(kind: int, src: int, dst: int) -> int:
    if not (0 <= kind <= 255):
        raise ValueError("kind must fit in u8")
    if not (0 <= src <= 0xFFFF and 0 <= dst <= 0xFFFF):
        raise ValueError("compartment ids must fit in u16")
    rec_flags = 0
    value = kind & 0xFF
    value |= (rec_flags & 0xFF) << 8
    value |= (src & 0xFFFF) << 16
    value |= (dst & 0xFFFF) << 32
    return value  # top 16 bits reserved 0


def _choose_lcg_params(total: int, seed: int) -> tuple[int, int]:
    """Choose LCG parameters (a, b) that yield a permutation modulo total.

    For modulus N, the map i -> (a * i + b) % N is a permutation iff gcd(a, N) = 1.
    We derive a and b deterministically from seed.
    """
    if total <= 0:
        return 1, 0
    rnd = random.Random(int(seed) & 0xFFFFFFFFFFFFFFFF)
    # pick b in [0, N-1]
    b = rnd.randrange(0, total)
    # try a few random candidates for a, ensure gcd(a, N) == 1
    for _ in range(16):
        a = rnd.randrange(1, total)
        if math.gcd(a, total) == 1:
            return a, b
    # fallback: a = total - 1 is always coprime with total
    a = max(1, total - 1)
    return a, b


def _prepare_allocation_arrays(
    allocations: List[tuple[Category, int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Return (encoded_values, counts, cumulative_counts, total_records)."""
    K = len(allocations)
    encoded = np.empty(K, dtype=np.uint64)
    counts = np.empty(K, dtype=np.uint64)
    for i, (cat, count) in enumerate(allocations):
        if count < 0:
            raise ValueError("allocation count must be non-negative")
        if cat.kind == "C":
            kind_code = 0
            src, dst = cat.src, 0
        else:
            kind_code = 1
            src, dst = cat.src, cat.dst
        encoded[i] = np.uint64(_encode_record_u64(kind_code, src, dst))
        counts[i] = np.uint64(count)
    cum = counts.cumsum(dtype=np.uint64)
    total = int(cum[-1]) if K > 0 else 0
    return encoded, counts, cum, total


def _write_records_streaming_lcg(
    f, encoded_values: np.ndarray, cumulative_counts: np.ndarray, total: int, seed: int
) -> None:
    """Write records in a single pass using an LCG permutation and vectorized mapping."""
    a, b = _choose_lcg_params(total, seed)
    a_u = np.uint64(a)
    b_u = np.uint64(b)
    N_u = np.uint64(total)
    # Tune block size: keep memory modest; ~2M elements => ~16MB per array
    block_size = 2_000_000
    written = 0
    with tqdm(total=total, desc="Writing assignment records", unit="rec") as pbar:
        while written < total:
            m = min(block_size, total - written)
            base = np.uint64(written)
            i = base + np.arange(m, dtype=np.uint64)
            j = (a_u * i + b_u) % N_u
            idx = np.searchsorted(cumulative_counts, j, side="right")
            out = np.empty(m, dtype=np.uint64)
            np.take(encoded_values, idx, out=out)
            out.tofile(f)
            written += m
            pbar.update(m)


def _write_records_in_order(f, encoded_values: np.ndarray, counts: np.ndarray) -> None:
    """Write records grouped by category without shuffling, in streaming chunks."""
    chunk = 2_000_000
    for i in tqdm(
        range(len(encoded_values)), desc="Writing assignment records (ordered)"
    ):
        val = encoded_values[i]
        remaining = int(counts[i])
        while remaining > 0:
            m = min(chunk, remaining)
            buf = np.empty(m, dtype=np.uint64)
            buf.fill(val)
            buf.tofile(f)
            remaining -= m


def write_assignments(
    output_path: str,
    weights_map: dict[str, float],
    total_examples: int,
    max_compartments: int,
    seed: int = 0,
    no_shuffle: bool = False,
) -> None:
    # Build weight entries
    weights: list[tuple[Category, float]] = []
    for key, w in sorted(weights_map.items()):
        if ">" in key:
            left, right = key.split(">", 1)
            src = int(left)
            dst = int(right)
            if src < 0 or dst < 0:
                raise ValueError("compartment ids must be non-negative")
            weights.append((Category("T", src, dst), float(w)))
        else:
            cid = int(key)
            if cid < 0:
                raise ValueError("compartment id must be non-negative")
            weights.append((Category("C", cid, 0), float(w)))

    if not weights:
        raise ValueError("weights_map is empty")

    # tie-breaker sort by category key for determinism
    weights.sort(key=lambda cw: cw[0].key())

    allocs = _largest_remainder_allocations(weights, int(total_examples))

    # Validate compartments within max
    highest_id = 0
    for cat, _ in allocs:
        highest_id = max(highest_id, cat.src)
        if cat.kind == "T":
            highest_id = max(highest_id, cat.dst)
    if max_compartments is None or max_compartments <= 0:
        raise ValueError("max_compartments must be provided and > 0")
    if highest_id >= max_compartments:
        raise ValueError(
            f"found compartment id {highest_id} but max_compartments={max_compartments}"
        )

    # Prepare encoded values and counts
    encoded_values, counts, cumulative_counts, total_records = (
        _prepare_allocation_arrays(allocs)
    )

    # Header
    flags = 1  # little-endian
    header = HEADER_STRUCT.pack(
        MAGIC,
        VERSION,
        RECORD_SIZE,
        flags,
        int(max_compartments),
        int(total_records),
        int(seed),
    )
    assert len(header) == HEADER_SIZE

    # Stream write directly to file, optionally shuffled by LCG permutation
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(header)
        if total_records > 0:
            if no_shuffle or total_records == 1:
                _write_records_in_order(f, encoded_values, counts)
            else:
                _write_records_streaming_lcg(
                    f, encoded_values, cumulative_counts, total_records, seed
                )
