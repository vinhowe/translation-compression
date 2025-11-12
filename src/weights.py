from __future__ import annotations

from typing import Literal


def compute_weights_map(
    n: int, t: float, scaling: Literal["equal", "unequal"]
) -> dict[str, float]:
    """Compute sampling weights for compartments and translations.

    Parameters
    ----------
    n: int
        Number of compartments. Must be >= 1.
    t: float
        Translation ratio. Must be >= 0. When t = 0, no translation weights are created.
    scaling: Literal["equal", "unequal"]
        Base compartment weighting scheme.

    Returns
    -------
    dict[str, float]
        A mapping of category keys to probabilities that sum to 1. Keys are:
        - "i" for compartment-only examples from compartment i
        - "i>j" and "j>i" for translation examples (symmetric, each direction gets half)
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if t < 0:
        raise ValueError("translation ratio t must be >= 0")
    if t > 0 and n < 2:
        raise ValueError("translation_ratio > 0 requires n >= 2")

    # Base compartment weights w_i
    if scaling == "equal":
        base_weights = [1.0 / float(n) for _ in range(n)]
    elif scaling == "unequal":
        # ids in code are 0..n-1, corresponding to (i+1) / Z
        z = float(n * (n + 1) // 2)
        base_weights = [float(i + 1) / z for i in range(n)]

    # Mass split between compartments and translations
    mc = float(n) / float(n + t)
    mt = float(t) / float(n + t)

    weights: dict[str, float] = {}

    # Compartment-only probabilities: p_i = M_c * w_i
    for i in range(n):
        weights[str(i)] = mc * base_weights[i]

    # Translation probabilities: allocate pair mass normalized over i<j
    if mt > 0.0 and n >= 2:
        # sum_{i<j} w_i w_j; safe formula avoids catastrophic cancellation
        pair_sum = 0.0
        for i in range(n):
            wi = base_weights[i]
            for j in range(i + 1, n):
                pair_sum += wi * base_weights[j]

        if pair_sum <= 0.0:
            # This should not happen for valid inputs unless n < 2 or base weights are
            # degenerate
            raise ValueError("pair_sum is zero; cannot allocate translation mass")

        for i in range(n):
            wi = base_weights[i]
            for j in range(i + 1, n):
                pij = mt * (wi * base_weights[j] / pair_sum)
                half = 0.5 * pij
                weights[f"{i}>{j}"] = half
                weights[f"{j}>{i}"] = half

    # Numerical hygiene: ensure negatives don't sneak in from FP; renormalize to 1
    # (write_assignments will re-normalize anyway, but keep map well-formed here)
    total = sum(max(0.0, v) for v in weights.values())
    if total <= 0.0:
        raise ValueError("total weight mass is non-positive")
    for k in list(weights.keys()):
        weights[k] = max(0.0, weights[k]) / total

    return weights
