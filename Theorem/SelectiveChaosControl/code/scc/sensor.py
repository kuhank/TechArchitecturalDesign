"""Chaos sensor: dimension-wise divergence and Lyapunov-style exponents.

Given K outputs produced from K slightly perturbed versions of the same
input, the sensor computes for every dimension j

    d_j      mean pairwise distance between the K canonical values (0..1)
    A_j      amplification ratio  d_j / eps   (eps = input perturbation size)
    lambda_j finite-step Lyapunov-style exponent  ln(A_j)
    p_hat_j  error-rate estimate implied by d_j (binary-disagreement model)

lambda_j > 0  -> the dimension amplifies input noise (chaotic)
lambda_j < 0  -> the dimension contracts input noise (stable)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Sequence


def value_distance(a, b) -> float:
    """Distance in [0,1] between two canonical values."""
    if a == b:
        return 0.0
    if isinstance(a, frozenset) and isinstance(b, frozenset):
        union = a | b
        return 1.0 - (len(a & b) / len(union)) if union else 0.0
    if isinstance(a, tuple) and isinstance(b, tuple) and len(a) == len(b):
        return sum(value_distance(x, y) for x, y in zip(a, b)) / len(a)
    return 1.0


def pairwise_divergence(values: Sequence) -> float:
    pairs = list(combinations(values, 2))
    if not pairs:
        return 0.0
    return sum(value_distance(a, b) for a, b in pairs) / len(pairs)


def implied_error_rate(d: float) -> float:
    """Upper bound on the per-sample error rate p implied by exact-match
    divergence d (single wrong alternative: d = 2p(1-p))."""
    d = min(max(d, 0.0), 0.5)
    return (1.0 - math.sqrt(1.0 - 2.0 * d)) / 2.0


def implied_error_rate_lower(d: float) -> float:
    """Lower bound on p (diffuse wrong answers: d = 2p - p^2)."""
    d = min(max(d, 0.0), 1.0)
    return 1.0 - math.sqrt(1.0 - d)


@dataclass
class DimensionReading:
    name: str
    divergence: float
    amplification: float
    exponent: float
    error_estimate: float
    state: str  # "stable" | "borderline" | "chaotic"


class ChaosSensor:
    def __init__(self, eps: float = 0.05, stable_below: float = 0.0, chaotic_above: float = 1.0,
                 floor: float = 1e-3):
        # thresholds are on the exponent lambda = ln(d/eps)
        self.eps = eps
        self.stable_below = stable_below
        self.chaotic_above = chaotic_above
        self.floor = floor

    def read(self, decomposed: List[Dict]) -> Dict[str, DimensionReading]:
        names = decomposed[0].keys()
        out = {}
        for n in names:
            vals = [s[n] for s in decomposed]
            d = pairwise_divergence(vals)
            amp = max(d, self.floor) / self.eps
            lam = math.log(amp)
            if lam < self.stable_below:
                state = "stable"
            elif lam > self.chaotic_above:
                state = "chaotic"
            else:
                state = "borderline"
            out[n] = DimensionReading(n, d, amp, lam, implied_error_rate(d), state)
        return out


def ftle_from_series(errors: Sequence[float]) -> float:
    """Finite-time Lyapunov exponent from an error series e(0..T):
    least-squares slope of ln e(t) versus t (Rosenstein-style)."""
    pts = [(t, math.log(e)) for t, e in enumerate(errors) if e > 0]
    if len(pts) < 2:
        return 0.0
    n = len(pts)
    mt = sum(t for t, _ in pts) / n
    my = sum(y for _, y in pts) / n
    num = sum((t - mt) * (y - my) for t, y in pts)
    den = sum((t - mt) ** 2 for t, _ in pts)
    return num / den if den else 0.0
