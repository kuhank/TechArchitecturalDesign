"""Agent Lyapunov time and predictability-horizon checkpoint scheduling.

If an agent's deviation grows as e(t) = e0 * exp(lambda * t), the step at
which it reaches the tolerance Delta is

    T_h = (1/lambda) * ln(Delta / e0)            (predictability horizon)

Checkpointing (re-anchoring e back to e0) every alpha*T_h steps,
0 < alpha < 1, bounds the worst-case deviation by

    e_max = e0 * (Delta / e0) ** alpha  < Delta
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

from .sensor import ftle_from_series


def predictability_horizon(lam: float, e0: float, delta: float) -> float:
    if lam <= 0:
        return math.inf
    return math.log(delta / e0) / lam


def max_error_with_checkpoints(lam: float, e0: float, interval: float) -> float:
    return e0 * math.exp(max(lam, 0.0) * interval)


@dataclass
class HorizonManager:
    e0: float = 0.01
    delta: float = 0.3
    alpha: float = 0.7
    min_interval: int = 1
    history: List[float] = field(default_factory=list)
    lam: float = 0.5   # prior exponent before enough history

    def observe(self, deviation: float) -> None:
        self.history.append(max(deviation, 1e-6))
        if len(self.history) >= 3:
            self.lam = max(ftle_from_series(self.history[-8:]), 1e-3)

    def interval(self) -> int:
        th = predictability_horizon(self.lam, self.e0, self.delta)
        if math.isinf(th):
            return 10 ** 6
        return max(self.min_interval, int(self.alpha * th))

    def should_checkpoint(self, steps_since_last: int) -> bool:
        return steps_since_last >= self.interval()

    def reset(self) -> None:
        self.history.clear()
