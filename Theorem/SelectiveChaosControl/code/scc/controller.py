"""Dimension controllers: threshold-triggered minimal intervention.

Inspired by OGY chaos control: do nothing while the trajectory stays in
the target region, apply the *smallest* perturbation that brings it back
when it leaves.

Intervention ladder (ordered by cost):
    0  none          dimension inside its target band
    1  consensus     take the medoid / majority value of the K samples (no extra call)
    2  ground        regenerate only this dimension with grounding facts
    3  verify        run an external check (e.g. execute a probe query)
    4  escalate      flag for human review

Selective chaos: each dimension keeps its own sampling temperature,
updated by a log-proportional feedback law

    tau_j <- clip( tau_j * exp(-eta * (d_j - d_j*)) )

so precision dimensions are driven toward tau -> 0 and creative
dimensions are held inside their diversity band.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .decompose import Dimension
from .sensor import DimensionReading, value_distance

ACTIONS = ["none", "consensus", "ground", "verify", "escalate"]
ACTION_COST = {"none": 0.0, "consensus": 0.0, "ground": 1.0, "verify": 1.5, "escalate": 10.0}


@dataclass
class ControlDecision:
    dimension: str
    action: str
    value: object
    cost: float
    note: str = ""


def medoid(values: List):
    counts = Counter(values)
    top, freq = counts.most_common(1)[0]
    if freq > 1:
        return top, freq / len(values)
    best = min(values, key=lambda v: sum(value_distance(v, w) for w in values))
    return best, 1.0 / len(values)


@dataclass
class DimensionController:
    dim: Dimension
    tau: float = 0.7
    eta: float = 2.0
    tau_min: float = 0.0
    tau_max: float = 1.2
    consensus_floor: float = 0.6     # min agreement share to trust consensus
    grounder: Optional[Callable[[str], object]] = None   # regenerate this dim with grounding
    verifier: Optional[Callable[[object], bool]] = None  # external check

    def target(self) -> float:
        lo, hi = self.dim.target_band
        return (lo + hi) / 2

    def update_temperature(self, divergence: float) -> float:
        self.tau = self.tau * math.exp(-self.eta * (divergence - self.target()))
        self.tau = min(max(self.tau, self.tau_min), self.tau_max)
        return self.tau

    def decide(self, reading: DimensionReading, samples: List) -> ControlDecision:
        lo, hi = self.dim.target_band
        d = reading.divergence

        if self.dim.role == "creative":
            # creative dims: keep diversity, just pick a representative
            self.update_temperature(d)
            return ControlDecision(self.dim.name, "none", samples[0], 0.0, f"creative d={d:.2f}")

        if lo <= d <= hi:
            return ControlDecision(self.dim.name, "none", samples[0], 0.0, "inside band")

        value, share = medoid(samples)
        self.update_temperature(d)
        if reading.state != "chaotic" and share >= self.consensus_floor:
            return ControlDecision(self.dim.name, "consensus", value, ACTION_COST["consensus"],
                                   f"agreement={share:.2f}")
        if self.grounder is not None:
            value = self.grounder(self.dim.name)
            if self.verifier is None or self.verifier(value):
                return ControlDecision(self.dim.name, "ground", value, ACTION_COST["ground"])
        if self.verifier is not None and self.verifier(value):
            return ControlDecision(self.dim.name, "verify", value, ACTION_COST["verify"])
        return ControlDecision(self.dim.name, "escalate", value, ACTION_COST["escalate"],
                               "unstable after ladder")


@dataclass
class ControllerBank:
    controllers: Dict[str, DimensionController] = field(default_factory=dict)

    @classmethod
    def from_dimensions(cls, dims: List[Dimension], **kw):
        return cls({d.name: DimensionController(d, **kw) for d in dims})

    def step(self, readings: Dict[str, DimensionReading], decomposed: List[Dict]) -> Dict[str, ControlDecision]:
        out = {}
        for name, ctrl in self.controllers.items():
            samples = [s[name] for s in decomposed]
            out[name] = ctrl.decide(readings[name], samples)
        return out
