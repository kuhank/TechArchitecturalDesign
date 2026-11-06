"""Master controller: compile per-dimension decisions into one output.

1. Sample K perturbed generations   (perturbation ensemble)
2. Decompose each into dimensions   (Decomposer)
3. Read the chaos map               (ChaosSensor)
4. Let each dimension controller act (ControllerBank)
5. Check cross-dimension consistency; any violating dimensions are
   treated as unstable and sent back through the ladder once
6. Compose the final output and report the global stability index

    S = 1 - sum_j w_j d_j / sum_j w_j
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from .controller import ControllerBank, ControlDecision
from .decompose import Decomposer
from .sensor import ChaosSensor, DimensionReading


@dataclass
class SCCResult:
    output: str
    readings: Dict[str, DimensionReading]
    decisions: Dict[str, ControlDecision]
    stability_index: float
    extra_cost: float
    samples: List[str]


class MasterController:
    def __init__(self, generate: Callable[[str, float], str], perturb: Callable[[str, int], List[str]],
                 decomposer: Decomposer, sensor: ChaosSensor, bank: ControllerBank,
                 compose: Callable[[Dict[str, object], List[str], List[Dict]], str],
                 consistency: Optional[Callable[[Dict[str, object]], List[str]]] = None,
                 k: int = 5):
        self.generate, self.perturb = generate, perturb
        self.decomposer, self.sensor, self.bank = decomposer, sensor, bank
        self.compose, self.consistency, self.k = compose, consistency, k

    def run(self, question: str) -> SCCResult:
        prompts = self.perturb(question, self.k)
        base_tau = min(c.tau for c in self.bank.controllers.values()
                       if c.dim.role == "precision")
        samples = [self.generate(p, base_tau) for p in prompts]
        decomposed = [self.decomposer(s) for s in samples]
        readings = self.sensor.read(decomposed)
        decisions = self.bank.step(readings, decomposed)

        chosen = {n: d.value for n, d in decisions.items()}
        if self.consistency:
            for bad in self.consistency(chosen):
                ctrl = self.bank.controllers[bad]
                if ctrl.grounder is not None:
                    chosen[bad] = ctrl.grounder(bad)
                    decisions[bad] = ControlDecision(bad, "ground", chosen[bad], 1.0, "consistency repair")

        dims = self.decomposer.dimensions
        wsum = sum(d.weight for d in dims if d.name in readings)
        s_idx = 1.0 - sum(d.weight * readings[d.name].divergence for d in dims if d.name in readings) / wsum
        cost = sum(d.cost for d in decisions.values())
        return SCCResult(self.compose(chosen, samples, decomposed), readings, decisions,
                         s_idx, cost, samples)
