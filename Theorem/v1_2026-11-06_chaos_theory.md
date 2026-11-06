# Day 1 (06 Nov 2026): Chaos theory, Poincaré–Bendixson, attractors

Source: a short video, "Four Equations That Never Repeat" (Rössler, Thomas, Rucklidge, Halvorsen).

## 1. What the equations are
- `ẋ` (x-dot) = how fast x is changing right now (dx/dt). These are **differential equations**: rules for how something changes.
- Each system has 3 rules (x, y, z). They work like turn-by-turn GPS directions: "you're here → move this way". Following them step by step traces a path.
- **Deterministic:** same start → exactly the same path. No randomness.

| System | Who / when | Origin | Equations |
|---|---|---|---|
| Rössler | Otto Rössler, 1976 | Chemical reactions; inspired by a taffy-pulling machine | ẋ=−y−z, ẏ=x+0.2y, ż=0.2+z(x−5.7) |
| Thomas | René Thomas, 1999 | Biology: genes switching each other on/off | ẋ=sin y−bx, ẏ=sin z−by, ż=sin x−bz, b=0.208 |
| Rucklidge | Alastair Rucklidge, 1992 | Fluid heated from below (rolling convection) | ẋ=−2x+6.7y−yz, ẏ=x, ż=−z+y² |
| Halvorsen | Arne Halvorsen | Pure maths; 3 identical rules, rotated | ẋ=−1.4x−4y−4z−y² (and cyclic) |
| Lorenz (cousin) | Edward Lorenz, 1963 | Simplified weather (convection) | the butterfly-shaped attractor |

## 2. The theorem: Poincaré–Bendixson
- **Statement (simple):** on a flat surface (2D), a smooth path that stays inside a box can only (a) stop at a point, or (b) go round the same loop forever.
- So **chaos is impossible in 2D**. Analogy: cars on a single-lane road can't overtake.
- **3D makes chaos possible** (not guaranteed): paths can go over and under each other, like flyovers.
- Note: this applies to **continuous flows**. Step-by-step (discrete) maps can be chaotic even in 1D, e.g. the logistic map.

## 3. Key ideas
- **Strange attractor:** the shape a path settles into. It is bounded (stays in a region), never repeats, and never crosses itself.
- **Stretch and fold (taffy / parotta dough):** stretching pulls nearby points apart; folding keeps everything in one region. Repeating both forever gives chaos.
- **Butterfly effect:** a tiny start difference grows exponentially. Rössler test with two starts 0.00000001 apart:

| time | gap |
|---|---|
| 0 | 0.00000001 |
| 100 | 0.00003 |
| 200 | 0.014 |
| 250 | 0.87 (completely different) |

- **Lyapunov exponent (λ):** the speed at which errors grow, error ≈ e₀·e^(λt).
- **Predictability horizon:** `T = (1/λ) · ln(tolerance / starting error)`.
  - Example: an error that doubles every 2 days and becomes useless at 1000× → about 20 days.
  - Measuring 10× more accurately adds only a few days, because errors grow exponentially while accuracy improves step by step.
- **Weather:** about a 2-week limit (Lorenz 1963/1969; Zhang et al. 2019). Forecasters run around 50 slightly different forecasts (an "ensemble") to get "% chance of rain".
- **More dimensions:** more *room* for chaos (4D+ allows "hyperchaos"), but not automatically more chaos. You also need stretching, folding and low friction.
  - Examples with no chaos: marbles settling in a bowl; pendulum clocks synchronising; Thomas with b=0.5.
- **Chaos control (OGY, 1990):** small, well-timed nudges keep a chaotic system on a stable loop.
  - Examples: pacemakers, lasers, Japan's Hiten spacecraft (1991).
  - Not possible for weather: the system is too big and can't be nudged precisely.
- **Sometimes chaos is wanted:** mixing, and encryption keys.

## 4. Link to AI / LLMs
- An LLM writes token by token and feeds each token back in, so it's an iterated map in thousands of dimensions.
- One word changed in a prompt, or one wrong early token, can lead to a very different answer; hallucinations "snowball".
- This is a close analogy, not proven chaos: training pushes models toward stability.
- **The fix is the same as chaos control:** check often, ground with real data, break long tasks into short checked steps, and use low temperature for factual tasks.

## 5. Research idea that came out of this → paper
"Selective Chaos Control" (SCC). See `SelectiveChaosControl/summary.md`.
- A chaos map per output dimension: an **agent per dimension**, with a master controller compiling the final output.
- Suppress chaos in precision dimensions; allow it in creative ones.
- **Agent Lyapunov time:** re-check just before the agent drifts.
- **Applications:** robots (creative exploration plus safety controllers), brain/EEG, aggregate crime patterns (no individual profiling), activity tracking.
