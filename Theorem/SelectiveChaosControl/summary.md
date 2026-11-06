# Selective Chaos Control (SCC): summary

**What:** A research paper and prototype that applies chaos theory (Lyapunov exponents and OGY chaos control) to LLM agents.
- Measures how "chaotic" each output dimension is.
- Fixes only the unstable dimensions and keeps creative dimensions free.
- Re-checks multi-step agents at their "agent Lyapunov time".

Target journal: *Chaos, Solitons & Fractals* (Elsevier). Sole author: **Kuhan K**, IIIT Kottayam, Kerala, India.

## Status (06 Nov 2026)
- v1 manuscript complete (PDF and LaTeX): theory with 5 propositions, simulation study E1–E6, applications, and an evaluation protocol.
- Prototype framework code done. The real-model text-to-SQL experiment script is ready but **not yet run**.
- Not yet submitted.

## Key results (synthetic)
- Per-dimension exponent localises errors: AUROC 0.947.
- SCC reaches 92.4% accuracy at 5.6 calls, versus whole-output self-consistency at 56.5% (5 calls) and verify-all at 85.7% (7 calls).
- Lyapunov-time checkpointing reaches 99.8% success, versus 79.8% for a fixed interval at equal cost.
- The temperature controller bifurcates at the predicted critical gain (0.9998·η_c).

## Key decisions
- Framework + prototype paper; sole author; affiliation IIIT Kottayam; email kuhankailash92@gmail.com.
- Crime application limited to aggregate patterns, with an ethics note.

## Open / next steps
1. Provisional patent (check the IIIT Kottayam IP policy) **before** any public post.
2. Run `code/experiments/run_text2sql.py` on Spider/BIRD and add the real results.
3. Fill in the generative AI declaration (red text at the end of the manuscript).
4. Submit via Editorial Manager (see `v1_submission_guide.md`). LinkedIn post only after the patent is filed.
5. Possible sponsor pitch once the real results are in.

## Files
- `paper/v1_SelectiveChaosControl.pdf`: manuscript
- `paper/v1_SelectiveChaosControl.tex`, `paper/references.bib`, `paper/figures/`: LaTeX source
- `paper/v1_highlights.txt`, `paper/v1_cover_letter.txt`: submission extras
- `v1_submission_guide.md`: upload steps and checklist
- `code/README.md`: how to run the simulations and the real experiment
- `code/simulation/results.json`: all simulation numbers
