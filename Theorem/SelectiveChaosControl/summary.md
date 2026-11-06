# Selective Chaos Control (SCC): summary

**What:** A research paper and prototype that applies chaos theory (Lyapunov exponents and OGY chaos control) to LLM agents.
- Measures how "chaotic" each output dimension is.
- Fixes only the unstable dimensions and keeps creative dimensions free.
- Re-checks multi-step agents at their "agent Lyapunov time".

Target journal: *Chaos, Solitons & Fractals* (Elsevier). Sole author: **Kuhan K**, Independent Researcher, Chennai, India.

## Status (06 Nov 2026)
- v1 manuscript complete (PDF and LaTeX): theory with 5 propositions, simulation study E1–E6, applications, and an evaluation protocol.
- Prototype framework code done. The real-model text-to-SQL experiment script is ready but **not yet run**.
- Not yet submitted.

## Key results (synthetic; corrected after an independent review)
- Per-dimension divergence localises errors: AUROC 0.968.
- SCC reaches 92.1% accuracy at 5.6 calls, versus whole-output self-consistency at 54.3% (5 calls) and verify-all at 85.7% (7 calls). The advantage depends on grounding quality.
- Horizon checkpointing with known growth rates: 97.5% vs 80.8% for a fixed interval at equal cost. Estimated online, the gain is small and only at moderate budgets (reported as an open problem).
- The temperature controller bifurcates within 0.2% of the predicted critical gain.

## Review fixes applied (06 Nov)
- Ties in votes are now broken at random; this was inflating the factorised baseline.
- The multi-step test now hides the true deviation between checkpoints.
- Highlights, abstract and conclusion now say "synthetic"; the code is described as a reference implementation.

## Key decisions
- Framework + prototype paper; sole author; affiliation Independent Researcher, Chennai; email kuhankailash92@gmail.com.
- Crime application limited to aggregate patterns, with an ethics note.

## Open / next steps
1. Provisional patent (individual filing; address Karaikal, Puducherry) **before** any public post. Spec and filing sheet are in `patent/`.
2. Run `code/experiments/run_text2sql.py` on Spider/BIRD and add the real results.
3. Fill in the generative AI declaration (red text at the end of the manuscript).
4. Submit via Editorial Manager (see `v1_submission_guide.md`). LinkedIn post only after the patent is filed.
5. Possible sponsor pitch once the real results are in.

## Files
- `patent/v1_provisional_specification.pdf`, `patent/v1_form1_filing_sheet.md`: patent filing
- `paper/v1_SelectiveChaosControl.pdf`: manuscript
- `paper/v1_SelectiveChaosControl.tex`, `paper/references.bib`, `paper/figures/`: LaTeX source
- `paper/v1_highlights.txt`, `paper/v1_cover_letter.txt`: submission extras
- `v1_submission_guide.md`: upload steps and checklist
- `code/README.md`: how to run the simulations and the real experiment
- `code/simulation/results.json`: all simulation numbers
