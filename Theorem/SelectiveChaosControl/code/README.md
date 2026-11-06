# Selective Chaos Control (SCC): code

## Layout

| Path | What it is |
|---|---|
| `scc/decompose.py` | Splits an output (SQL) into dimensions: tables, projection, joins, predicates, aggregation, ordering/limit |
| `scc/sensor.py` | Chaos sensor: divergence per dimension, finite-step exponent, error-rate bracket, FTLE from a series |
| `scc/controller.py` | Dimension controllers: intervention ladder (none, consensus, ground, verify, escalate) and per-dimension temperature law |
| `scc/horizon.py` | Agent Lyapunov time, predictability horizon, checkpoint scheduling |
| `scc/master.py` | Master controller: ensemble, chaos map, decisions, consistency repair, stability index |
| `scc/llm.py` | Adapter for any OpenAI-compatible chat endpoint, plus prompt perturbations |
| `simulation/simulate.py` | All numerical experiments E1–E6 in the paper; writes figures to `../paper/figures` and `results.json` |
| `experiments/run_text2sql.py` | Real-model evaluation on Spider / BIRD (execution accuracy) |

## Setup

```bash
pip install numpy scipy matplotlib sqlglot requests
```

## 1. Reproduce the paper's simulations (no model needed, < 1 min)

Run on: your laptop (Python 3.10+)

```bash
cd simulation
python3 simulate.py
```

What it does: runs experiments E1–E6 with a fixed seed (2026), regenerates every figure in `paper/figures/`, and writes all numbers to `simulation/results.json`.

## 2. Real experiment on text-to-SQL (needed before submission)

Run on: your laptop, against a model endpoint (hosted API or local Ollama / vLLM)

```bash
export SCC_BASE_URL=http://localhost:11434/v1   # or a hosted OpenAI-compatible URL
export SCC_API_KEY=none                         # your key for hosted endpoints
export SCC_MODEL=qwen2.5-coder:7b               # any chat model

cd experiments
python3 run_text2sql.py --data <spider_or_bird>/dev.json --db-root <spider_or_bird>/database --limit 200 --k 5
```

What it does: for each question it runs single, self-consistency and SCC; executes each SQL on the SQLite database; and compares the result with the gold query.
Output: `results_text2sql.json`, containing execution accuracy, calls per question, and which dimension controllers acted.

Datasets (download yourself):
- Spider: https://yale-lily.github.io/spider (`dev.json` plus `database/<db_id>/<db_id>.sqlite`)
- BIRD: https://bird-bench.github.io (dev set, same layout)
