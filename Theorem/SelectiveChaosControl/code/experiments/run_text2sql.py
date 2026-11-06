"""Empirical evaluation on a text-to-SQL benchmark (Spider / BIRD dev set).

Usage (on your machine):
    export SCC_BASE_URL=... SCC_API_KEY=... SCC_MODEL=...
    python run_text2sql.py --data dev.json --db-root databases/ --limit 200 --k 5

Expected dev.json items: {"question": ..., "db_id": ..., "query"|"SQL": ...}
Databases are SQLite files at <db-root>/<db_id>/<db_id>.sqlite (Spider/BIRD layout).

Strategies compared:
    single     one greedy generation
    selfcons   K samples, majority vote on execution result (whole output)
    scc        Selective Chaos Control (proposed; reference implementation)
Planned additions: verify-all baseline, per-dimension temperatures, measured eps.
Metrics: execution accuracy, LLM calls per question, intervention rate per dimension.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scc import ChaosSensor, ControllerBank, Decomposer, MasterController, SQL_DIMENSIONS  # noqa: E402
from scc import llm  # noqa: E402


def schema_of(db: Path) -> str:
    con = sqlite3.connect(db)
    rows = con.execute("select sql from sqlite_master where type='table' and sql is not null").fetchall()
    con.close()
    return "\n".join(r[0] for r in rows)


def execute(db: Path, sql: str):
    try:
        con = sqlite3.connect(db)
        con.execute("PRAGMA query_only = 1")
        res = con.execute(sql).fetchmany(1000)
        con.close()
        return frozenset(map(tuple, res))
    except Exception:
        return None


def run(args):
    data = json.load(open(args.data))[: args.limit]
    stats = defaultdict(lambda: {"correct": 0, "calls": 0})
    interventions = Counter()

    for i, ex in enumerate(data):
        db = Path(args.db_root) / ex["db_id"] / f"{ex['db_id']}.sqlite"
        schema = schema_of(db)
        gold = execute(db, ex.get("query") or ex.get("SQL"))
        gen = lambda p, t: llm.generate(p, t, schema)  # noqa: E731

        # single
        s = gen(ex["question"], 0.0)
        stats["single"]["correct"] += execute(db, s) == gold
        stats["single"]["calls"] += 1

        # self-consistency
        outs = [gen(ex["question"], 0.7) for _ in range(args.k)]
        votes = Counter(r for r in (execute(db, o) for o in outs) if r is not None)
        best = votes.most_common(1)[0][0] if votes else None
        stats["selfcons"]["correct"] += best == gold
        stats["selfcons"]["calls"] += args.k

        # scc
        def grounder(dim, q=ex["question"]):
            # re-ask with a dimension-focused hint, at temperature 0
            hint = f"{q} (Be precise about the {dim.replace('_', ' ')}; use exact schema names.)"
            return Decomposer()(gen(hint, 0.0))[dim]

        bank = ControllerBank.from_dimensions(SQL_DIMENSIONS)
        for c in bank.controllers.values():
            c.grounder = grounder
        compose = lambda chosen, samples, dec: min(  # noqa: E731
            samples, key=lambda s: sum(dec[samples.index(s)][n] != v for n, v in chosen.items()))
        master = MasterController(gen, llm.perturb, Decomposer(), ChaosSensor(eps=0.15), bank,
                                  compose, k=args.k)
        res = master.run(ex["question"])
        stats["scc"]["correct"] += execute(db, res.output) == gold
        stats["scc"]["calls"] += args.k + sum(d.action == "ground" for d in res.decisions.values())
        for d in res.decisions.values():
            interventions[(d.dimension, d.action)] += 1

        print(f"[{i + 1}/{len(data)}] " + "  ".join(
            f"{k}={v['correct']}/{i + 1}" for k, v in stats.items()), flush=True)

    n = len(data)
    report = {k: {"exec_acc": v["correct"] / n, "calls_per_q": v["calls"] / n} for k, v in stats.items()}
    report["interventions"] = {f"{d}:{a}": c for (d, a), c in interventions.items()}
    json.dump(report, open(args.out, "w"), indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--db-root", required=True)
    ap.add_argument("--limit", type=int, default=200)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", default="results_text2sql.json")
    run(ap.parse_args())
