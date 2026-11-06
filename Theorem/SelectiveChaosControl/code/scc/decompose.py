"""Semantic decomposition of an agent output into dimensions.

A decomposer maps a raw output (here: a SQL string) to a dict
{dimension_name: canonical_value}. Canonical values are hashable
(frozensets / tuples / strings) so they can be compared across samples.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Hashable

import sqlglot
from sqlglot import exp

Canonical = Hashable


@dataclass
class Dimension:
    name: str
    role: str = "precision"          # "precision" or "creative"
    weight: float = 1.0               # importance in the global stability index
    target_band: tuple = (0.0, 0.0)   # acceptable divergence band [lo, hi]

    def __post_init__(self):
        if self.role == "creative" and self.target_band == (0.0, 0.0):
            self.target_band = (0.2, 0.6)


SQL_DIMENSIONS = [
    Dimension("tables", weight=1.5),
    Dimension("projection", weight=1.0),
    Dimension("joins", weight=1.2),
    Dimension("predicates", weight=1.5),
    Dimension("aggregation", weight=1.0),
    Dimension("ordering_limit", weight=0.6),
]

EXPLANATION_DIMENSION = Dimension("explanation", role="creative", weight=0.2)


def _norm(e: exp.Expression) -> str:
    return e.sql(dialect="postgres", normalize=True).lower()


def decompose_sql(sql: str) -> Dict[str, Canonical]:
    """Split a SQL query into canonical per-dimension values."""
    try:
        tree = sqlglot.parse_one(sql, read="postgres")
    except Exception:
        return {d.name: "__PARSE_ERROR__" for d in SQL_DIMENSIONS}

    tables = frozenset(t.name.lower() for t in tree.find_all(exp.Table))
    projection = frozenset(_norm(p) for p in tree.expressions) if hasattr(tree, "expressions") else frozenset()
    joins = frozenset(_norm(j) for j in tree.find_all(exp.Join))

    preds = []
    for clause in (tree.args.get("where"), tree.args.get("having")):
        if clause is not None:
            cond = clause.this
            # split top-level ANDs so predicate order does not matter
            parts = list(cond.flatten()) if isinstance(cond, exp.And) else [cond]
            preds += [_norm(p) for p in parts]
    predicates = frozenset(preds)

    group = tree.args.get("group")
    aggs = frozenset(_norm(a) for a in tree.find_all(exp.AggFunc))
    aggregation = (frozenset(_norm(g) for g in group.expressions) if group else frozenset(), aggs)

    order = tree.args.get("order")
    limit = tree.args.get("limit")
    ordering_limit = (_norm(order) if order else "", _norm(limit) if limit else "")

    return {
        "tables": tables,
        "projection": projection,
        "joins": joins,
        "predicates": predicates,
        "aggregation": aggregation,
        "ordering_limit": ordering_limit,
    }


@dataclass
class Decomposer:
    dimensions: list = field(default_factory=lambda: list(SQL_DIMENSIONS))
    fn: Callable[[str], Dict[str, Canonical]] = decompose_sql

    def __call__(self, output: str) -> Dict[str, Canonical]:
        return self.fn(output)
