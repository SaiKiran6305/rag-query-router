"""
Golden query set, derived programmatically from ground truth.

Expected answers are computed from the ground-truth records rather than typed
by hand, so the eval cannot drift from the corpus. Regenerate the corpus with a
different seed and every expected answer updates with it.

SEMANTIC cases are included deliberately. Both systems route them down the same
retrieval path and should therefore score identically. A benchmark where the
proposed system wins every category is a benchmark someone rigged; the useful
claim is narrower and more defensible -- retrieval is fine at the one thing
retrieval is for, and structurally incapable at five things it is routinely
asked to do anyway.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from app.models import QueryType


@dataclass
class GoldenQuery:
    query: str
    category: QueryType
    expected: Any                      # int for counts, set[str] for id sets
    answer_kind: str                   # "scalar" | "set" | "retrieval"
    note: str = ""
    tags: list[str] = field(default_factory=list)


def _q1_window(year: int) -> tuple[str, str]:
    return f"{year}-01-01", f"{year}-03-31"


def build_golden_set(ground_truth_path: str | Path) -> list[GoldenQuery]:
    gt: list[dict] = json.loads(Path(ground_truth_path).read_text(encoding="utf-8"))
    by_id = {t["contract_id"]: t for t in gt}

    def ids(pred) -> set[str]:
        return {t["contract_id"] for t in gt if pred(t)}

    G: list[GoldenQuery] = []

    # ---- AGGREGATE: counts over the full corpus ---------------------------
    G.append(GoldenQuery(
        "How many contracts have auto renewal?", QueryType.AGGREGATE,
        len(ids(lambda t: t["auto_renew"])), "scalar",
        "A count requires the corpus. Top-k returns a sample of it.",
        ["counting"]))
    G.append(GoldenQuery(
        "How many contracts require insurance?", QueryType.AGGREGATE,
        len(ids(lambda t: t["insurance_required"])), "scalar",
        tags=["counting"]))
    G.append(GoldenQuery(
        "How many contracts include indemnification?", QueryType.AGGREGATE,
        len(ids(lambda t: t["indemnification"])), "scalar",
        tags=["counting"]))
    G.append(GoldenQuery(
        "How many contracts permit assignment?", QueryType.AGGREGATE,
        len(ids(lambda t: t["assignment_allowed"])), "scalar",
        tags=["counting"]))

    # ---- ABSENCE: the category where retrieval ranks truth last -----------
    G.append(GoldenQuery(
        "Which contracts have no liability cap?", QueryType.ABSENCE,
        ids(lambda t: t["liability_cap_usd"] is None), "set",
        "Two textual shapes: clause omitted, or clause present stating no limit. "
        "Neither is retrievable by similarity to 'liability cap'.",
        ["negation", "absence"]))
    G.append(GoldenQuery(
        "How many contracts have no liability cap?", QueryType.ABSENCE,
        len(ids(lambda t: t["liability_cap_usd"] is None)), "scalar",
        "Negation and counting compounded.", ["negation", "counting"]))
    G.append(GoldenQuery(
        "Which contracts do not require insurance?", QueryType.ABSENCE,
        ids(lambda t: not t["insurance_required"]), "set",
        tags=["negation", "absence"]))
    G.append(GoldenQuery(
        "Which contracts lack an indemnification clause?", QueryType.ABSENCE,
        ids(lambda t: not t["indemnification"]), "set",
        tags=["negation", "absence"]))
    G.append(GoldenQuery(
        "Which contracts have no confidentiality period?", QueryType.ABSENCE,
        ids(lambda t: t["confidentiality_years"] is None), "set",
        tags=["negation", "absence"]))

    # ---- NUMERIC: magnitude comparison ------------------------------------
    G.append(GoldenQuery(
        "Which contracts have penalties above $100,000?", QueryType.NUMERIC,
        ids(lambda t: bool(t["late_penalty_usd"]) and t["late_penalty_usd"] > 100_000), "set",
        "Embeddings encode that a number is present, not its magnitude.",
        ["magnitude"]))
    G.append(GoldenQuery(
        "Which contracts have a liability cap under $1,000,000?", QueryType.NUMERIC,
        ids(lambda t: t["liability_cap_usd"] is not None and t["liability_cap_usd"] < 1_000_000), "set",
        tags=["magnitude"]))
    G.append(GoldenQuery(
        "Which contracts require more than 30 days termination notice?", QueryType.NUMERIC,
        ids(lambda t: t["termination_notice_days"] > 30), "set",
        tags=["magnitude"]))
    G.append(GoldenQuery(
        "Which contracts have liability capped at or above $2,000,000?", QueryType.NUMERIC,
        ids(lambda t: t["liability_cap_usd"] is not None and t["liability_cap_usd"] >= 2_000_000), "set",
        tags=["magnitude"]))

    # ---- TEMPORAL ----------------------------------------------------------
    for yr in (2026, 2027):
        lo, hi = _q1_window(yr)
        G.append(GoldenQuery(
            f"List contracts expiring in Q1 {yr}", QueryType.TEMPORAL,
            ids(lambda t, lo=lo, hi=hi: lo <= t["expiry_date"] <= hi), "set",
            "Vector space has no ordering over dates.", ["temporal"]))
    G.append(GoldenQuery(
        "Which contracts expire in 2027?", QueryType.TEMPORAL,
        ids(lambda t: t["expiry_date"].startswith("2027")), "set",
        tags=["temporal"]))

    # ---- MULTIHOP: dependent traversal ------------------------------------
    for law in ("California", "Illinois", "New York"):
        parents = {t["contract_id"] for t in gt if t["governing_law"] == law}
        G.append(GoldenQuery(
            f"Which vendors are on statements of work issued under {law} master agreements?",
            QueryType.MULTIHOP,
            ids(lambda t, p=parents: t["amends_contract_id"] in p), "set",
            "Hop 2 consumes hop 1's output. One retrieval pass cannot.",
            ["multihop"]))

    # ---- ENUMERATE: completeness guaranteed --------------------------------
    for law in ("Delaware", "Texas"):
        G.append(GoldenQuery(
            f"List every contract governed by {law} law", QueryType.ENUMERATE,
            ids(lambda t, l=law: t["governing_law"] == l), "set",
            "Top-k cannot signal that its list is partial.", ["completeness"]))

    # ---- SEMANTIC: the control group ---------------------------------------
    # Scored on whether the retrieved contracts actually contain the clause asked
    # about. Both systems share this path, so parity here is the expected result
    # and is what makes the other categories credible.
    G.append(GoldenQuery(
        "What does the indemnification clause say?", QueryType.SEMANTIC,
        ids(lambda t: t["indemnification"]), "retrieval",
        "Control: retrieval is the right tool for this. Scored on precision@k.", ["control"]))
    G.append(GoldenQuery(
        "Explain the confidentiality obligations", QueryType.SEMANTIC,
        ids(lambda t: t["confidentiality_years"] is not None), "retrieval",
        tags=["control"]))
    G.append(GoldenQuery(
        "What are the termination provisions?", QueryType.SEMANTIC,
        ids(lambda t: t["termination_notice_days"] is not None), "retrieval",
        tags=["control"]))

    return G


if __name__ == "__main__":
    import sys
    gs = build_golden_set(sys.argv[1] if len(sys.argv) > 1 else "data/corpus/ground_truth.json")
    from collections import Counter
    print(f"{len(gs)} golden queries")
    for cat, n in Counter(g.category.value for g in gs).most_common():
        print(f"  {cat:12} {n}")
