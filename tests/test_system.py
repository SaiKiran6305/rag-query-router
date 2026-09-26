"""
Tests. Two kinds, deliberately separated.

Correctness tests assert the engines produce the right answer against ground
truth. Characterisation tests assert the *baseline fails* on the structural
categories -- they encode the project's central claim, so if a future change
makes naive RAG suddenly competent at counting, the suite should say so loudly
rather than let the README keep asserting something untrue.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.eval.golden import build_golden_set
from app.models import QueryType
from app.pipeline import ContractIntelligence

CORPUS = Path(__file__).resolve().parent.parent / "data" / "corpus"


@pytest.fixture(scope="module")
def system() -> ContractIntelligence:
    return ContractIntelligence(CORPUS / "contracts")


@pytest.fixture(scope="module")
def truth() -> list[dict]:
    return json.loads((CORPUS / "ground_truth.json").read_text(encoding="utf-8"))


# --- extraction -----------------------------------------------------------

def test_extraction_matches_ground_truth(system, truth):
    by_id = {t["contract_id"]: t for t in truth}
    fields = ["vendor", "governing_law", "effective_date", "expiry_date", "term_months",
              "auto_renew", "liability_cap_usd", "late_penalty_usd", "insurance_required",
              "indemnification", "assignment_allowed", "termination_notice_days"]
    rows = system.store.scan_all()
    assert len(rows) == len(truth)
    for r in rows:
        want = by_id[r["contract_id"]]
        for f in fields:
            got = bool(r[f]) if isinstance(want[f], bool) else r[f]
            assert got == want[f], f"{r['contract_id']}.{f}: {got!r} != {want[f]!r}"


def test_liability_basis_three_states(system, truth):
    """The three-state field the absence engine depends on."""
    by_id = {t["contract_id"]: t for t in truth}
    for r in system.store.scan_all():
        assert r["liability_cap_basis"] == by_id[r["contract_id"]]["liability_cap_style"]


# --- routing --------------------------------------------------------------

@pytest.mark.parametrize("query,expected", [
    ("How many contracts have auto renewal?", QueryType.AGGREGATE),
    ("Which contracts have no liability cap?", QueryType.ABSENCE),
    ("Which contracts lack indemnification?", QueryType.ABSENCE),
    ("Which contracts have penalties above $100,000?", QueryType.NUMERIC),
    ("List contracts expiring in Q1 2026", QueryType.TEMPORAL),
    ("Which vendors are on SOWs issued under Delaware master agreements?", QueryType.MULTIHOP),
    ("List every contract governed by Texas law", QueryType.ENUMERATE),
    ("What does the indemnification clause say?", QueryType.SEMANTIC),
])
def test_router_classification(system, query, expected):
    assert system.plan(query).query_type == expected


def test_inclusive_comparator_beats_strict():
    """'at or above' must not be read as a strict '>'.

    Regression guard: the strict pattern contains the inclusive one's keyword,
    so pattern order in COMPARATORS silently decides whether every row sitting
    exactly on the boundary is dropped.
    """
    from app.router.classify import QueryRouter
    r = QueryRouter()
    assert r.route("liability capped at or above $2,000,000").operator == "gte"
    assert r.route("liability capped above $2,000,000").operator == "gt"
    assert r.route("penalties at most $50,000").operator == "lte"
    assert r.route("penalties under $50,000").operator == "lt"


# --- engine correctness ---------------------------------------------------

def test_golden_set_routed_is_exact(system):
    """Every structural query must be answered exactly. No partial credit."""
    from app.eval.run_eval import score
    failures = []
    for g in build_golden_set(CORPUS / "ground_truth.json"):
        if g.category == QueryType.SEMANTIC:
            continue
        s = score(system.ask(g.query), g)
        if s["score"] < 1.0:
            failures.append((g.query, s["detail"]))
    assert not failures, f"non-exact structural answers: {failures}"


def test_completeness_flag_is_honest(system):
    """complete=True may only be set by an engine that scanned the corpus."""
    n = system.store.count()
    for q in ["How many contracts have auto renewal?",
              "Which contracts have no liability cap?",
              "List contracts expiring in Q1 2026"]:
        a = system.ask(q)
        assert a.complete is True and a.scanned == n

    a = system.ask("What does the indemnification clause say?")
    assert a.complete is False and a.scanned < n


def test_absence_reports_both_shapes(system):
    """Omitted clause and explicit disclaimer must both be found."""
    a = system.ask("Which contracts have no liability cap?")
    assert "omitted" in a.explanation and "explicit_unlimited" in a.explanation


def test_absence_abstains_without_determinacy(system, monkeypatch):
    """With no basis column and sparse coverage, absence must decline to answer."""
    import app.engines.core as core
    monkeypatch.setattr(core, "BASIS_FIELDS", {})          # remove determinacy signal
    monkeypatch.setattr(core, "ABSENCE_COVERAGE_FLOOR", 0.99)
    a = system.ask("Which contracts have no liability cap?")
    assert a.abstained and "cannot be distinguished" in a.explanation


# --- characterisation: the baseline must fail -----------------------------

def test_baseline_cannot_count(system, truth):
    """The headline claim. Top-k sees k chunks, so its count is of the window."""
    true_count = sum(1 for t in truth if t["auto_renew"])
    b = system.ask_baseline("How many contracts have auto renewal?")
    assert b.value != true_count
    assert b.complete is False
    assert b.scanned < system.store.count()


def test_baseline_misses_absence(system, truth):
    true_ids = {t["contract_id"] for t in truth if t["liability_cap_usd"] is None}
    got = set(system.ask_baseline("Which contracts have no liability cap?").value or [])
    recall = len(got & true_ids) / len(true_ids)
    assert recall < 0.25, f"baseline recall unexpectedly high ({recall:.2f}) -- re-check the claim"


def test_baseline_parity_on_semantic(system):
    """Control. Both systems share the retrieval path, so they must agree here.

    If this ever diverges, the comparison elsewhere is no longer apples to apples.
    """
    q = "What does the indemnification clause say?"
    assert set(system.ask(q).value or []) == set(system.ask_baseline(q).value or [])
