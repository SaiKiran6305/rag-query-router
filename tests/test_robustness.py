"""
Tests for the robustness work: temporal operators, lexicon coverage, scope
filters, the SQL predicate compiler, semantic abstention, and the LLM tier.

The LLM tier is tested with an injected fake, so the suite needs no network
and no credentials. What is under test is the part that matters for safety:
that every proposal is validated against the schema before it can execute.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from app.index.structured import StructuredStore, compile_predicates
from app.models import QueryType
from app.pipeline import ContractIntelligence
from app.router.classify import QueryRouter
from app.router.llm import HybridRouter, LLMPlanner, build_router, validate

CORPUS = Path(__file__).resolve().parent.parent / "data" / "corpus"


@pytest.fixture(scope="module")
def system() -> ContractIntelligence:
    return ContractIntelligence(CORPUS / "contracts")


@pytest.fixture(scope="module")
def truth() -> list[dict]:
    return json.loads((CORPUS / "ground_truth.json").read_text(encoding="utf-8"))


def ids(truth, pred) -> set[str]:
    return {t["contract_id"] for t in truth if pred(t)}


# --- temporal: operators, windows, fields -----------------------------------

@pytest.mark.parametrize("query,field,op,value", [
    ("What expires before June 2026?", "expiry_date", "lt", "2026-06-01"),
    ("Which contracts expire after March 1, 2027?", "expiry_date", "gt", "2027-03-01"),
    ("Contracts expiring on or after 2027-03-01", "expiry_date", "gte", "2027-03-01"),
    ("Contracts expiring on or before 12/31/2026", "expiry_date", "lte", "2026-12-31"),
    ("Contracts ending in the first quarter of 2026", "expiry_date", "between", ("2026-01-01", "2026-03-31")),
    ("List contracts expiring in Q1 2026", "expiry_date", "between", ("2026-01-01", "2026-03-31")),
    ("Contracts expiring in March 2026", "expiry_date", "between", ("2026-03-01", "2026-03-31")),
    ("Contracts that started in 2025", "effective_date", "between", ("2025-01-01", "2025-12-31")),
    ("Which contracts were signed between January 2024 and June 2024?", "effective_date", "between",
     ("2024-01-01", "2024-06-30")),
])
def test_temporal_plans(query, field, op, value):
    p = QueryRouter().route(query)
    assert (p.query_type, p.field, p.operator, p.value) == (QueryType.TEMPORAL, field, op, value)


def test_inclusive_date_cue_beats_strict():
    """'on or after' contains 'after'; strict-first would drop the boundary day."""
    r = QueryRouter()
    assert r.route("contracts expiring on or after March 1, 2027").operator == "gte"
    assert r.route("contracts expiring after March 1, 2027").operator == "gt"


def test_relative_dates_use_injected_today():
    r = QueryRouter(today=date(2026, 1, 15))
    assert r.route("Which contracts expire in the next 90 days?").value == ("2026-01-15", "2026-04-15")
    assert r.route("contracts expiring this year").value == ("2026-01-01", "2026-12-31")
    p = r.route("Which contracts have already expired?")
    assert (p.operator, p.value) == ("lt", "2026-01-15")


def test_temporal_answer_before_is_exact(system, truth):
    a = system.ask("What expires before June 2026?")
    assert set(a.value) == ids(truth, lambda t: t["expiry_date"] < "2026-06-01")
    assert a.complete


def test_default_date_field_is_stated(system):
    a = system.ask("contracts in 2027")
    assert "assumed expiry_date" in a.explanation


# --- lexicon and precedence --------------------------------------------------

@pytest.mark.parametrize("query,qtype,field", [
    ("Which agreements are uncapped?", QueryType.ABSENCE, "liability_cap_usd"),
    ("Are there contracts where liability is unlimited?", QueryType.ABSENCE, "liability_cap_usd"),
    ("Count the deals that renew on their own", QueryType.AGGREGATE, "auto_renew"),
    ("Show me all Delaware-governed contracts", QueryType.ENUMERATE, "governing_law"),
    ("Which contracts are under Texas law?", QueryType.ENUMERATE, "governing_law"),
    ("Which contracts are Statements of Work?", QueryType.ENUMERATE, "agreement_type"),
    ("Contracts with a term longer than 24 months", QueryType.NUMERIC, "term_months"),
    ("Which agreements require insurance coverage of at least $2,000,000?", QueryType.NUMERIC,
     "insurance_min_usd"),
])
def test_paraphrase_routing(query, qtype, field):
    p = QueryRouter().route(query)
    assert (p.query_type, p.field) == (qtype, field)


def test_negated_comparator_is_not_absence():
    """'no less than' is a comparator; its 'no' must not trigger the absence engine."""
    r = QueryRouter()
    p = r.route("Which contracts have penalties no less than $50,000?")
    assert (p.query_type, p.operator) == (QueryType.NUMERIC, "gte")
    p = r.route("Which contracts have liability capped not more than $1,000,000?")
    assert (p.query_type, p.operator) == (QueryType.NUMERIC, "lte")


def test_total_number_of_is_a_count_not_a_sum():
    p = QueryRouter().route("Total number of contracts with penalties")
    assert p.aggregation == "count"


def test_scope_filter_is_applied(system, truth):
    """'Texas contracts' used to be dropped, returning the count over every contract."""
    a = system.ask("How many Texas contracts auto-renew?")
    assert a.value == len(ids(truth, lambda t: t["auto_renew"] and t["governing_law"] == "Texas"))


def test_multihop_parent_and_child_types():
    p = QueryRouter().route("Find SOWs referencing Texas-governed MSAs")
    assert p.query_type == QueryType.MULTIHOP
    assert p.secondary == {"parent_governing_law": "Texas",
                           "parent_type": "Master Services Agreement",
                           "child_type": "Statement of Work"}


# --- SQL compiler --------------------------------------------------------------

def test_compiler_rejects_unknown_column_and_operator():
    with pytest.raises(ValueError):
        compile_predicates([("vendor; DROP TABLE contracts", "eq", "x")])
    with pytest.raises(ValueError):
        compile_predicates([("vendor", "regexp", "x")])


def test_user_text_is_only_ever_a_parameter(system):
    evil = "Texas'); DROP TABLE contracts; --"
    where, params = compile_predicates([("governing_law", "eq", evil)])
    assert evil not in where and params == [evil]
    assert system.store.select([("governing_law", "eq", evil)]) == []
    assert system.store.count() == 200


def test_compiler_matches_python_semantics(system):
    """The baseline and engines share the compiler; spot-check it against rows."""
    from app.engines.core import _matches
    rows = system.store.scan_all()
    cases = [("auto_renew", "not_null", None), ("auto_renew", "eq", False),
             ("liability_cap_usd", "is_null", None), ("late_penalty_usd", "gte", 250_000),
             ("governing_law", "eq", "new york"), ("expiry_date", "between", ("2026-01-01", "2026-06-30"))]
    for f, op, v in cases:
        want = {r["contract_id"] for r in rows if _matches(r, f, op, v)}
        got = {r["contract_id"] for r in system.store.select([(f, op, v)])}
        assert got == want, (f, op, v)


def test_empty_store_coverage_is_zero():
    assert set(StructuredStore().field_coverage().values()) == {0.0}


# --- semantic abstention -------------------------------------------------------

@pytest.mark.parametrize("query", ["What is the weather in Dallas?", "Best biryani recipe",
                                   "How many companies have no employees?"])
def test_out_of_scope_is_declined(system, query):
    a = system.ask(query)
    assert a.abstained and a.value is None and "Declined" in a.explanation


def test_real_clause_question_still_answers_with_passages(system):
    a = system.ask("What does the indemnification clause say?")
    assert not a.abstained and not a.complete
    assert all(i.citation and "INDEMNIFICATION" in i.citation.snippet.upper() for i in a.items[:3])


def test_rrf_score_cannot_detect_out_of_scope(system):
    """Why the gate uses raw similarity: fused scores are rank-only."""
    idx = system.index
    off = idx.search("best biryani recipe", k=1)[0][1]
    on = idx.search("What does the indemnification clause say?", k=1)[0][1]
    assert off >= on                                   # RRF can't tell them apart
    assert idx.relevance("best biryani recipe") < idx.relevance("What does the indemnification clause say?")


# --- LLM tier -------------------------------------------------------------------

class FakeLLM:
    def __init__(self, reply):
        self.reply, self.calls = reply, 0

    def __call__(self, system_prompt, question):
        self.calls += 1
        assert "liability_cap_usd" in system_prompt      # the schema catalog is in the prompt
        return self.reply(question) if callable(self.reply) else self.reply


def hybrid(reply) -> tuple[HybridRouter, FakeLLM]:
    fake = FakeLLM(reply)
    return HybridRouter(planner=LLMPlanner(complete_json=fake)), fake


def test_llm_not_called_when_rules_are_confident():
    router, fake = hybrid({"query_type": "aggregate"})
    router.route("How many contracts have auto renewal?")
    router.route("What does the indemnification clause say?")   # semantic with evidence
    assert fake.calls == 0


def test_llm_rescues_a_fallthrough(truth):
    router, fake = hybrid({"answerable": True, "query_type": "aggregate", "field": "auto_renew",
                           "operator": "eq", "value": True, "aggregation": "count"})
    s = ContractIntelligence(CORPUS / "contracts", router=router)
    q = "How many deals roll over each year?"
    assert QueryRouter().route(q).needs_escalation
    plan = s.plan(q)
    assert plan.tier == "llm" and plan.query_type == QueryType.AGGREGATE
    assert s.ask(q, plan).value == sum(t["auto_renew"] for t in truth)


def test_llm_hallucinated_field_is_rejected():
    router, _ = hybrid({"answerable": True, "query_type": "aggregate", "field": "employee_count",
                        "operator": "is_null", "aggregation": "count"})
    plan = router.route("How many companies have no employees?")
    assert plan.tier == "rules" and any("unknown field" in s for s in plan.signals)


def test_llm_unanswerable_declines(system):
    router, _ = hybrid({"answerable": False, "reason": "headcount is not recorded in contracts"})
    plan = router.route("How many people work in the legal department?")
    assert plan.tier == "llm"
    a = system.ask(plan.query, plan)
    assert a.abstained and "headcount" in a.explanation


def test_confident_misroute_is_not_escalated():
    """Characterisation of a known limit, not a feature.

    'vendors' matches the vendor field and 'how many' is a count, so the rules
    produce a confident AGGREGATE plan and the LLM is never consulted -- the
    answer is a count of every contract. Escalation only catches fall-throughs.
    If this test starts failing, the limitation was fixed: update README.
    """
    router, fake = hybrid({"answerable": False, "reason": "n/a"})
    plan = router.route("How many vendors have more than 500 staff?")
    assert plan.query_type == QueryType.AGGREGATE and plan.tier == "rules" and fake.calls == 0


def test_bm25_tokens_ignore_punctuation_and_question_words():
    from app.index.retrieval import tokenize
    assert tokenize("5. INDEMNIFICATION. Supplier shall") == ["5", "indemnification", "supplier", "shall"]
    assert tokenize("What does the indemnification clause say?") == ["indemnification"]


def test_llm_calls_are_cached():
    router, fake = hybrid({"answerable": False, "reason": "n/a"})
    router.route("best biryani recipe")
    router.route("Best biryani recipe ")
    assert fake.calls == 1


@pytest.mark.parametrize("proposal,error", [
    ({"query_type": "numeric", "field": "vendor", "operator": "gt", "value": 5}, "not valid"),
    ({"query_type": "temporal", "field": "expiry_date", "operator": "lt", "value": "next spring"}, "ISO date"),
    ({"query_type": "enumerate", "field": "governing_law", "operator": "eq", "value": "Ontario"}, "not an allowed"),
    ({"query_type": "aggregate", "field": "auto_renew", "operator": "eq", "value": True,
      "aggregation": "avg"}, "cannot avg"),
    ({"query_type": "teleport"}, "unknown query_type"),
    ({"query_type": "enumerate", "field": "vendor", "operator": "eq", "value": "x",
      "filters": {"vendor_ssn": "1"}}, "cannot filter"),
])
def test_validator_rejects_bad_proposals(proposal, error):
    plan, errors = validate("q", proposal)
    assert plan is None and any(error in e for e in errors), errors


def test_validator_coerces_and_canonicalises():
    plan, errors = validate("q", {"query_type": "numeric", "field": "liability_cap_usd",
                                  "operator": "gte", "value": "$2,000,000",
                                  "filters": {"governing_law": "new york"}})
    assert not errors and plan.value == 2_000_000 and plan.filters == {"governing_law": "New York"}


def test_hybrid_without_key_fails_loudly(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        build_router("hybrid")
    assert build_router("rules").kind == "rules"


# --- extraction on documents the extractor cannot read ------------------------

UNSTRUCTURED = (
    "SUPPLY AGREEMENT\nThis agreement is made between Acme and Beta. Beta will supply widgets. "
    "Each party's total liability shall not exceed the fees paid in the prior twelve months. "
    "This agreement is governed by the laws of the State of Delaware.\n"
)


def test_unstructured_document_reports_unknown_not_omitted():
    """No recognisable clause headings -> the extractor may not claim absence.

    On CUAD the old behaviour marked 255 contracts that *have* a liability
    clause as basis='omitted', which the absence engine treats as settled fact.
    """
    from app.ingest.extract import DeterministicExtractor
    d = DeterministicExtractor().extract("X-1", UNSTRUCTURED, "x.txt")
    assert d.value("liability_cap_basis") == "unknown"
    assert d.value("confidentiality_basis") == "unknown"
    assert d.value("insurance_required") is None and d.value("indemnification") is None
    assert d.value("governing_law") == "Delaware"


def test_absence_abstains_on_unreadable_corpus(tmp_path):
    (tmp_path / "X-1.txt").write_text(UNSTRUCTURED, encoding="utf-8")
    (tmp_path / "X-2.txt").write_text(UNSTRUCTURED.replace("Delaware", "Texas"), encoding="utf-8")
    s = ContractIntelligence(tmp_path)
    for q in ["Which contracts have no liability cap?", "Which contracts do not require insurance?"]:
        a = s.ask(q)
        assert a.abstained and "cannot be distinguished" in a.explanation


def test_synthetic_extraction_is_exact_on_all_fields():
    from app.eval.extraction_eval import eval_synthetic
    rep = eval_synthetic(CORPUS)
    assert rep["fields"] == 19 and rep["fields_at_100"] == 19, rep["examples_of_misses"]


def test_cuad_adapter_and_eval(tmp_path):
    from app.eval.extraction_eval import eval_cuad
    from app.ingest.cuad import export
    fixture = {"version": "test", "data": [{
        "title": "ACME CORP - SUPPLY AGREEMENT", "paragraphs": [{
            "context": UNSTRUCTURED,
            "qas": [
                {"id": "ACME__Governing Law", "question": "", "is_impossible": False,
                 "answers": [{"text": "governed by the laws of the State of Delaware", "answer_start": 0}]},
                {"id": "ACME__Cap On Liability", "question": "", "is_impossible": False,
                 "answers": [{"text": "total liability shall not exceed the fees", "answer_start": 0}]},
                {"id": "ACME__Insurance", "question": "", "is_impossible": True, "answers": []},
                {"id": "ACME__Non-Compete", "question": "", "is_impossible": True, "answers": []},
            ]}]}]}
    src = tmp_path / "CUADv1.json"
    src.write_text(json.dumps(fixture), encoding="utf-8")
    assert export(src, tmp_path / "cuad") == 1
    files = list((tmp_path / "cuad" / "contracts").glob("*.txt"))
    assert [f.name for f in files] == ["ACME_CORP_-_SUPPLY_AGREEMENT.txt"]
    labels = json.loads((tmp_path / "cuad" / "cuad_labels.json").read_text())
    assert "Non-Compete" not in labels["ACME_CORP_-_SUPPLY_AGREEMENT"]["labels"]   # outside schema
    rep = eval_cuad(tmp_path / "cuad")
    assert rep["detection"]["Governing Law"]["recall"] == 1.0
    assert rep["governing_law_value_accuracy"] == 1.0
    assert rep["liability_basis_omitted_but_experts_found_clause"] == 0
