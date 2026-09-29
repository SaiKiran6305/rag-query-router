"""
Execution engines. One per query type, plus a dispatcher.

Every structural engine answers through StructuredStore.select(), which runs a
parameterized WHERE clause over the whole table, and so sets
Answer.complete = True. SemanticEngine is the exception: it sees the top-k
retrieved chunks, not the corpus. That single flag is the honest difference
between this system and a retrieval pipeline: a count taken over eight
retrieved chunks is not a count, and nothing in a top-k pipeline can tell the
user so.

Two places decline to answer rather than guess:

AbsenceEngine gates on *determinacy*. A null value is ambiguous -- it can mean
the clause is genuinely absent, or that extraction missed it. Treating those as
the same thing would be exactly the overconfidence this project is arguing
against, so the engine abstains when it cannot tell the two apart. Determinacy
is deliberately not the same as value coverage: liability_cap_usd is null for
42% of this corpus because 42% of contracts genuinely have no cap, and gating
on raw coverage would abstain on a question it can answer exactly.

SemanticEngine abstains when retrieval cannot be right: when the question asks
for a count (a sample is not a count), or when nothing in the corpus is
relevant to it at all ("what is the weather in Dallas"). Before this, every
unrecognised question returned eight arbitrary contracts.
"""

from __future__ import annotations

import os
import time
from typing import Any

from app.index.retrieval import HybridIndex
from app.index.structured import Predicate, StructuredStore
from app.models import Answer, AnswerItem, Citation, QueryType
from app.router.classify import BOOLEAN_FIELDS, QueryPlan
from app.schema import BASIS_FIELDS as _REGISTRY_BASIS

# Below this determinacy, a null cannot be read as a genuine absence.
ABSENCE_COVERAGE_FLOOR = 0.80

# Some fields carry a companion column recording *why* a value is absent. For
# those, a null is a determined finding rather than a missing extraction, and
# raw value coverage is the wrong thing to gate on: liability_cap_usd is null
# for 42% of this corpus precisely because 42% of contracts genuinely have no
# cap. Determinacy is measured on the basis column instead.
BASIS_FIELDS: dict[str, str] = dict(_REGISTRY_BASIS)
INDETERMINATE_BASIS = {None, "", "unparsed", "unknown"}

# Minimum raw similarity for the semantic engine to answer at all. Calibrated
# for the TF-IDF backend on this corpus: off-topic questions score 0.00, real
# clause questions 0.145-0.203. Dense embeddings never score exactly zero, so the
# MiniLM default is a starting point to re-tune with the paraphrase eval.
SEMANTIC_MIN_RELEVANCE = {"tfidf": 0.05, "sentence-transformers": 0.30}

ITEM_LIMIT = 50


def _determinacy(store: "StructuredStore", field_name: str) -> tuple[float, str]:
    """Fraction of rows where the field's value is known to be settled.

    Returns (score, what_was_measured) so the explanation can state which
    signal the abstention decision actually rested on.
    """
    basis_col = BASIS_FIELDS.get(field_name)
    if basis_col:
        rows = store.scan_all()
        if not rows:
            return 0.0, basis_col
        settled = sum(1 for r in rows if r.get(basis_col) not in INDETERMINATE_BASIS)
        return settled / len(rows), basis_col
    return store.field_coverage().get(field_name, 0.0), field_name


def _scope(plan: QueryPlan) -> list[Predicate]:
    """Scope filters ("Texas contracts") as predicates every engine applies."""
    return [(col, "eq", val) for col, val in (plan.filters or {}).items()]


def _scope_text(plan: QueryPlan) -> str:
    if not plan.filters:
        return ""
    return " within " + ", ".join(f"{k}={v}" for k, v in plan.filters.items())


def _assumption_text(plan: QueryPlan) -> str:
    return "".join(f" Assumption: {a}." for a in plan.assumptions)


def _missing_text(store: StructuredStore, f: str | None, scope: list) -> str:
    """'Complete' means every row was checked -- not that every row had the field.

    A contract whose value was never extracted cannot match any predicate on
    it. On the synthetic corpus a null usually means the clause is genuinely
    absent; on real contracts it is often an extraction miss (CUAD: governing
    law found in ~72% of contracts). Either way the user should see the number.
    """
    if not f:
        return ""
    missing = store.count_where([*scope, (f, "is_null", None)])
    return f" {missing} contracts in scope have no extracted value for {f} and cannot match." if missing else ""


class Engine:
    name = "base"

    def run(self, plan: QueryPlan, store: StructuredStore, index: HybridIndex | None) -> Answer:
        raise NotImplementedError

    @staticmethod
    def _item(store: StructuredStore, row: dict[str, Any], field_name: str | None,
              reason: str) -> AnswerItem:
        cit = store.citation(row["contract_id"], field_name) if field_name else None
        return AnswerItem(contract_id=row["contract_id"], reason=reason, citation=cit)

    def _abstain(self, plan: QueryPlan, reason: str, t0: float, scanned: int = 0,
                 complete: bool = False) -> Answer:
        return Answer(plan.query, plan.query_type, self.name, value=None, abstained=True,
                      complete=complete, scanned=scanned, explanation=reason,
                      latency_ms=(time.perf_counter() - t0) * 1000)


class SemanticEngine(Engine):
    """Ordinary retrieval. Correct for open-ended clause questions and nothing else."""
    name = "semantic"
    k = 8

    def run(self, plan: QueryPlan, store: StructuredStore, index: HybridIndex | None) -> Answer:
        t0 = time.perf_counter()
        if index is None:
            return self._abstain(plan, "no retrieval index available", t0)

        # A count cannot come from a sample. If the router saw "how many" but
        # could not map it to a field, the honest answer is to say so.
        if "count" in plan.unresolved or plan.aggregation == "count":
            return self._abstain(plan, (
                "Declined. The question asks for a count, but it does not match any field the "
                "system has extracted, and retrieval cannot count -- it would be counting a "
                "sample of 8 chunks. Known fields are listed at /api/schema."), t0)

        relevance = index.relevance(plan.query)
        floor = float(os.environ.get("SEMANTIC_MIN_RELEVANCE",
                                     SEMANTIC_MIN_RELEVANCE.get(index.embed.kind, 0.05)))
        if relevance < floor:
            return self._abstain(plan, (
                f"Declined. Nothing in the contract corpus is relevant to this question "
                f"(best similarity {relevance:.3f}, below the {floor:.2f} floor)."), t0)

        hits = index.search(plan.query, k=self.k)
        items = [
            AnswerItem(
                contract_id=c.contract_id,
                reason=f"retrieval score {s:.4f}",
                citation=Citation(c.contract_id, c.start, c.end, c.text.strip()[:600]),
            )
            for c, s in hits
        ]
        caution = ""
        if plan.unresolved:
            caution = (f" Caution: the question contains {', '.join(plan.unresolved)} intent the "
                       f"router could not map to a field, so this sample may not answer it.")
        elif plan.confidence < 0.5:
            caution = " Caution: low routing confidence; nothing marked this as a clause question."
        return Answer(
            query=plan.query, query_type=plan.query_type, engine=self.name,
            value=[i.contract_id for i in items], items=items,
            complete=False,                      # saw k chunks, not the corpus
            scanned=len(hits),
            explanation=(
                f"Answered from the top {self.k} retrieved chunks (best similarity "
                f"{relevance:.3f}). This result is a sample, not a complete set, and must not "
                f"be read as a count." + caution
            ),
            latency_ms=(time.perf_counter() - t0) * 1000,
        )


class AggregateEngine(Engine):
    """Counts, sums, averages and percentages, computed by SQL over the full table."""
    name = "aggregate"

    def run(self, plan: QueryPlan, store: StructuredStore, index: HybridIndex | None) -> Answer:
        t0 = time.perf_counter()
        f = plan.field
        scope = _scope(plan)
        preds = scope + ([(f, plan.operator, plan.value)] if f else [])
        n_scope = store.count_where(scope)
        n_matched = store.count_where(preds)

        kind = plan.aggregation or "count"
        if kind == "percentage":
            value: Any = round(100.0 * n_matched / max(n_scope, 1), 1)
        elif kind in ("sum", "avg") and f:
            total, n_values = store.sum_where(f, preds)
            if kind == "sum":
                value = total if total is not None else 0
            else:
                value = round(total / n_values, 2) if n_values else None
        else:
            kind = "count"
            value = n_matched

        rows = store.select(preds, limit=ITEM_LIMIT)
        return Answer(
            query=plan.query, query_type=plan.query_type, engine=self.name,
            value=value,
            items=[self._item(store, r, f, f"{f}={r.get(f)}") for r in rows],
            complete=True, scanned=store.count(),
            explanation=(
                f"Computed {kind} over all {n_scope} contracts{_scope_text(plan)} on field "
                f"'{f}' ({plan.operator}={plan.value}). {n_matched} matched."
                + _missing_text(store, f, scope)
            ),
            latency_ms=(time.perf_counter() - t0) * 1000,
        )


class AbsenceEngine(Engine):
    """Enumerate then check. The category where retrieval ranks the answer last.

    Contracts lacking a liability cap fall into two textual shapes: the clause
    is missing entirely, or a clause is present stating no limitation applies.
    Semantic search surfaces neither -- the first has no text to match, and the
    second is nearest in embedding space to the contracts that *do* cap.
    """
    name = "absence"

    def run(self, plan: QueryPlan, store: StructuredStore, index: HybridIndex | None) -> Answer:
        t0 = time.perf_counter()
        f = plan.field
        n_total = store.count()

        coverage, measured_on = _determinacy(store, f) if f else (0.0, "none")
        boolean_field = f in BOOLEAN_FIELDS

        # A null is ambiguous for every field: for value fields it may be an
        # extraction miss, and for booleans the extractor now writes null (not
        # False) when it could not see the document's structure. Determinacy
        # decides whether we are entitled to an answer at all. On the synthetic
        # corpus every boolean is determined (coverage 100%), so this gate only
        # bites on documents the extractor cannot read -- which is the point.
        if coverage < ABSENCE_COVERAGE_FLOOR:
            return self._abstain(plan, (
                f"Abstained. Determinacy for '{f}' (measured on '{measured_on}') is "
                f"{coverage:.0%}, below the {ABSENCE_COVERAGE_FLOOR:.0%} floor, so a null "
                f"cannot be distinguished from an extraction miss. Answering would report "
                f"extraction failures as contractual absences."), t0, scanned=n_total, complete=True)

        preds = _scope(plan) + [(f, "falsy" if boolean_field else "is_null", None)]
        matched = store.select(preds)
        basis_col = BASIS_FIELDS.get(f)
        if boolean_field:
            reason_for = lambda r: f"{f} is false"          # noqa: E731
        else:
            reason_for = lambda r: (                         # noqa: E731
                f"no value for {f}" + (f" (basis: {r.get(basis_col)})" if basis_col else "")
            )

        # Where we know *why* the value is absent, surface it -- the distinction
        # between an omitted clause and an explicit disclaimer matters legally.
        breakdown: dict[str, int] = {}
        if basis_col:
            for r in matched:
                b = r.get(basis_col) or "unknown"
                breakdown[b] = breakdown.get(b, 0) + 1

        return Answer(
            query=plan.query, query_type=plan.query_type, engine=self.name,
            value=len(matched) if plan.aggregation == "count" else [r["contract_id"] for r in matched],
            items=[self._item(store, r, f, reason_for(r)) for r in matched[:ITEM_LIMIT]],
            complete=True, scanned=n_total,
            explanation=(
                f"Checked all {n_total} contracts{_scope_text(plan)} for absence of '{f}'. "
                f"Found {len(matched)}."
                + (f" Breakdown by basis: {breakdown}." if breakdown else "")
                + f" Determinacy for this field (on '{measured_on}'): {coverage:.0%}."
            ),
            latency_ms=(time.perf_counter() - t0) * 1000,
        )


class _PredicateEngine(Engine):
    """Shared body for engines that are one typed predicate over the table."""

    def run(self, plan: QueryPlan, store: StructuredStore, index: HybridIndex | None) -> Answer:
        t0 = time.perf_counter()
        f, op, v = plan.field, plan.operator, plan.value
        matched = store.select(_scope(plan) + [(f, op, v)])
        value = len(matched) if plan.aggregation == "count" else [r["contract_id"] for r in matched]
        return Answer(
            query=plan.query, query_type=plan.query_type, engine=self.name,
            value=value,
            items=[self._item(store, r, f, self._reason(r, f, op, v)) for r in matched[:self.item_limit]],
            complete=True, scanned=store.count(),
            explanation=(
                f"{self.verb} {f} {op} {v} across all {store.count()} contracts{_scope_text(plan)}; "
                f"{len(matched)} matched." + _assumption_text(plan) + _missing_text(store, f, _scope(plan))
            ),
            latency_ms=(time.perf_counter() - t0) * 1000,
        )

    item_limit = ITEM_LIMIT
    verb = "Applied"

    def _reason(self, r: dict[str, Any], f: str, op: str, v: Any) -> str:
        return f"{f}={r.get(f)}"


class NumericEngine(_PredicateEngine):
    """Typed magnitude comparison. Embeddings have no ordering over numbers."""
    name = "numeric"

    def _reason(self, r, f, op, v):
        return f"{f}={r.get(f)} {op} {v}"


class TemporalEngine(_PredicateEngine):
    """Date predicates -- before, after, on-or-before, on-or-after, between -- over ISO dates."""
    name = "temporal"
    verb = "Filtered"


class EnumerateEngine(_PredicateEngine):
    """Complete listing. Distinct from semantic search because completeness is guaranteed."""
    name = "enumerate"
    item_limit = 100
    verb = "Complete listing where"


class MultiHopEngine(Engine):
    """Traversal a single retrieval pass cannot perform.

    'Vendors on statements of work issued under Delaware-governed master
    agreements' requires: filter parents by law (and type), collect their ids,
    then filter children whose parent pointer is in that set. Two dependent
    steps, where the second needs the output of the first.
    """
    name = "multihop"

    def run(self, plan: QueryPlan, store: StructuredStore, index: HybridIndex | None) -> Answer:
        t0 = time.perf_counter()
        hints = plan.secondary or {}

        parent_preds: list[Predicate] = []
        if "parent_governing_law" in hints:
            parent_preds.append(("governing_law", "eq", hints["parent_governing_law"]))
        if "parent_type" in hints:
            parent_preds.append(("agreement_type", "eq", hints["parent_type"]))
        parent_ids = [r["contract_id"] for r in store.select(parent_preds)]

        child_preds: list[Predicate] = [("amends_contract_id", "in", parent_ids)]
        if "child_type" in hints:
            child_preds.append(("agreement_type", "eq", hints["child_type"]))
        children = store.select(_scope(plan) + child_preds)
        value = len(children) if plan.aggregation == "count" else [r["contract_id"] for r in children]

        return Answer(
            query=plan.query, query_type=plan.query_type, engine=self.name,
            value=value,
            items=[
                self._item(store, r, "amends_contract_id",
                           f"amends {r.get('amends_contract_id')} (vendor: {r.get('vendor')})")
                for r in children[:ITEM_LIMIT]
            ],
            complete=True, scanned=store.count(),
            explanation=(
                f"Hop 1: {len(parent_ids)} parent contracts matched {hints or 'no filter'}. "
                f"Hop 2: {len(children)} contracts reference one of them."
            ),
            latency_ms=(time.perf_counter() - t0) * 1000,
        )


def _matches(row: dict[str, Any], field_name: str | None, op: str | None, value: Any) -> bool:
    """In-memory predicate check with the same semantics as compile_predicates.

    Kept for callers that already hold a row. Engines and the baseline query
    through StructuredStore.select() so predicate semantics live in one place.
    """
    if field_name is None:
        return True
    cur = row.get(field_name)
    if op in (None, "not_null"):
        return cur is not None and cur != 0 if field_name in BOOLEAN_FIELDS else cur is not None
    if op == "is_null":
        return cur is None
    if op == "falsy":
        return not cur
    if op == "eq":
        if isinstance(value, bool):
            return bool(cur) == value
        if isinstance(value, str) and isinstance(cur, str):
            return cur.lower() == value.lower()
        return cur == value
    if op == "ne":
        return cur != value
    if cur is None or value is None:
        return False
    try:
        if op == "gt":
            return cur > value
        if op == "gte":
            return cur >= value
        if op == "lt":
            return cur < value
        if op == "lte":
            return cur <= value
        if op == "between":
            lo, hi = value
            return lo <= cur <= hi
        if op == "contains":
            return str(value).lower() in str(cur).lower()
        if op == "in":
            return cur in value
    except TypeError:
        return False
    return False


ENGINES: dict[QueryType, Engine] = {
    QueryType.SEMANTIC: SemanticEngine(),
    QueryType.AGGREGATE: AggregateEngine(),
    QueryType.ABSENCE: AbsenceEngine(),
    QueryType.NUMERIC: NumericEngine(),
    QueryType.TEMPORAL: TemporalEngine(),
    QueryType.MULTIHOP: MultiHopEngine(),
    QueryType.ENUMERATE: EnumerateEngine(),
}


def dispatch(plan: QueryPlan, store: StructuredStore, index: HybridIndex | None) -> Answer:
    if plan.abstain_reason:
        # The planner itself concluded the question is outside the schema.
        return Answer(plan.query, plan.query_type, "router", value=None, abstained=True,
                      explanation=f"Declined by the {plan.tier} router: {plan.abstain_reason}")
    return ENGINES[plan.query_type].run(plan, store, index)
