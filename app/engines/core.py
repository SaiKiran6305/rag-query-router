"""
Execution engines. One per query type, plus a dispatcher.

Every engine except SemanticEngine sets Answer.complete = True, because every
engine except SemanticEngine examines the full corpus. That single flag is the
honest difference between this system and a retrieval pipeline: a count taken
over eight retrieved chunks is not a count, and nothing in a top-k pipeline
can tell the user so.

AbsenceEngine carries the one piece of genuine epistemics here. A null value
is ambiguous -- it can mean the clause is genuinely absent, or that extraction
missed it. Treating those as the same thing would be exactly the overconfidence
this project is arguing against, so the engine gates on *determinacy* (do we
know why the value is missing?) and abstains when it cannot tell the two apart.
Determinacy is deliberately not the same as value coverage: liability_cap_usd
is null for 42% of this corpus because 42% of contracts genuinely have no cap,
and gating on raw coverage would abstain on a question it can answer exactly.
"""

from __future__ import annotations

import time
from typing import Any

from app.index.retrieval import HybridIndex
from app.index.structured import StructuredStore
from app.models import Answer, AnswerItem, QueryType
from app.router.classify import BOOLEAN_FIELDS, QueryPlan

# Below this determinacy, a null cannot be read as a genuine absence.
ABSENCE_COVERAGE_FLOOR = 0.80

# Some fields carry a companion column recording *why* a value is absent. For
# those, a null is a determined finding rather than a missing extraction, and
# raw value coverage is the wrong thing to gate on: liability_cap_usd is null
# for 42% of this corpus precisely because 42% of contracts genuinely have no
# cap. Determinacy is measured on the basis column instead.
BASIS_FIELDS: dict[str, str] = {
    "liability_cap_usd": "liability_cap_basis",
    "confidentiality_years": "confidentiality_basis",
}
INDETERMINATE_BASIS = {None, "", "unparsed", "unknown"}


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


class Engine:
    name = "base"

    def run(self, plan: QueryPlan, store: StructuredStore, index: HybridIndex | None) -> Answer:
        raise NotImplementedError

    @staticmethod
    def _item(store: StructuredStore, row: dict[str, Any], field_name: str | None,
              reason: str) -> AnswerItem:
        cit = store.citation(row["contract_id"], field_name) if field_name else None
        return AnswerItem(contract_id=row["contract_id"], reason=reason, citation=cit)


class SemanticEngine(Engine):
    """Ordinary retrieval. Correct for open-ended clause questions and nothing else."""
    name = "semantic"

    def run(self, plan: QueryPlan, store: StructuredStore, index: HybridIndex | None) -> Answer:
        t0 = time.perf_counter()
        if index is None:
            return Answer(plan.query, plan.query_type, self.name, value=None,
                          abstained=True, explanation="no retrieval index available",
                          latency_ms=(time.perf_counter() - t0) * 1000)
        hits = index.search(plan.query, k=8)
        items = [
            AnswerItem(
                contract_id=c.contract_id,
                reason=f"retrieval score {s:.4f}",
                citation=None,
            )
            for c, s in hits
        ]
        return Answer(
            query=plan.query, query_type=plan.query_type, engine=self.name,
            value=[i.contract_id for i in items], items=items,
            complete=False,                      # saw k chunks, not the corpus
            scanned=len(hits),
            explanation=(
                "Answered from the top 8 retrieved chunks. This result is a sample, "
                "not a complete set, and must not be read as a count."
            ),
            latency_ms=(time.perf_counter() - t0) * 1000,
        )


class AggregateEngine(Engine):
    """Counts, sums and averages over a full corpus scan."""
    name = "aggregate"

    def run(self, plan: QueryPlan, store: StructuredStore, index: HybridIndex | None) -> Answer:
        t0 = time.perf_counter()
        rows = store.scan_all()
        f = plan.field
        matched = [r for r in rows if _matches(r, f, plan.operator, plan.value)]

        kind = plan.aggregation or "count"
        if kind == "count":
            value: Any = len(matched)
        elif kind == "percentage":
            value = round(100.0 * len(matched) / max(len(rows), 1), 1)
        elif kind == "sum":
            value = sum(r[f] for r in matched if isinstance(r.get(f), (int, float)))
        elif kind == "avg":
            vals = [r[f] for r in matched if isinstance(r.get(f), (int, float))]
            value = round(sum(vals) / len(vals), 2) if vals else None
        else:
            value = len(matched)

        return Answer(
            query=plan.query, query_type=plan.query_type, engine=self.name,
            value=value,
            items=[self._item(store, r, f, f"{f}={r.get(f)}") for r in matched[:50]],
            complete=True, scanned=len(rows),
            explanation=(
                f"Computed {kind} over all {len(rows)} contracts on field '{f}' "
                f"({plan.operator}={plan.value}). {len(matched)} matched."
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
        rows = store.scan_all()

        coverage, measured_on = _determinacy(store, f) if f else (0.0, "none")
        boolean_field = f in BOOLEAN_FIELDS

        # For boolean fields absence is an explicit False, so determinacy is not
        # a confounder. For value fields a null is ambiguous, and determinacy
        # decides whether we are entitled to an answer at all.
        if not boolean_field and coverage < ABSENCE_COVERAGE_FLOOR:
            return Answer(
                query=plan.query, query_type=plan.query_type, engine=self.name,
                value=None, complete=True, scanned=len(rows), abstained=True,
                explanation=(
                    f"Abstained. Determinacy for '{f}' (measured on '{measured_on}') is "
                    f"{coverage:.0%}, below the {ABSENCE_COVERAGE_FLOOR:.0%} floor, so a null "
                    f"cannot be distinguished from an extraction miss. Answering would report "
                    f"extraction failures as contractual absences."
                ),
                latency_ms=(time.perf_counter() - t0) * 1000,
            )

        if boolean_field:
            matched = [r for r in rows if not r.get(f)]
            reason_for = lambda r: f"{f} is false"          # noqa: E731
        else:
            matched = [r for r in rows if r.get(f) is None]
            reason_for = lambda r: (                         # noqa: E731
                f"no value for {f}"
                + (f" (basis: {r.get('liability_cap_basis')})" if f == "liability_cap_usd" else "")
            )

        # Where we know *why* the value is absent, surface it -- the distinction
        # between an omitted clause and an explicit disclaimer matters legally.
        breakdown: dict[str, int] = {}
        if f == "liability_cap_usd":
            for r in matched:
                b = r.get("liability_cap_basis") or "unknown"
                breakdown[b] = breakdown.get(b, 0) + 1

        return Answer(
            query=plan.query, query_type=plan.query_type, engine=self.name,
            value=len(matched) if plan.aggregation == "count" else [r["contract_id"] for r in matched],
            items=[self._item(store, r, f, reason_for(r)) for r in matched[:50]],
            complete=True, scanned=len(rows),
            explanation=(
                f"Checked all {len(rows)} contracts for absence of '{f}'. Found {len(matched)}."
                + (f" Breakdown by basis: {breakdown}." if breakdown else "")
                + f" Determinacy for this field (on '{measured_on}'): {coverage:.0%}."
            ),
            latency_ms=(time.perf_counter() - t0) * 1000,
        )


class NumericEngine(Engine):
    """Typed magnitude comparison. Embeddings have no ordering over numbers."""
    name = "numeric"

    def run(self, plan: QueryPlan, store: StructuredStore, index: HybridIndex | None) -> Answer:
        t0 = time.perf_counter()
        rows = store.scan_all()
        f, op, v = plan.field, plan.operator, plan.value
        matched = [r for r in rows if _matches(r, f, op, v)]
        value = len(matched) if plan.aggregation == "count" else [r["contract_id"] for r in matched]
        return Answer(
            query=plan.query, query_type=plan.query_type, engine=self.name,
            value=value,
            items=[self._item(store, r, f, f"{f}={r.get(f)} {op} {v}") for r in matched[:50]],
            complete=True, scanned=len(rows),
            explanation=f"Applied {f} {op} {v} across all {len(rows)} contracts; {len(matched)} matched.",
            latency_ms=(time.perf_counter() - t0) * 1000,
        )


class TemporalEngine(Engine):
    """Date range predicates over typed dates."""
    name = "temporal"

    def run(self, plan: QueryPlan, store: StructuredStore, index: HybridIndex | None) -> Answer:
        t0 = time.perf_counter()
        rows = store.scan_all()
        f = plan.field or "expiry_date"
        lo, hi = plan.value if isinstance(plan.value, (tuple, list)) else (None, None)
        matched = [r for r in rows if r.get(f) and lo <= str(r[f]) <= hi]
        value = len(matched) if plan.aggregation == "count" else [r["contract_id"] for r in matched]
        return Answer(
            query=plan.query, query_type=plan.query_type, engine=self.name,
            value=value,
            items=[self._item(store, r, f, f"{f}={r.get(f)}") for r in matched[:50]],
            complete=True, scanned=len(rows),
            explanation=f"Filtered {f} to [{lo}, {hi}] over all {len(rows)} contracts; {len(matched)} matched.",
            latency_ms=(time.perf_counter() - t0) * 1000,
        )


class MultiHopEngine(Engine):
    """Traversal a single retrieval pass cannot perform.

    'Vendors on statements of work issued under Delaware-governed master
    agreements' requires: filter parents by law, collect their ids, then filter
    children whose parent pointer is in that set. Two dependent steps, where
    the second needs the output of the first.
    """
    name = "multihop"

    def run(self, plan: QueryPlan, store: StructuredStore, index: HybridIndex | None) -> Answer:
        t0 = time.perf_counter()
        rows = store.scan_all()
        hints = plan.secondary or {}

        parents = rows
        if "parent_governing_law" in hints:
            parents = [r for r in parents if r.get("governing_law") == hints["parent_governing_law"]]
        parent_ids = {r["contract_id"] for r in parents}

        children = [r for r in rows if r.get("amends_contract_id") in parent_ids]
        value = len(children) if plan.aggregation == "count" else [r["contract_id"] for r in children]

        return Answer(
            query=plan.query, query_type=plan.query_type, engine=self.name,
            value=value,
            items=[
                self._item(store, r, "amends_contract_id",
                           f"amends {r.get('amends_contract_id')} (vendor: {r.get('vendor')})")
                for r in children[:50]
            ],
            complete=True, scanned=len(rows),
            explanation=(
                f"Hop 1: {len(parent_ids)} parent contracts matched {hints or 'no filter'}. "
                f"Hop 2: {len(children)} contracts reference one of them."
            ),
            latency_ms=(time.perf_counter() - t0) * 1000,
        )


class EnumerateEngine(Engine):
    """Complete listing. Distinct from semantic search because completeness is guaranteed."""
    name = "enumerate"

    def run(self, plan: QueryPlan, store: StructuredStore, index: HybridIndex | None) -> Answer:
        t0 = time.perf_counter()
        rows = store.scan_all()
        f, op, v = plan.field, plan.operator, plan.value
        matched = [r for r in rows if _matches(r, f, op, v)]
        return Answer(
            query=plan.query, query_type=plan.query_type, engine=self.name,
            value=[r["contract_id"] for r in matched],
            items=[self._item(store, r, f, f"{f}={r.get(f)}") for r in matched[:100]],
            complete=True, scanned=len(rows),
            explanation=f"Complete listing over {len(rows)} contracts where {f} {op} {v}: {len(matched)} matched.",
            latency_ms=(time.perf_counter() - t0) * 1000,
        )


def _matches(row: dict[str, Any], field_name: str | None, op: str | None, value: Any) -> bool:
    if field_name is None:
        return True
    cur = row.get(field_name)
    if op in (None, "not_null"):
        return cur is not None and cur != 0 if field_name in BOOLEAN_FIELDS else cur is not None
    if op == "is_null":
        return cur is None
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
    return ENGINES[plan.query_type].run(plan, store, index)
