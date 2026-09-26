"""
Query router: natural language question -> typed execution plan.

The router does not answer anything. It decides *which engine can answer this
correctly*, which is the decision naive RAG never makes -- it sends every
question down the retrieval path regardless of whether retrieval is capable
of answering it.

A plan carries a field, an operator and a value, so routing is inspectable.
When the system gets an answer wrong you can see whether the router misread
the question or the engine mishandled a correct plan. That separation is
what makes the thing debuggable.

Rule based on purpose. The classification signals here (negation markers,
comparison operators, aggregation verbs) are lexically explicit in English,
so a rule layer is accurate, free, instant and auditable. An LLM classifier
is available behind the same interface for paraphrase-heavy production
traffic; it is not needed to demonstrate the thesis.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field as dc_field
from typing import Any

from app.models import QueryType


@dataclass
class QueryPlan:
    query: str
    query_type: QueryType
    field: str | None = None
    operator: str | None = None          # eq, ne, gt, gte, lt, lte, is_null, not_null, between, contains
    value: Any = None
    aggregation: str | None = None       # count, list, sum, avg, min, max
    secondary: dict[str, Any] = dc_field(default_factory=dict)
    signals: list[str] = dc_field(default_factory=list)
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "query_type": self.query_type.value,
            "field": self.field,
            "operator": self.operator,
            "value": self.value,
            "aggregation": self.aggregation,
            "secondary": self.secondary,
            "signals": self.signals,
            "confidence": self.confidence,
        }


# Natural language -> structured field. Ordered: longer phrases first so
# "liability cap" wins over a bare "liability".
FIELD_LEXICON: list[tuple[str, str]] = [
    (r"auto[\s\-]?renew\w*|automatic\w*\s+renew\w*|renew\s+automatically", "auto_renew"),
    (r"liability\s+cap|cap\s+on\s+liability|limitation\s+of\s+liability|liability\s+limit|capped\s+liability", "liability_cap_usd"),
    (r"liquidated\s+damages|late\s+penalt\w*|penalt\w*", "late_penalty_usd"),
    (r"insurance\s+certificate|certificate\s+of\s+insurance|insurance", "insurance_required"),
    (r"indemnif\w*", "indemnification"),
    (r"governing\s+law|governed\s+by|jurisdiction", "governing_law"),
    (r"termination\s+notice|notice\s+period|notice\s+to\s+terminate", "termination_notice_days"),
    (r"confidential\w*", "confidentiality_years"),
    (r"assign\w*", "assignment_allowed"),
    (r"expir\w*|expiration|end\s+date|terminat\w*\s+date", "expiry_date"),
    (r"effective\s+date|start\s+date|commenc\w*", "effective_date"),
    (r"term\s+length|contract\s+term|term\s+of\s+the\s+agreement", "term_months"),
    (r"vendor|supplier|counterparty", "vendor"),
    (r"agreement\s+type|type\s+of\s+agreement", "agreement_type"),
    (r"amend\w*|master\s+agreement|under\s+the\s+msa|parent\s+agreement", "amends_contract_id"),
]

BOOLEAN_FIELDS = {"auto_renew", "insurance_required", "indemnification", "assignment_allowed"}
NUMERIC_FIELDS = {
    "liability_cap_usd", "late_penalty_usd", "termination_notice_days",
    "term_months", "confidentiality_years", "insurance_min_usd",
}
DATE_FIELDS = {"expiry_date", "effective_date"}

NEGATION = re.compile(
    r"\b(no|not|without|lack(?:s|ing)?|missing|absent|omit\w*|fail\s+to|"
    r"do(?:es)?\s+n[o']t|don'?t|doesn'?t|never|neither|none)\b", re.I
)
AGGREGATION = re.compile(
    r"\b(how\s+many|count|number\s+of|total|sum|average|avg|mean|"
    r"what\s+percentage|what\s+share|how\s+much)\b", re.I
)
ENUMERATION = re.compile(r"\b(list|which|show\s+(?:me\s+)?all|every|name\s+all|identify\s+all|all\s+of)\b", re.I)

# Order is load bearing: inclusive forms must be tested before their strict
# counterparts, because "at or above" contains "above" and would otherwise be
# read as a strict greater-than, silently dropping every row sitting exactly on
# the boundary.
COMPARATORS: list[tuple[str, str]] = [
    (r"\b(?:at\s+least|at\s+or\s+above|at\s+or\s+over|or\s+above|or\s+more|"
     r"no\s+less\s+than|not\s+less\s+than|minimum\s+of)\b", "gte"),
    (r"\b(?:at\s+most|at\s+or\s+below|at\s+or\s+under|or\s+below|or\s+fewer|or\s+less|"
     r"no\s+more\s+than|not\s+more\s+than|maximum\s+of)\b", "lte"),
    (r"\b(?:greater\s+than|more\s+than|above|over|exceed(?:s|ing)?|in\s+excess\s+of|higher\s+than)\b", "gt"),
    (r"\b(?:less\s+than|under|below|beneath|lower\s+than|fewer\s+than)\b", "lt"),
    (r"\b(?:exactly|equal\s+to|equals)\b", "eq"),
]
MULTIHOP = re.compile(
    r"\b(amend\w*\s+(?:a|an|the)|issued\s+under|under\s+(?:a|an|the)\s+\w+\s+(?:agreement|msa)|"
    r"that\s+reference|referencing|whose\s+(?:master|parent))\b", re.I
)
TEMPORAL = re.compile(
    r"\b(expir\w*|before|after|between|during|within|as\s+of|current(?:ly)?|"
    r"q[1-4]\b|quarter|20\d{2}|next\s+\d+\s+(?:days|months)|this\s+(?:year|month)|"
    r"overdue|upcoming)\b", re.I
)

_MONEY = re.compile(r"\$\s?([\d,]+(?:\.\d+)?)\s*([KkMm])?\b")
_BARE_NUM = re.compile(r"\b(\d[\d,]*)\s*(?:days?|months?|years?)\b", re.I)
_QUARTER = re.compile(r"\bq([1-4])\s*(?:of\s*)?(20\d{2})\b", re.I)
_YEAR = re.compile(r"\b(20\d{2})\b")


def _parse_amount(q: str) -> int | None:
    m = _MONEY.search(q)
    if m:
        v = float(m.group(1).replace(",", ""))
        suf = (m.group(2) or "").upper()
        if suf == "K":
            v *= 1_000
        elif suf == "M":
            v *= 1_000_000
        return int(v)
    m = _BARE_NUM.search(q)
    if m:
        return int(m.group(1).replace(",", ""))
    return None


def _detect_field(q: str) -> tuple[str | None, str | None]:
    for pattern, fname in FIELD_LEXICON:
        m = re.search(pattern, q, re.I)
        if m:
            return fname, m.group(0)
    return None, None


class QueryRouter:
    """Classification order encodes precedence, and the order is load bearing.

    "How many contracts have no liability cap" is both an aggregation and an
    absence query. Absence must win the *filter* while aggregation wins the
    *output shape*, otherwise you count the wrong set. So absence is detected
    first and aggregation is recorded as the aggregation mode on top of it.
    """

    def route(self, query: str) -> QueryPlan:
        q = query.strip()
        signals: list[str] = []
        fname, matched = _detect_field(q)
        if matched:
            signals.append(f"field:{fname}<-'{matched}'")

        wants_count = bool(AGGREGATION.search(q))
        wants_list = bool(ENUMERATION.search(q))
        agg = "count" if wants_count else ("list" if wants_list else None)
        if wants_count:
            signals.append("aggregation:count")
        elif wants_list:
            signals.append("aggregation:list")

        # 1. multi-hop -- checked first because it implies a join no other engine does
        if MULTIHOP.search(q):
            signals.append("multihop")
            return QueryPlan(q, QueryType.MULTIHOP, field=fname, aggregation=agg or "list",
                             signals=signals, secondary=self._multihop_hints(q))

        # 2. absence -- must precede numeric/semantic; negation inverts the answer set
        neg = NEGATION.search(q)
        if neg and fname:
            signals.append(f"negation:'{neg.group(0)}'")
            return QueryPlan(q, QueryType.ABSENCE, field=fname, operator="is_null",
                             aggregation=agg or "list", signals=signals)

        # 3. numeric predicate
        amount = _parse_amount(q)
        for pattern, op in COMPARATORS:
            if re.search(pattern, q, re.I) and amount is not None and fname in NUMERIC_FIELDS:
                signals.append(f"comparator:{op} value:{amount}")
                return QueryPlan(q, QueryType.NUMERIC, field=fname, operator=op, value=amount,
                                 aggregation=agg or "list", signals=signals)

        # 4. temporal predicate
        if TEMPORAL.search(q) and (fname in DATE_FIELDS or fname is None):
            window = self._temporal_window(q)
            if window:
                signals.append(f"temporal:{window}")
                return QueryPlan(q, QueryType.TEMPORAL, field=fname or "expiry_date",
                                 operator="between", value=window,
                                 aggregation=agg or "list", signals=signals)

        # 5. aggregation over a known field (counts, sums, averages)
        if wants_count and fname:
            op, val = self._equality_predicate(q, fname)
            signals.append(f"aggregate:{op}={val}")
            return QueryPlan(q, QueryType.AGGREGATE, field=fname, operator=op, value=val,
                             aggregation=self._agg_kind(q), signals=signals)

        # 6. enumeration over a known field
        if wants_list and fname:
            op, val = self._equality_predicate(q, fname)
            signals.append(f"enumerate:{op}={val}")
            return QueryPlan(q, QueryType.ENUMERATE, field=fname, operator=op, value=val,
                             aggregation="list", signals=signals)

        # 7. fall through to retrieval -- the only category it is right for
        signals.append("fallback:semantic")
        return QueryPlan(q, QueryType.SEMANTIC, field=fname, aggregation=agg,
                         signals=signals, confidence=0.6 if fname else 0.4)

    # -- helpers ----------------------------------------------------------
    def _agg_kind(self, q: str) -> str:
        low = q.lower()
        if re.search(r"\baverage|avg|mean\b", low):
            return "avg"
        if re.search(r"\btotal|sum\b", low):
            return "sum"
        if re.search(r"\bpercentage|share|proportion\b", low):
            return "percentage"
        return "count"

    def _equality_predicate(self, q: str, fname: str) -> tuple[str, Any]:
        if fname in BOOLEAN_FIELDS:
            return "eq", True
        if fname == "governing_law":
            m = re.search(
                r"\b(Delaware|New\s+York|California|Texas|Illinois|Massachusetts)\b", q, re.I)
            if m:
                return "eq", m.group(1).title().replace("  ", " ")
            return "not_null", None
        if fname in NUMERIC_FIELDS:
            amt = _parse_amount(q)
            if amt is not None:
                return "eq", amt
            return "not_null", None
        return "not_null", None

    def _temporal_window(self, q: str) -> tuple[str, str] | None:
        m = _QUARTER.search(q)
        if m:
            quarter, year = int(m.group(1)), int(m.group(2))
            start_month = 3 * (quarter - 1) + 1
            end_month = start_month + 2
            last_day = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][end_month - 1]
            if end_month == 2 and year % 4 == 0:
                last_day = 29
            return (f"{year}-{start_month:02d}-01", f"{year}-{end_month:02d}-{last_day:02d}")
        m = _YEAR.search(q)
        if m:
            y = int(m.group(1))
            return (f"{y}-01-01", f"{y}-12-31")
        return None

    def _multihop_hints(self, q: str) -> dict[str, Any]:
        hints: dict[str, Any] = {}
        m = re.search(r"\b(Delaware|New\s+York|California|Texas|Illinois|Massachusetts)\b", q, re.I)
        if m:
            hints["parent_governing_law"] = m.group(1).title().replace("  ", " ")
        m2 = re.search(r"\b(master\s+services\s+agreement|msa|software\s+license|supply\s+agreement)\b", q, re.I)
        if m2:
            hints["parent_type_hint"] = m2.group(1).lower()
        return hints
