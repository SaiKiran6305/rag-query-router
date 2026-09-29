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
comparison operators, aggregation verbs, date expressions) are lexically
explicit in English, so a rule layer is accurate, free, instant and auditable.
Rules miss paraphrases they were not written for; `app/router/llm.py` adds an
optional LLM tier behind the same interface, consulted only when the rules fall
through with unresolved structural intent. Field knowledge (names, kinds,
synonyms) lives in `app/schema.py`, not here.

Every plan now carries a confidence and, for the semantic fallback, the reason
it fell through. Semantic has to *earn* its route with positive evidence (an
explanation verb, a clause named); falling through with nothing is low
confidence, and falling through with a count or a negation nobody could
resolve is flagged, because retrieval cannot answer those correctly.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field as dc_field
from datetime import date, timedelta
from typing import Any

from app.models import QueryType
from app.schema import (  # re-exported: engines, baseline and tests import them from here
    AGREEMENT_TYPES, BOOLEAN_FIELDS, DATE_FIELDS, FIELD_LEXICON, NUMERIC_FIELDS,
    NUMERIC_TWINS, STATE_PATTERN,
)

__all__ = ["QueryPlan", "QueryRouter", "BOOLEAN_FIELDS", "NUMERIC_FIELDS", "DATE_FIELDS",
           "FIELD_LEXICON"]


@dataclass
class QueryPlan:
    query: str
    query_type: QueryType
    field: str | None = None
    operator: str | None = None          # eq, ne, gt, gte, lt, lte, is_null, not_null, between, contains, in
    value: Any = None
    aggregation: str | None = None       # count, list, sum, avg, percentage
    secondary: dict[str, Any] = dc_field(default_factory=dict)
    signals: list[str] = dc_field(default_factory=list)
    confidence: float = 1.0
    filters: dict[str, Any] = dc_field(default_factory=dict)   # scope, e.g. {"governing_law": "Texas"}
    assumptions: list[str] = dc_field(default_factory=list)    # defaults the router filled in
    unresolved: list[str] = dc_field(default_factory=list)     # structural intent it could not map
    tier: str = "rules"                  # rules | llm
    abstain_reason: str | None = None    # set when the plan itself says "cannot answer"

    @property
    def needs_escalation(self) -> bool:
        """True when a smarter router should take a second look.

        Only a semantic fallback qualifies: a structural plan came from explicit
        signals and is either right or visibly wrong in /api/plan.
        """
        return self.query_type == QueryType.SEMANTIC and (bool(self.unresolved) or self.confidence < 0.5)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "query_type": self.query_type.value,
            "field": self.field,
            "operator": self.operator,
            "value": self.value,
            "aggregation": self.aggregation,
            "filters": self.filters,
            "secondary": self.secondary,
            "signals": self.signals,
            "assumptions": self.assumptions,
            "unresolved": self.unresolved,
            "confidence": self.confidence,
            "tier": self.tier,
            "abstain_reason": self.abstain_reason,
        }


# ---------------------------------------------------------------------------
# Signal patterns
# ---------------------------------------------------------------------------

NEGATION = re.compile(
    r"\b(no|not|without|lack(?:s|ing)?|missing|absent|omit\w*|fail\s+to|"
    r"do(?:es)?\s+n[o']t|don'?t|doesn'?t|never|neither|none)\b", re.I
)
# Absence stated without a negation word.
IMPLICIT_ABSENCE = re.compile(
    r"\b(uncapped|unlimited\s+liability|liability\s+(?:is|are|was|be)\s+(?:unlimited|uncapped))\b", re.I
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
    (r"\b(?:at\s+least|at\s+or\s+above|at\s+or\s+over|or\s+above|or\s+more|or\s+longer|"
     r"no\s+less\s+than|not\s+less\s+than|no\s+fewer\s+than|minimum\s+of)\b", "gte"),
    (r"\b(?:at\s+most|at\s+or\s+below|at\s+or\s+under|or\s+below|or\s+fewer|or\s+less|or\s+shorter|"
     r"no\s+more\s+than|not\s+more\s+than|maximum\s+of)\b", "lte"),
    (r"\b(?:greater\s+than|more\s+than|longer\s+than|above|over|exceed(?:s|ing)?|in\s+excess\s+of|"
     r"higher\s+than)\b", "gt"),
    (r"\b(?:less\s+than|shorter\s+than|under|below|beneath|lower\s+than|fewer\s+than)\b", "lt"),
    (r"\b(?:exactly|equal\s+to|equals)\b", "eq"),
]
# Comparator phrases that contain a negation word. Removed before negation is
# tested, otherwise "penalties no less than $50,000" is read as an absence query.
_NEGATED_COMPARATORS = re.compile(
    r"\b(?:no|not)\s+(?:less|more|fewer|later|earlier)\s+than\b", re.I)

MULTIHOP = re.compile(
    r"\b(amend\w*\s+(?:a|an|the)|issued\s+under|under\s+(?:a|an|the)\s+\w+\s+(?:agreement|msa)|"
    r"that\s+reference|referencing|whose\s+(?:master|parent))\b", re.I
)

# Positive evidence that retrieval is the right tool: the user wants to read or
# understand text, not compute over it.
EXPLAIN = re.compile(
    r"\b(explain|describe|summari[sz]e|outline|walk\s+me\s+through|what\s+(?:does|do)\b|"
    r"how\s+(?:does|do|is|are)\b|provisions?|clauses?|language|obligations?|requirements?|"
    r"section|wording)\b", re.I
)

_MONEY = re.compile(r"\$\s?([\d,]+(?:\.\d+)?)\s*([KkMm])?\b")
_BARE_NUM = re.compile(r"\b(\d[\d,]*)\s*(?:days?|months?|years?)\b", re.I)

# --- date expressions ------------------------------------------------------

_MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august",
           "september", "october", "november", "december"]
_MONTH_RE = r"(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?|" \
            r"sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
_ORDINAL_QUARTER = {"first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3,
                    "fourth": 4, "4th": 4, "last": 4}

_DATE_EXPRS: list[tuple[str, re.Pattern]] = [
    ("full", re.compile(rf"\b{_MONTH_RE}\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(20\d{{2}})\b", re.I)),
    ("iso", re.compile(r"\b(20\d{2})-(\d{2})-(\d{2})\b")),
    ("slash", re.compile(r"\b(\d{1,2})/(\d{1,2})/(20\d{2})\b")),
    ("quarter", re.compile(r"\bq([1-4])\s*(?:of\s*)?(20\d{2})\b", re.I)),
    ("ordinal_quarter", re.compile(
        r"\b(first|1st|second|2nd|third|3rd|fourth|4th|last)\s+quarter\s+(?:of\s+)?(20\d{2})\b", re.I)),
    ("month", re.compile(rf"\b{_MONTH_RE}\.?\s+(?:of\s+)?(20\d{{2}})\b", re.I)),
    ("year", re.compile(r"\b(20\d{2})\b")),
]
_RELATIVE = [
    ("next_n", re.compile(r"\b(?:in\s+|within\s+)?(?:the\s+)?next\s+(\d+)\s+(day|week|month|year)s?\b", re.I)),
    ("within_n", re.compile(r"\bwithin\s+(\d+)\s+(day|week|month|year)s?\b", re.I)),
    ("this_year", re.compile(r"\bthis\s+year\b", re.I)),
    ("next_year", re.compile(r"\bnext\s+year\b", re.I)),
    ("last_year", re.compile(r"\blast\s+year\b", re.I)),
    ("this_month", re.compile(r"\bthis\s+month\b", re.I)),
    ("expired", re.compile(r"\b(?:already\s+expired|have\s+expired|has\s+expired|overdue|lapsed)\b", re.I)),
]

# Cue immediately before a date expression -> operator. Inclusive before strict,
# for the same reason as COMPARATORS: "on or after" contains "after".
_DATE_CUES: list[tuple[str, str]] = [
    (r"on\s+or\s+before|no\s+later\s+than|by|until|through|up\s+to", "lte_end"),
    (r"on\s+or\s+after|no\s+earlier\s+than|since|from|starting(?:\s+in)?|beginning(?:\s+in)?", "gte_start"),
    (r"before|prior\s+to|earlier\s+than", "lt_start"),
    (r"after|later\s+than|following", "gt_end"),
]


def _month_index(token: str) -> int:
    t = token.lower().rstrip(".")
    for i, m in enumerate(_MONTHS):
        if m.startswith(t[:3]):
            return i + 1
    raise ValueError(token)


def _last_day(y: int, m: int) -> int:
    nxt = date(y + (m == 12), m % 12 + 1, 1)
    return (nxt - timedelta(days=1)).day


def _add_months(d: date, n: int) -> date:
    m = d.month - 1 + n
    y, m = d.year + m // 12, m % 12 + 1
    return date(y, m, min(d.day, _last_day(y, m)))


def _interval(kind: str, m: re.Match) -> tuple[date, date]:
    if kind == "full":
        d = date(int(m.group(3)), _month_index(m.group(1)), int(m.group(2)))
        return d, d
    if kind == "iso":
        d = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        return d, d
    if kind == "slash":
        d = date(int(m.group(3)), int(m.group(1)), int(m.group(2)))
        return d, d
    if kind in ("quarter", "ordinal_quarter"):
        q = int(m.group(1)) if kind == "quarter" else _ORDINAL_QUARTER[m.group(1).lower()]
        y = int(m.group(2))
        sm = 3 * (q - 1) + 1
        return date(y, sm, 1), date(y, sm + 2, _last_day(y, sm + 2))
    if kind == "month":
        y, mo = int(m.group(2)), _month_index(m.group(1))
        return date(y, mo, 1), date(y, mo, _last_day(y, mo))
    y = int(m.group(1))
    return date(y, 1, 1), date(y, 12, 31)


def find_dates(q: str) -> list[tuple[int, int, date, date, str]]:
    """Absolute date expressions in the question: (start_pos, end_pos, lo, hi, kind).

    Patterns are tried most specific first and a matched span is consumed, so
    "March 1, 2027" is read once as a day and not again as a month and a year.
    """
    taken: list[tuple[int, int]] = []
    out = []
    for kind, pat in _DATE_EXPRS:
        for m in pat.finditer(q):
            if any(m.start() < e and s < m.end() for s, e in taken):
                continue
            try:
                lo, hi = _interval(kind, m)
            except (ValueError, KeyError):
                continue
            taken.append((m.start(), m.end()))
            out.append((m.start(), m.end(), lo, hi, kind))
    return sorted(out)


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


def _comparator(q: str) -> str | None:
    for pattern, op in COMPARATORS:
        if re.search(pattern, q, re.I):
            return op
    return None


def _detect_field(q: str) -> tuple[str | None, str | None]:
    for pattern, fname in FIELD_LEXICON:
        m = re.search(pattern, q, re.I)
        if m:
            return fname, m.group(0)
    # A bare state name is a governing-law question ("contracts under Texas law").
    m = re.search(STATE_PATTERN, q, re.I)
    if m:
        return "governing_law", m.group(0)
    return None, None


def _state(q: str) -> str | None:
    m = re.search(STATE_PATTERN, q, re.I)
    return re.sub(r"\s+", " ", m.group(1)).title() if m else None


def _agreement_types(q: str) -> list[tuple[int, str]]:
    found = []
    for name, pat in AGREEMENT_TYPES.items():
        for m in re.finditer(pat, q, re.I):
            found.append((m.start(), name))
    return sorted(found)


class QueryRouter:
    """Classification order encodes precedence, and the order is load bearing.

    "How many contracts have no liability cap" is both an aggregation and an
    absence query. Absence must win the *filter* while aggregation wins the
    *output shape*, otherwise you count the wrong set. So absence is detected
    first and aggregation is recorded as the aggregation mode on top of it.

    `today` is injectable so relative dates ("next 90 days") are testable.
    """

    def __init__(self, today: date | None = None):
        self._today = today

    @property
    def today(self) -> date:
        return self._today or date.today()

    def route(self, query: str) -> QueryPlan:
        q = query.strip()
        signals: list[str] = []
        fname, matched = _detect_field(q)
        if matched:
            signals.append(f"field:{fname}<-'{matched}'")

        amount = _parse_amount(q)
        comparator = _comparator(q)

        # "insurance of at least $2M" is about the amount, not the yes/no field.
        if fname in NUMERIC_TWINS and amount is not None and comparator:
            twin = NUMERIC_TWINS[fname]
            signals.append(f"field:{fname}->{twin} (amount given)")
            fname = twin

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

        filters = self._scope_filters(q, fname)
        if filters:
            signals.append(f"scope:{filters}")

        # 2. absence -- must precede numeric/semantic; negation inverts the answer set
        neg = NEGATION.search(_NEGATED_COMPARATORS.sub(" ", q))
        implicit = IMPLICIT_ABSENCE.search(q)
        if fname and (neg or implicit):
            signals.append(f"negation:'{(neg or implicit).group(0)}'")
            return QueryPlan(q, QueryType.ABSENCE, field=fname, operator="is_null",
                             aggregation=agg or "list", signals=signals, filters=filters)

        # 3. numeric predicate
        if comparator and amount is not None and fname in NUMERIC_FIELDS:
            signals.append(f"comparator:{comparator} value:{amount}")
            return QueryPlan(q, QueryType.NUMERIC, field=fname, operator=comparator, value=amount,
                             aggregation=agg or "list", signals=signals, filters=filters)

        # 4. temporal predicate
        if fname in DATE_FIELDS or fname is None:
            pred = self._temporal_predicate(q)
            if pred:
                op, value, desc = pred
                assumptions = []
                if fname is None:
                    assumptions.append("no date field named in the question; assumed expiry_date")
                signals.append(f"temporal:{desc}")
                return QueryPlan(q, QueryType.TEMPORAL, field=fname or "expiry_date",
                                 operator=op, value=value, aggregation=agg or "list",
                                 signals=signals, filters=filters, assumptions=assumptions)

        # 5. aggregation over a known field (counts, sums, averages)
        if wants_count and fname:
            op, val = self._equality_predicate(q, fname)
            signals.append(f"aggregate:{op}={val}")
            return QueryPlan(q, QueryType.AGGREGATE, field=fname, operator=op, value=val,
                             aggregation=self._agg_kind(q), signals=signals, filters=filters)

        # 6. enumeration over a known field
        if wants_list and fname:
            op, val = self._equality_predicate(q, fname)
            signals.append(f"enumerate:{op}={val}")
            return QueryPlan(q, QueryType.ENUMERATE, field=fname, operator=op, value=val,
                             aggregation="list", signals=signals, filters=filters)

        # 7. fall through to retrieval -- the only category it is right for.
        # Semantic has to earn the route: an explanation verb or a named clause
        # is positive evidence; falling through with neither is low confidence.
        explain = bool(EXPLAIN.search(q))
        confidence = 0.9 if (explain and fname) else 0.7 if explain else 0.6 if fname else 0.3
        unresolved = []
        if wants_count:
            unresolved.append("count")
        if neg or implicit:
            unresolved.append("negation")
        if comparator and amount is not None:
            unresolved.append("comparison")
        if find_dates(q):
            unresolved.append("date")
        signals.append("fallback:semantic" + (f" unresolved:{','.join(unresolved)}" if unresolved else ""))
        return QueryPlan(q, QueryType.SEMANTIC, field=fname, aggregation=agg, signals=signals,
                         confidence=confidence, unresolved=unresolved)

    # -- helpers ----------------------------------------------------------
    def _agg_kind(self, q: str) -> str:
        low = q.lower()
        # Count words win: "total number of contracts" is a count, not a sum.
        if re.search(r"\b(?:how\s+many|number\s+of|count)\b", low):
            return "count"
        if re.search(r"\b(?:average|avg|mean)\b", low):
            return "avg"
        if re.search(r"\b(?:percentage|percent|share|proportion)\b", low):
            return "percentage"
        if re.search(r"\b(?:total|sum)\b", low):
            return "sum"
        return "count"

    def _equality_predicate(self, q: str, fname: str) -> tuple[str, Any]:
        if fname in BOOLEAN_FIELDS:
            return "eq", True
        if fname == "governing_law":
            state = _state(q)
            return ("eq", state) if state else ("not_null", None)
        if fname == "agreement_type":
            types = _agreement_types(q)
            return ("eq", types[0][1]) if types else ("not_null", None)
        if fname in NUMERIC_FIELDS:
            amt = _parse_amount(q)
            if amt is not None:
                return "eq", amt
            return "not_null", None
        return "not_null", None

    def _scope_filters(self, q: str, primary: str | None) -> dict[str, Any]:
        """Constraints on a *different* field than the one being asked about.

        "How many Texas contracts auto-renew?" asks about auto_renew within the
        Texas contracts. Without this the state was silently dropped and the
        answer covered every contract -- a wrong answer labelled complete.
        """
        out: dict[str, Any] = {}
        if primary != "governing_law":
            state = _state(q)
            if state:
                out["governing_law"] = state
        if primary != "agreement_type":
            types = _agreement_types(q)
            if types:
                out["agreement_type"] = types[0][1]
        return out

    def _temporal_predicate(self, q: str) -> tuple[str, Any, str] | None:
        """Date expression plus the cue before it -> (operator, value, description)."""
        today = self.today
        for kind, pat in _RELATIVE:
            m = pat.search(q)
            if not m:
                continue
            if kind in ("next_n", "within_n"):
                n, unit = int(m.group(1)), m.group(2).lower()
                if unit == "day":
                    hi = today + timedelta(days=n)
                elif unit == "week":
                    hi = today + timedelta(weeks=n)
                elif unit == "month":
                    hi = _add_months(today, n)
                else:
                    hi = _add_months(today, 12 * n)
                return "between", (today.isoformat(), hi.isoformat()), f"next {n} {unit}s from {today}"
            if kind == "expired":
                return "lt", today.isoformat(), f"before today ({today})"
            if kind == "this_month":
                lo = today.replace(day=1)
                hi = today.replace(day=_last_day(today.year, today.month))
                return "between", (lo.isoformat(), hi.isoformat()), "this month"
            y = today.year + {"this_year": 0, "next_year": 1, "last_year": -1}[kind]
            return "between", (f"{y}-01-01", f"{y}-12-31"), kind.replace("_", " ")

        dates = find_dates(q)
        if not dates:
            return None
        start, _, lo, hi, _ = dates[0]

        if len(dates) >= 2 and re.search(r"\bbetween\b", q[:start], re.I):
            end_hi = dates[1][3]
            return "between", (lo.isoformat(), end_hi.isoformat()), f"between {lo} and {end_hi}"

        prefix = q[:start].rstrip().lower()
        prefix = re.sub(r"\b(?:the|of)\s*$", "", prefix).rstrip()
        for cue, how in _DATE_CUES:
            if re.search(rf"\b(?:{cue})$", prefix):
                if how == "lte_end":
                    return "lte", hi.isoformat(), f"on or before {hi}"
                if how == "gte_start":
                    return "gte", lo.isoformat(), f"on or after {lo}"
                if how == "lt_start":
                    return "lt", lo.isoformat(), f"before {lo}"
                return "gt", hi.isoformat(), f"after {hi}"
        return "between", (lo.isoformat(), hi.isoformat()), f"{lo} to {hi}"

    def _multihop_hints(self, q: str) -> dict[str, Any]:
        """Parent and child constraints for a two-hop query.

        Agreement types mentioned after the hop cue ("issued under", "referencing")
        describe the parent; types mentioned before it describe the child.
        """
        hints: dict[str, Any] = {}
        state = _state(q)
        if state:
            hints["parent_governing_law"] = state
        cue = MULTIHOP.search(q)
        cue_pos = cue.start() if cue else len(q)
        for pos, name in _agreement_types(q):
            key = "parent_type" if pos >= cue_pos else "child_type"
            hints.setdefault(key, name)
        return hints
