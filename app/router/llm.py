"""
LLM router tier: a second opinion for the questions the rules could not place.

Design
------
Tiered, not replaced. The rule router answers first -- it is free, instant and
auditable, and on the explicit phrasings most traffic uses it is also correct.
The LLM is consulted only when the rules *fall through* to the semantic engine
with a reason to doubt that fall-through: a count, negation, comparison or date
the rules saw but could not map to a field, or no evidence at all that this is
a clause-reading question (QueryPlan.needs_escalation). Cost therefore scales
with how unusual the traffic is, not with how much of it there is.

Propose, then verify. The LLM never writes SQL and never answers. It proposes
a QueryPlan as JSON, and `validate()` checks every part of that proposal
against the schema registry before anything executes: the field must exist,
the operator must be legal for the field's type, values are coerced and
range-checked, closed vocabularies (states, agreement types) must match. A
hallucinated column ("employee_count") is rejected here, not discovered as a
wrong answer later. The LLM may also say the question is outside the schema,
in which case the system declines instead of guessing.

What this tier cannot fix: a *confident* rule misroute. If the rules produce a
structural plan, the LLM is never asked. See README, "Robustness".

Enabled with ROUTER=hybrid and OPENAI_API_KEY. `complete_json` is injectable so
the tier is fully testable without network access or credentials.
"""

from __future__ import annotations

import json
import os
from datetime import date
from typing import Any, Callable

from app.models import QueryType
from app.router.classify import QueryPlan, QueryRouter
from app.schema import (
    AGREEMENT_TYPES, BOOL, BY_NAME, DATE, GOVERNING_LAWS, MONEY, NUMBER, OPERATORS_BY_KIND,
    SCOPE_FIELDS, catalog,
)

CompleteJSON = Callable[[str, str], dict]

# Fields a question may target. Basis columns are internal bookkeeping.
QUERYABLE = {name for name, f in BY_NAME.items() if f.synonyms}
AGGREGATIONS = {"count", "list", "sum", "avg", "percentage"}
_VALUELESS_OPS = {"is_null", "not_null"}

SYSTEM_PROMPT = """You convert questions about a corpus of commercial contracts into a query plan.
You do not answer the question. You only produce the plan, as a single JSON object.

Query types:
- aggregate: count, sum, average or percentage over contracts matching a predicate
- absence:   contracts where a field is missing or false ("no cap", "not assignable")
- numeric:   contracts where a money/number field compares to a value
- temporal:  contracts where a date field is before/after/between dates
- enumerate: complete list of contracts matching a predicate
- multihop:  contracts issued under a parent agreement that matches a condition
- semantic:  the user wants to read or understand clause text, not compute anything

Fields (the ONLY fields that exist):
{catalog}

Output JSON with these keys:
  "answerable": true | false
  "reason": short string (required when answerable is false)
  "query_type": one of the types above
  "field": a field name from the list, or null for semantic/multihop
  "operator": eq | gt | gte | lt | lte | between | is_null | not_null | contains
  "value": number, ISO date "YYYY-MM-DD", string, true/false, or a 2-item list for between
  "aggregation": count | list | sum | avg | percentage
  "filters": optional object restricting scope, keys among {scope}
  "parent_governing_law", "parent_type", "child_type": multihop only

Rules:
- If the question needs information that is not in the fields above, return
  {{"answerable": false, "reason": "..."}}. Never invent a field.
- Money is in whole US dollars. Dates are ISO. Use the exact allowed values for
  governing_law and agreement_type.
"""


class LLMPlanner:
    """Proposes plans with an LLM; every proposal goes through validate()."""

    def __init__(self, complete_json: CompleteJSON | None = None, model: str | None = None):
        self.model = model or os.environ.get("ROUTER_LLM_MODEL", "gpt-4o-mini")
        self._complete = complete_json or self._openai
        self.calls = 0
        self.rejections = 0
        self._cache: dict[str, dict] = {}

    def _openai(self, system: str, user: str) -> dict:  # pragma: no cover - needs network
        from openai import OpenAI

        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        resp = client.chat.completions.create(
            model=self.model,
            response_format={"type": "json_object"},
            temperature=0,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return json.loads(resp.choices[0].message.content or "{}")

    def propose(self, query: str) -> dict:
        key = query.strip().lower()
        if key not in self._cache:
            self.calls += 1
            system = SYSTEM_PROMPT.format(catalog=catalog(), scope=", ".join(SCOPE_FIELDS))
            self._cache[key] = self._complete(system, query)
        return self._cache[key]

    def plan(self, query: str) -> tuple[QueryPlan | None, list[str]]:
        try:
            data = self.propose(query)
        except Exception as e:  # network, auth, malformed JSON
            self.rejections += 1
            return None, [f"llm call failed: {type(e).__name__}"]
        plan, errors = validate(query, data)
        if plan is None:
            self.rejections += 1
        else:
            plan.signals.append(f"model:{self.model}")
        return plan, errors


def _coerce_value(kind: str, field: str, value: Any, errors: list[str]) -> Any:
    if kind in (MONEY, NUMBER):
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            errors.append(f"value for {field} must be a number")
            return None
        try:
            return int(float(str(value).replace(",", "").replace("$", "")))
        except ValueError:
            errors.append(f"value for {field} must be a number, got {value!r}")
            return None
    if kind == DATE:
        try:
            return date.fromisoformat(str(value)).isoformat()
        except ValueError:
            errors.append(f"value for {field} must be an ISO date, got {value!r}")
            return None
    if kind == BOOL:
        if isinstance(value, bool):
            return value
        if str(value).lower() in ("true", "false"):
            return str(value).lower() == "true"
        errors.append(f"value for {field} must be true/false")
        return None
    spec = BY_NAME[field]
    if spec.values:
        for allowed in spec.values:
            if str(value).strip().lower() == allowed.lower():
                return allowed
        errors.append(f"{value!r} is not an allowed {field} (allowed: {', '.join(spec.values)})")
        return None
    return str(value)


def _closed(value: Any, allowed: list[str] | tuple[str, ...], what: str, errors: list[str]) -> str | None:
    for a in allowed:
        if str(value).strip().lower() == a.lower():
            return a
    errors.append(f"{value!r} is not a known {what}")
    return None


def validate(query: str, data: Any) -> tuple[QueryPlan | None, list[str]]:
    """Check an LLM proposal against the schema. Returns (plan, errors)."""
    errors: list[str] = []
    if not isinstance(data, dict):
        return None, ["proposal is not a JSON object"]

    if data.get("answerable") is False:
        reason = str(data.get("reason") or "the question needs information the contracts do not record")
        return QueryPlan(query, QueryType.SEMANTIC, signals=["tier:llm", "llm:unanswerable"],
                         tier="llm", confidence=0.8, abstain_reason=reason), []

    try:
        qtype = QueryType(str(data.get("query_type", "")).lower())
    except ValueError:
        return None, [f"unknown query_type {data.get('query_type')!r}"]

    aggregation = data.get("aggregation")
    if aggregation is not None and aggregation not in AGGREGATIONS:
        errors.append(f"unknown aggregation {aggregation!r}")

    filters: dict[str, Any] = {}
    for k, v in (data.get("filters") or {}).items():
        if k not in SCOPE_FIELDS:
            errors.append(f"cannot filter on {k!r}")
            continue
        allowed = GOVERNING_LAWS if k == "governing_law" else list(AGREEMENT_TYPES)
        ok = _closed(v, allowed, k, errors)
        if ok:
            filters[k] = ok

    signals = ["tier:llm"]

    if qtype == QueryType.MULTIHOP:
        hints: dict[str, Any] = {}
        if data.get("parent_governing_law"):
            law = _closed(data["parent_governing_law"], GOVERNING_LAWS, "governing law", errors)
            if law:
                hints["parent_governing_law"] = law
        for key in ("parent_type", "child_type"):
            if data.get(key):
                t = _closed(data[key], list(AGREEMENT_TYPES), "agreement type", errors)
                if t:
                    hints[key] = t
        if errors:
            return None, errors
        return QueryPlan(query, qtype, aggregation=aggregation or "list", secondary=hints,
                         filters=filters, signals=signals, tier="llm", confidence=0.8), []

    field = data.get("field")
    if qtype == QueryType.SEMANTIC:
        if field is not None and field not in QUERYABLE:
            errors.append(f"unknown field {field!r}")
        if errors:
            return None, errors
        return QueryPlan(query, qtype, field=field, aggregation=aggregation, signals=signals,
                         tier="llm", confidence=0.8), []

    if field not in QUERYABLE:
        return None, [f"unknown field {field!r}"]
    kind = BY_NAME[field].kind
    op = data.get("operator")
    value = data.get("value")

    if qtype == QueryType.ABSENCE:
        op, value = "is_null", None
    elif qtype == QueryType.NUMERIC and kind not in (MONEY, NUMBER):
        errors.append(f"numeric query on non-numeric field {field}")
    elif qtype == QueryType.TEMPORAL and kind != DATE:
        errors.append(f"temporal query on non-date field {field}")

    if op is None:
        op = "eq" if kind == BOOL else "not_null"
        value = True if kind == BOOL else None
    if op not in OPERATORS_BY_KIND[kind]:
        errors.append(f"operator {op!r} is not valid for {kind} field {field}")
    elif op in _VALUELESS_OPS:
        value = None
    elif op == "between":
        if not (isinstance(value, (list, tuple)) and len(value) == 2):
            errors.append("between needs a 2-item value")
        else:
            value = tuple(_coerce_value(kind, field, v, errors) for v in value)
    else:
        value = _coerce_value(kind, field, value, errors)

    if aggregation in ("sum", "avg") and kind not in (MONEY, NUMBER):
        errors.append(f"cannot {aggregation} a {kind} field")

    if errors:
        return None, errors
    if aggregation is None:
        aggregation = "count" if qtype == QueryType.AGGREGATE else "list"
    return QueryPlan(query, qtype, field=field, operator=op, value=value, aggregation=aggregation,
                     filters=filters, signals=signals, tier="llm", confidence=0.8), []


class HybridRouter:
    """Rules first; the LLM only for fall-throughs the rules flagged as doubtful."""

    def __init__(self, rules: QueryRouter | None = None, planner: LLMPlanner | None = None):
        self.rules = rules or QueryRouter()
        self.planner = planner

    @property
    def kind(self) -> str:
        return "hybrid" if self.planner else "rules"

    def route(self, query: str) -> QueryPlan:
        plan = self.rules.route(query)
        if self.planner is None or not plan.needs_escalation:
            return plan
        proposed, errors = self.planner.plan(query)
        if proposed is not None:
            proposed.signals.insert(0, f"escalated from rules ({', '.join(plan.unresolved) or 'low confidence'})")
            return proposed
        plan.signals.append("llm_rejected: " + "; ".join(errors))
        return plan


def build_router(kind: str | None = None, planner: LLMPlanner | None = None) -> HybridRouter:
    """ROUTER=rules (default) or ROUTER=hybrid (needs OPENAI_API_KEY)."""
    kind = (kind or os.environ.get("ROUTER", "rules")).lower()
    if kind == "hybrid":
        if planner is None and not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("ROUTER=hybrid needs OPENAI_API_KEY (or pass a planner)")
        return HybridRouter(planner=planner or LLMPlanner())
    return HybridRouter()
