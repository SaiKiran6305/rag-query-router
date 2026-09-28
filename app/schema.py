"""
Schema registry: the single source of truth for what the system knows.

Before this module the same facts about each field lived in four places -- the
SQL DDL, the router's lexicon, the router's type sets (BOOLEAN_FIELDS, ...), and
the LLM extractor's field list -- and nothing kept them in sync. Now each field
is declared once, with:

  kind         how it is typed and which operators are legal on it
  description  one line, used verbatim in the LLM router's schema catalog
  synonyms     regex fragments the rule router uses to recognise the field
  basis        companion column recording *why* a value is absent, if any

Everything else is derived from this list: the rule lexicon, the type sets, the
column whitelist the SQL compiler enforces, and the catalog the LLM planner is
shown and validated against. Extending the system to a new field -- or, later,
a new table -- starts here.

Lexicon order is load bearing. `FIELDS` is scanned top to bottom and the first
field whose synonym matches wins, so a more specific phrase must appear before
any field whose synonym it contains (the same superstring rule DESIGN_NOTES
section 2 describes for comparators).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

BOOL, MONEY, NUMBER, DATE, TEXT, ID = "bool", "money", "number", "date", "text", "id"

# Operators legal per kind. The SQL compiler and the LLM plan validator both
# enforce this, so a plan can never ask for "liability_cap_usd contains 'x'".
OPERATORS_BY_KIND: dict[str, set[str]] = {
    BOOL: {"eq", "is_null", "not_null"},
    MONEY: {"eq", "ne", "gt", "gte", "lt", "lte", "between", "is_null", "not_null"},
    NUMBER: {"eq", "ne", "gt", "gte", "lt", "lte", "between", "is_null", "not_null"},
    DATE: {"eq", "gt", "gte", "lt", "lte", "between", "is_null", "not_null"},
    TEXT: {"eq", "ne", "contains", "is_null", "not_null", "in"},
    ID: {"eq", "in", "is_null", "not_null"},
}

GOVERNING_LAWS = ["Delaware", "New York", "California", "Texas", "Illinois", "Massachusetts"]

# Canonical agreement types, with the phrases people use for them.
AGREEMENT_TYPES: dict[str, str] = {
    "Master Services Agreement": r"master\s+services?\s+agreements?|\bmsas?\b|master\s+agreements?",
    "Software License Agreement": r"software\s+licen[cs]e(?:\s+agreements?)?",
    "Supply Agreement": r"supply\s+agreements?",
    "Statement of Work": r"statements?\s+of\s+work|\bsows?\b",
    "Maintenance and Support Agreement": r"maintenance\s+and\s+support(?:\s+agreements?)?",
    "Professional Services Agreement": r"professional\s+services(?:\s+agreements?)?",
}

STATE_PATTERN = r"\b(Delaware|New\s+York|California|Texas|Illinois|Massachusetts)\b"


@dataclass(frozen=True)
class FieldSpec:
    name: str
    kind: str
    description: str
    synonyms: tuple[str, ...] = ()
    basis: str | None = None           # companion "why is this empty" column
    values: tuple[str, ...] = ()       # closed vocabulary, if any

    @property
    def pattern(self) -> str | None:
        return "|".join(self.synonyms) if self.synonyms else None


# Ordered for lexicon precedence -- see module docstring.
FIELDS: list[FieldSpec] = [
    FieldSpec("auto_renew", BOOL, "whether the contract renews automatically at the end of its term",
              (r"auto[\s\-]?renew\w*", r"automatic\w*\s+renew\w*", r"renew\w*\s+automatically",
               r"renew\w*\s+on\s+(?:their|its)\s+own", r"self[\s\-]?renew\w*", r"evergreen")),
    FieldSpec("liability_cap_usd", MONEY,
              "maximum aggregate liability in USD; null when there is no cap (see liability_cap_basis)",
              (r"liability\s+cap", r"cap\s+on\s+liability", r"limitation\s+of\s+liability",
               r"liability\s+limit", r"capped\s+liability", r"\buncapped\b", r"unlimited\s+liability",
               r"liability\s+(?:is|are|was|be)\s+(?:unlimited|uncapped|capped)"),
              basis="liability_cap_basis"),
    FieldSpec("late_penalty_usd", MONEY, "liquidated damages per late delivery, in USD",
              (r"liquidated\s+damages", r"late\s+penalt\w*", r"penalt\w*")),
    # insurance_min_usd is reached from insurance_required when the question
    # carries an amount; see NUMERIC_TWINS below.
    FieldSpec("insurance_required", BOOL, "whether the supplier must carry liability insurance",
              (r"insurance\s+certificate", r"certificate\s+of\s+insurance", r"insurance",
               r"\binsured\b")),
    FieldSpec("indemnification", BOOL, "whether the contract contains an indemnification clause",
              (r"indemni(?:f\w*|t(?:y|ies))",)),
    FieldSpec("governing_law", TEXT, "the US state whose law governs the contract",
              (r"governing\s+law", r"governed\s+by", r"\w+[\s\-]governed", r"jurisdiction"),
              values=tuple(GOVERNING_LAWS)),
    FieldSpec("termination_notice_days", NUMBER, "days of notice required to terminate for convenience",
              (r"termination\s+notice", r"notice\s+period", r"notice\s+to\s+terminate")),
    FieldSpec("confidentiality_years", NUMBER, "years confidentiality obligations last after the contract ends",
              (r"confidential\w*",), basis="confidentiality_basis"),
    FieldSpec("assignment_allowed", BOOL, "whether either party may assign the contract",
              (r"assign\w*",)),
    FieldSpec("expiry_date", DATE, "date the contract's current term ends (ISO YYYY-MM-DD)",
              (r"expir\w*", r"expiration", r"end\s+date", r"terminat\w*\s+date", r"\bend(?:s|ing)\b")),
    FieldSpec("effective_date", DATE, "date the contract took effect (ISO YYYY-MM-DD)",
              (r"effective\s+date", r"start\s+date", r"commenc\w*", r"\bstart(?:s|ed|ing)?\b",
               r"\bsigned\b", r"\bexecuted\b")),
    FieldSpec("term_months", NUMBER, "length of the initial term in months",
              (r"term\s+length", r"contract\s+term", r"term\s+of\s+the\s+agreement", r"\bterm\b")),
    FieldSpec("renewal_notice_days", NUMBER, "days of notice needed to prevent an auto-renewal",
              (r"renewal\s+notice", r"non[\s\-]?renewal\s+notice")),
    FieldSpec("insurance_min_usd", MONEY, "minimum insurance coverage per occurrence, in USD",
              (r"insurance\s+(?:coverage|minimum|limit)s?", r"minimum\s+insurance")),
    FieldSpec("vendor", TEXT, "the supplier's company name",
              (r"vendor", r"supplier", r"counterparty")),
    FieldSpec("agreement_type", TEXT, "kind of agreement",
              (r"agreement\s+type", r"type\s+of\s+agreement",
               *AGREEMENT_TYPES.values()),
              values=tuple(AGREEMENT_TYPES)),
    FieldSpec("amends_contract_id", ID, "contract_id of the parent master agreement this one is issued under",
              (r"amend\w*", r"parent\s+agreement", r"under\s+the\s+msa")),
    # Basis columns: not addressable from questions, used by the absence engine.
    FieldSpec("liability_cap_basis", TEXT,
              "why liability_cap_usd is what it is: capped | explicit_unlimited | omitted | unparsed",
              values=("capped", "explicit_unlimited", "omitted", "unparsed")),
    FieldSpec("confidentiality_basis", TEXT,
              "why confidentiality_years is what it is: present | omitted | unparsed",
              values=("present", "omitted", "unparsed")),
]

BY_NAME: dict[str, FieldSpec] = {f.name: f for f in FIELDS}

# A boolean field whose numeric companion answers amount questions about it:
# "require insurance" -> insurance_required, "insurance of at least $2M" -> insurance_min_usd.
NUMERIC_TWINS: dict[str, str] = {"insurance_required": "insurance_min_usd"}

# Fields a question can scope by while asking about a different field
# ("How many *Texas* contracts auto-renew?").
SCOPE_FIELDS = ("governing_law", "agreement_type")

# Derived views -------------------------------------------------------------
COLUMNS: list[str] = ["contract_id", "source_path", *[f.name for f in FIELDS]]
BOOLEAN_FIELDS = {f.name for f in FIELDS if f.kind == BOOL}
NUMERIC_FIELDS = {f.name for f in FIELDS if f.kind in (MONEY, NUMBER)}
DATE_FIELDS = {f.name for f in FIELDS if f.kind == DATE}
BASIS_FIELDS = {f.name: f.basis for f in FIELDS if f.basis}
FIELD_LEXICON: list[tuple[str, str]] = [(f.pattern, f.name) for f in FIELDS if f.pattern]

_SQL_TYPES = {BOOL: "INTEGER", MONEY: "INTEGER", NUMBER: "INTEGER", DATE: "TEXT", TEXT: "TEXT", ID: "TEXT"}


def ddl() -> str:
    """CREATE TABLE for the contracts table, generated from the registry."""
    cols = ["    contract_id TEXT PRIMARY KEY", "    source_path TEXT"]
    cols += [f"    {f.name} {_SQL_TYPES[f.kind]}" for f in FIELDS]
    return "CREATE TABLE IF NOT EXISTS contracts (\n" + ",\n".join(cols) + "\n);"


def canonical_agreement_type(text: str | None) -> str | None:
    """Map any phrasing ("SOW", "STATEMENT OF WORK") to its canonical name."""
    if not text:
        return None
    for name, pat in AGREEMENT_TYPES.items():
        if re.fullmatch(rf"\s*(?:{pat})\s*", text, re.I):
            return name
    return text.strip().title()


def catalog() -> str:
    """Plain-text schema description for the LLM planner prompt."""
    lines = []
    for f in FIELDS:
        vals = f" allowed values: {', '.join(f.values)}." if f.values else ""
        ops = ", ".join(sorted(OPERATORS_BY_KIND[f.kind]))
        lines.append(f"- {f.name} ({f.kind}): {f.description}.{vals} operators: {ops}")
    return "\n".join(lines)
