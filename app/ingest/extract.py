"""
Structured extraction: contract prose -> typed facts with span citations.

Two providers behind one interface.

DeterministicExtractor runs with no API key and no model download. It exists
so the whole pipeline -- including the eval that proves the thesis -- is
reproducible by anyone who clones the repo. It is pattern based and therefore
brittle outside this corpus, which is stated plainly rather than hidden.

LLMExtractor is the path for real, messy contracts. It emits the same Fact
objects against the same schema, so every downstream engine, the router and
the eval harness are unchanged when you swap providers. That substitutability
is the point of the interface.

The extraction target that actually matters is liability cap, because it has
three states and two of them mean "no cap":
    capped               -> "aggregate liability ... shall not exceed $500,000"
    explicit_unlimited   -> "no monetary limitation shall apply"
    omitted              -> no liability section at all
A retrieval system asked "which contracts lack a cap" surfaces the first group,
because that is the text most similar to the query. Both correct groups are
invisible to it -- one is semantically opposite, the other does not exist as text.
"""

from __future__ import annotations

import os
import re
from abc import ABC, abstractmethod
from datetime import date, datetime
from pathlib import Path

from app.models import Citation, ExtractedContract, Fact
from app.schema import FIELDS as SCHEMA_FIELDS, canonical_agreement_type

# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_MONEY_WORDS = {
    "two hundred fifty thousand": 250_000,
    "five hundred thousand": 500_000,
    "seven hundred fifty thousand": 750_000,
    "one million": 1_000_000,
    "two million": 2_000_000,
    "five million": 5_000_000,
}

_MONEY_RE = re.compile(
    r"\$\s?(?P<num>[\d,]+(?:\.\d+)?)\s*(?P<suffix>[KM])?\b",
    re.IGNORECASE,
)

_DATE_PATTERNS = [
    (re.compile(r"\b(?P<m>January|February|March|April|May|June|July|August|September|October|November|December)\s+(?P<d>\d{1,2}),\s*(?P<y>\d{4})\b"), "long"),
    (re.compile(r"\b(?P<mm>\d{2})/(?P<dd>\d{2})/(?P<yyyy>\d{4})\b"), "slash"),
    (re.compile(r"\b(?P<yyyy>\d{4})-(?P<mm>\d{2})-(?P<dd>\d{2})\b"), "iso"),
]

_MONTHS = {m: i + 1 for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June",
     "July", "August", "September", "October", "November", "December"]
)}


def parse_money(text: str) -> int | None:
    """Handle $500,000 / $500K / $5M / 'Five Hundred Thousand Dollars ($500,000)'.

    The word form always carries a parenthetical with digits in this corpus, so
    the digit path wins; the word table is a fallback for forms that omit it.
    """
    m = _MONEY_RE.search(text)
    if m:
        raw = m.group("num").replace(",", "")
        try:
            val = float(raw)
        except ValueError:
            return None
        suffix = (m.group("suffix") or "").upper()
        if suffix == "K":
            val *= 1_000
        elif suffix == "M":
            val *= 1_000_000
        return int(val)

    low = text.lower()
    for phrase, val in _MONEY_WORDS.items():
        if phrase in low:
            return val
    return None


def parse_date(text: str) -> date | None:
    for pat, kind in _DATE_PATTERNS:
        m = pat.search(text)
        if not m:
            continue
        try:
            if kind == "long":
                return date(int(m.group("y")), _MONTHS[m.group("m")], int(m.group("d")))
            if kind == "slash":
                return date(int(m.group("yyyy")), int(m.group("mm")), int(m.group("dd")))
            return date(int(m.group("yyyy")), int(m.group("mm")), int(m.group("dd")))
        except (ValueError, KeyError):
            continue
    return None


def _section(text: str, number: int) -> tuple[str, int] | None:
    """Return (section_text, absolute_start_offset) for a numbered clause."""
    pat = re.compile(rf"^\s*{number}\.\s+[A-Z][A-Z \-]+\.", re.MULTILINE)
    m = pat.search(text)
    if not m:
        return None
    start = m.start()
    nxt = re.compile(rf"^\s*{number + 1}\.\s+[A-Z][A-Z \-]+\.", re.MULTILINE).search(text, m.end())
    end = nxt.start() if nxt else len(text)
    return text[start:end], start


def _find_section_by_heading(text: str, heading: str) -> tuple[str, int] | None:
    """Locate a clause by its heading word, independent of numbering."""
    m = re.search(rf"^\s*\d+\.\s+{heading}[A-Z \-]*\.", text, re.MULTILINE | re.IGNORECASE)
    if not m:
        return None
    start = m.start()
    nxt = re.compile(r"^\s*\d+\.\s+[A-Z][A-Z \-]+\.", re.MULTILINE).search(text, m.end())
    end = nxt.start() if nxt else len(text)
    return text[start:end], start


def _cite(contract_id: str, text: str, abs_start: int, local_span: tuple[int, int] | None = None,
          max_len: int = 220) -> Citation:
    if local_span:
        s = abs_start + local_span[0]
        e = abs_start + local_span[1]
    else:
        s = abs_start
        e = abs_start + min(len(text), max_len)
    snippet = text[:max_len].strip() if not local_span else text[local_span[0]:local_span[1]].strip()
    return Citation(contract_id=contract_id, start=s, end=e, snippet=snippet)


# ---------------------------------------------------------------------------
# Provider interface
# ---------------------------------------------------------------------------

class Extractor(ABC):
    name: str = "base"

    @abstractmethod
    def extract(self, contract_id: str, text: str, source_path: str) -> ExtractedContract:
        ...

    def extract_dir(self, directory: Path) -> list[ExtractedContract]:
        out: list[ExtractedContract] = []
        for p in sorted(directory.glob("*.txt")):
            out.append(self.extract(p.stem, p.read_text(encoding="utf-8"), str(p)))
        return out


# A document whose clause structure the extractor cannot see gives it no right
# to report a clause as *absent*. Every synthetic contract has 6+ numbered
# headings in this form; 436 of CUAD's 510 real contracts have none.
_HEADING = re.compile(r"^\s*\d+\.\s+[A-Z][A-Z \-]+\.", re.MULTILINE)
MIN_HEADINGS_FOR_ABSENCE = 5


class DeterministicExtractor(Extractor):
    """Pattern based. No API key, no model download, fully reproducible.

    Honest limitation: tuned to the clause conventions of this corpus. On real
    contracts with unseen drafting styles, recall degrades and the LLM provider
    is the correct choice. The eval reports extraction accuracy per field so
    this is measured rather than assumed.
    """
    name = "deterministic"

    def extract(self, contract_id: str, text: str, source_path: str) -> ExtractedContract:
        c = ExtractedContract(contract_id=contract_id, source_path=source_path, text=text)

        # Can we see this document's clause structure? If not, "no LIABILITY
        # heading found" means "unknown", not "the contract has no cap" -- on
        # CUAD the old behaviour marked 255 contracts with a liability clause as
        # 'omitted', which the absence engine then reported as settled fact.
        structured = len(_HEADING.findall(text)) >= MIN_HEADINGS_FOR_ABSENCE
        absent_basis = "omitted" if structured else "unknown"
        absent_flag = False if structured else None

        def put(name: str, value, citation: Citation | None, conf: float = 1.0) -> None:
            c.facts[name] = Fact(name=name, value=value, citation=citation, confidence=conf)

        # --- identity -------------------------------------------------------
        m = re.search(r"Contract Reference:\s*(\S+)", text)
        put("contract_id_stated", m.group(1) if m else contract_id,
            _cite(contract_id, m.group(0), m.start()) if m else None)

        # Anchor on the Customer parenthetical: a bare `and (.+?)` would match the
        # first "and" in the preamble and swallow the customer clause.
        m = re.search(r'\("Customer"\),\s*and\s+(.+?)\s*\("Supplier"\)', text, re.DOTALL)
        put("vendor", m.group(1).strip() if m else None,
            _cite(contract_id, m.group(0), m.start()) if m else None)

        m = re.search(r"^([A-Z][A-Z \-]+)\n", text)
        # Canonical name, not .title(): "STATEMENT OF WORK".title() is
        # "Statement Of Work", which silently failed exact matches.
        put("agreement_type", canonical_agreement_type(m.group(1)) if m else None,
            _cite(contract_id, m.group(0), m.start()) if m else None)

        # Multi-word states ("New York") need the repeated capitalised group.
        m = re.search(r"State of ([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)", text)
        put("governing_law", m.group(1) if m else None,
            _cite(contract_id, m.group(0), m.start()) if m else None)

        # --- dates ----------------------------------------------------------
        m = re.search(r"entered into as of (.+?) \(the \"Effective Date\"\)", text)
        eff = parse_date(m.group(1)) if m else None
        put("effective_date", eff.isoformat() if eff else None,
            _cite(contract_id, m.group(0), m.start()) if m else None)

        term_sec = _find_section_by_heading(text, "TERM")
        expiry = None
        term_months = None
        auto_renew = None
        renewal_notice = None
        if term_sec:
            sec_text, sec_start = term_sec
            m_exp = re.search(r"(?:expir(?:es|ing)|ending|terminating|expires on)\s+(?:on\s+)?([A-Za-z0-9,/\- ]+?)(?:,|\.|\s+unless|\s+a period)", sec_text)
            if m_exp:
                expiry = parse_date(m_exp.group(1))
            if expiry is None:
                # fall back to any date in the clause other than the effective date
                for pat, _ in _DATE_PATTERNS:
                    for mm in pat.finditer(sec_text):
                        d = parse_date(mm.group(0))
                        if d and d != eff:
                            expiry = d
                            break
                    if expiry:
                        break

            m_term = re.search(r"(\d+)\s*month", sec_text)
            term_months = int(m_term.group(1)) if m_term else None

            low = sec_text.lower()
            negative = ("does not renew automatically" in low
                        or "continuation beyond" in low
                        or "requires mutual written agreement" in low)
            positive = ("automatically renew" in low
                        or "renews automatically" in low
                        or "renew for successive" in low)
            auto_renew = bool(positive and not negative)

            if auto_renew:
                m_not = re.search(r"(\d+)\s*days?\s+(?:prior|advance)", sec_text)
                renewal_notice = int(m_not.group(1)) if m_not else None

            cit = _cite(contract_id, sec_text, sec_start)
        else:
            cit = None

        put("expiry_date", expiry.isoformat() if expiry else None, cit)
        put("term_months", term_months, cit)
        put("auto_renew", auto_renew, cit)
        put("renewal_notice_days", renewal_notice, cit)

        # --- liability cap: the three-state field ---------------------------
        liab = _find_section_by_heading(text, "LIMITATION OF LIABILITY") or _find_section_by_heading(text, "LIABILITY")
        if liab is None:
            put("liability_cap_usd", None, None)
            put("liability_cap_basis", absent_basis, None)
        else:
            sec_text, sec_start = liab
            low = sec_text.lower()
            unlimited = ("no monetary limitation" in low
                         or "without cap or limitation" in low
                         or "no limitation shall apply" in low)
            if unlimited:
                put("liability_cap_usd", None, _cite(contract_id, sec_text, sec_start))
                put("liability_cap_basis", "explicit_unlimited", _cite(contract_id, sec_text, sec_start))
            else:
                m_cap = re.search(r"shall not exceed\s+([^.]+)", sec_text)
                amt = parse_money(m_cap.group(1)) if m_cap else None
                put("liability_cap_usd", amt, _cite(contract_id, sec_text, sec_start))
                put("liability_cap_basis", "capped" if amt else "unparsed",
                    _cite(contract_id, sec_text, sec_start))

        # --- penalties ------------------------------------------------------
        pay = _find_section_by_heading(text, "PAYMENT")
        penalty = None
        if pay:
            sec_text, sec_start = pay
            # `[^,]+?` broke on thousands separators inside the amount ($750,000).
            m_pen = re.search(r"liquidated damages of\s+(.+?)\s+per occurrence", sec_text)
            if m_pen:
                penalty = parse_money(m_pen.group(1))
            put("late_penalty_usd", penalty, _cite(contract_id, sec_text, sec_start))
        else:
            put("late_penalty_usd", None, None)

        # --- insurance ------------------------------------------------------
        ins = _find_section_by_heading(text, "INSURANCE")
        if ins:
            sec_text, sec_start = ins
            m_ins = re.search(r"not less than\s+([^ ]+(?:,\d{3})*)\s+per occurrence", sec_text)
            amt = parse_money(m_ins.group(1)) if m_ins else parse_money(sec_text)
            put("insurance_required", True, _cite(contract_id, sec_text, sec_start))
            put("insurance_min_usd", amt, _cite(contract_id, sec_text, sec_start))
        else:
            put("insurance_required", absent_flag, None)
            put("insurance_min_usd", None, None)

        # --- booleans and remaining scalars ---------------------------------
        indem = _find_section_by_heading(text, "INDEMNIFICATION")
        put("indemnification", True if indem is not None else absent_flag,
            _cite(contract_id, indem[0], indem[1]) if indem else None)

        assign = _find_section_by_heading(text, "ASSIGNMENT")
        if assign:
            sec_text, sec_start = assign
            allowed = "may assign" in sec_text.lower() and "neither party may assign" not in sec_text.lower()
            put("assignment_allowed", allowed, _cite(contract_id, sec_text, sec_start))
        else:
            put("assignment_allowed", None, None)

        # Basis columns generalise the liability pattern: record *why* a value is
        # absent so a downstream absence query can tell a settled finding ("the
        # clause is not in this contract") from an extraction miss ("we failed to
        # parse a clause that is there"). Without it the absence engine has to
        # abstain, which is correct but unhelpful when the information is known.
        conf_sec = _find_section_by_heading(text, "CONFIDENTIALITY")
        if conf_sec:
            sec_text, sec_start = conf_sec
            m_y = re.search(r"period of\s+(\d+)\s*years?", sec_text)
            cit_c = _cite(contract_id, sec_text, sec_start)
            put("confidentiality_years", int(m_y.group(1)) if m_y else None, cit_c)
            put("confidentiality_basis", "present" if m_y else "unparsed", cit_c)
        else:
            put("confidentiality_years", None, None)
            put("confidentiality_basis", absent_basis, None)

        term_sec2 = _find_section_by_heading(text, "TERMINATION")
        if term_sec2:
            sec_text, sec_start = term_sec2
            m_n = re.search(r"upon\s+(\d+)\s*days", sec_text)
            put("termination_notice_days", int(m_n.group(1)) if m_n else None,
                _cite(contract_id, sec_text, sec_start))
        else:
            put("termination_notice_days", None, None)

        # --- multi-hop link -------------------------------------------------
        m_parent = re.search(r"bearing reference\s+(\S+?)[.\s]", text)
        put("amends_contract_id", m_parent.group(1) if m_parent else None,
            _cite(contract_id, m_parent.group(0), m_parent.start()) if m_parent else None)

        return c


class LLMExtractor(Extractor):
    """Schema-constrained extraction via an LLM. Requires OPENAI_API_KEY.

    Emits the identical Fact schema as the deterministic provider, so routing,
    engines and eval are untouched by the swap. Included as the production path
    for real contracts; the repo's default eval runs without it so results are
    reproducible with no credentials.
    """
    name = "llm"

    # Derived from the schema registry so the prompt can never drift from the table.
    FIELDS = [f.name for f in SCHEMA_FIELDS]

    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "LLMExtractor needs OPENAI_API_KEY. Use DeterministicExtractor for the "
                "credential-free path."
            )

    def extract(self, contract_id: str, text: str, source_path: str) -> ExtractedContract:  # pragma: no cover
        from openai import OpenAI  # imported lazily so the package stays optional

        client = OpenAI(api_key=self.api_key)
        schema_hint = ", ".join(self.FIELDS)
        resp = client.chat.completions.create(
            model=self.model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content":
                    "Extract contract facts as JSON. For every field also return "
                    "`<field>_quote`: the exact substring of the contract that justifies it, "
                    "or null. Never invent a quote. If a clause is absent set the field to null "
                    "and the quote to null. For liability_cap_basis use exactly one of: "
                    "capped, explicit_unlimited, omitted. For confidentiality_basis use "
                    "exactly one of: present, omitted, unparsed. The *_basis fields are what "
                    "let a downstream absence query tell a settled finding from a parse miss, "
                    "so never guess one."},
                {"role": "user", "content": f"Fields: {schema_hint}\n\nContract:\n{text}"},
            ],
        )
        import json as _json
        data = _json.loads(resp.choices[0].message.content or "{}")

        c = ExtractedContract(contract_id=contract_id, source_path=source_path, text=text)
        for fname in self.FIELDS:
            val = data.get(fname)
            quote = data.get(f"{fname}_quote")
            cit = None
            if isinstance(quote, str) and quote:
                idx = text.find(quote)
                if idx >= 0:  # only cite quotes that genuinely appear -- guards hallucinated spans
                    cit = Citation(contract_id, idx, idx + len(quote), quote)
            c.facts[fname] = Fact(fname, val, cit, confidence=1.0 if cit else 0.5)
        return c


def get_extractor(name: str = "deterministic") -> Extractor:
    if name == "llm":
        return LLMExtractor()
    return DeterministicExtractor()
