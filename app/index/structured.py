"""
Structured fact store: SQLite over extracted facts, with citations preserved.

This is the layer that makes Group B queries answerable. Counting, absence
detection and magnitude comparison are ordinary SQL once the facts are typed
and complete. The engineering that matters happened upstream in extraction --
here it is just a table.

Two properties the engines depend on:

  scan_all()    returns every row, so an aggregate is computed over the corpus
                rather than over a retrieved sample. The Answer.complete flag
                is only ever set True by a path that went through here.

  citations     survive extraction, so any row in an answer set can be traced
                to the exact character span that justified it.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from app.models import Citation, ExtractedContract

SCHEMA = """
CREATE TABLE IF NOT EXISTS contracts (
    contract_id             TEXT PRIMARY KEY,
    source_path             TEXT,
    vendor                  TEXT,
    agreement_type          TEXT,
    governing_law           TEXT,
    effective_date          TEXT,
    expiry_date             TEXT,
    term_months             INTEGER,
    auto_renew              INTEGER,
    renewal_notice_days     INTEGER,
    liability_cap_usd       INTEGER,
    liability_cap_basis     TEXT,
    late_penalty_usd        INTEGER,
    termination_notice_days INTEGER,
    insurance_required      INTEGER,
    insurance_min_usd       INTEGER,
    indemnification         INTEGER,
    assignment_allowed      INTEGER,
    confidentiality_years   INTEGER,
    confidentiality_basis   TEXT,
    amends_contract_id      TEXT
);

CREATE TABLE IF NOT EXISTS citations (
    contract_id TEXT,
    field       TEXT,
    start       INTEGER,
    end         INTEGER,
    snippet     TEXT,
    PRIMARY KEY (contract_id, field)
);

CREATE INDEX IF NOT EXISTS idx_expiry   ON contracts(expiry_date);
CREATE INDEX IF NOT EXISTS idx_law      ON contracts(governing_law);
CREATE INDEX IF NOT EXISTS idx_parent   ON contracts(amends_contract_id);
"""

COLUMNS = [
    "contract_id", "source_path", "vendor", "agreement_type", "governing_law",
    "effective_date", "expiry_date", "term_months", "auto_renew", "renewal_notice_days",
    "liability_cap_usd", "liability_cap_basis", "late_penalty_usd",
    "termination_notice_days", "insurance_required", "insurance_min_usd",
    "indemnification", "assignment_allowed", "confidentiality_years",
    "confidentiality_basis", "amends_contract_id",
]


class StructuredStore:
    def __init__(self, db_path: str | Path = ":memory:"):
        self.db_path = str(db_path)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # -- write ------------------------------------------------------------
    def load(self, docs: Iterable[ExtractedContract]) -> int:
        rows, cits = [], []
        for d in docs:
            row = d.to_row()
            rows.append(tuple(_coerce(row.get(c)) for c in COLUMNS))
            for field_name, fact in d.facts.items():
                if fact.citation:
                    c = fact.citation
                    cits.append((d.contract_id, field_name, c.start, c.end, c.snippet))

        placeholders = ",".join("?" * len(COLUMNS))
        self.conn.executemany(
            f"INSERT OR REPLACE INTO contracts ({','.join(COLUMNS)}) VALUES ({placeholders})", rows
        )
        self.conn.executemany(
            "INSERT OR REPLACE INTO citations (contract_id, field, start, end, snippet) "
            "VALUES (?,?,?,?,?)", cits
        )
        self.conn.commit()
        return len(rows)

    # -- read -------------------------------------------------------------
    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) AS n FROM contracts").fetchone()["n"]

    def scan_all(self) -> list[dict[str, Any]]:
        """Full corpus scan. The only honest basis for a count."""
        return [dict(r) for r in self.conn.execute("SELECT * FROM contracts")]

    def query(self, where: str = "", params: tuple = ()) -> list[dict[str, Any]]:
        sql = "SELECT * FROM contracts"
        if where:
            sql += f" WHERE {where}"
        return [dict(r) for r in self.conn.execute(sql, params)]

    def get(self, contract_id: str) -> dict[str, Any] | None:
        r = self.conn.execute(
            "SELECT * FROM contracts WHERE contract_id = ?", (contract_id,)
        ).fetchone()
        return dict(r) if r else None

    def citation(self, contract_id: str, field_name: str) -> Citation | None:
        r = self.conn.execute(
            "SELECT * FROM citations WHERE contract_id = ? AND field = ?",
            (contract_id, field_name),
        ).fetchone()
        if not r:
            return None
        return Citation(r["contract_id"], r["start"], r["end"], r["snippet"])

    def field_coverage(self) -> dict[str, float]:
        """Fraction of rows with a non-null value per field.

        Low coverage is the signal that extraction, not retrieval, is the thing
        to fix -- and it is the number an absence engine must check before
        trusting a null as a genuine absence rather than a extraction miss.
        """
        n = self.count() or 1
        out: dict[str, float] = {}
        for c in COLUMNS:
            if c in ("contract_id", "source_path"):
                continue
            hit = self.conn.execute(
                f"SELECT COUNT(*) AS k FROM contracts WHERE {c} IS NOT NULL"
            ).fetchone()["k"]
            out[c] = hit / n
        return out

    def close(self) -> None:
        self.conn.close()


def _coerce(v: Any) -> Any:
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (dict, list)):
        return json.dumps(v)
    return v
