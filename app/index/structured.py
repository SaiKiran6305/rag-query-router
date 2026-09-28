"""
Structured fact store: SQLite over extracted facts, with citations preserved.

This is the layer that makes Group B queries answerable. Counting, absence
detection and magnitude comparison are ordinary SQL once the facts are typed
and complete. The engineering that matters happened upstream in extraction --
here it is a table plus a small, safe query compiler.

Three properties the engines depend on:

  select()      compiles a plan's predicates to a *parameterized* WHERE clause
                and runs it over the whole table. Column names are checked
                against the schema registry and operators against a fixed
                whitelist; user text only ever travels as a bound parameter.
                The same compiler serves the baseline, so both sides of the
                benchmark apply byte-identical predicate semantics.

  complete      every engine that answers through select() examined every row
                in scope, so Answer.complete is only ever set True on a path
                that went through here.

  citations     survive extraction, so any row in an answer set can be traced
                to the exact character span that justified it.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from app.models import Citation, ExtractedContract
from app.schema import BOOLEAN_FIELDS, COLUMNS, ddl

SCHEMA = ddl() + """

CREATE TABLE IF NOT EXISTS citations (
    contract_id TEXT,
    field       TEXT,
    start       INTEGER,
    end         INTEGER,
    snippet     TEXT,
    PRIMARY KEY (contract_id, field)
);

CREATE INDEX IF NOT EXISTS idx_expiry    ON contracts(expiry_date);
CREATE INDEX IF NOT EXISTS idx_effective ON contracts(effective_date);
CREATE INDEX IF NOT EXISTS idx_law       ON contracts(governing_law);
CREATE INDEX IF NOT EXISTS idx_parent    ON contracts(amends_contract_id);
"""

# A predicate is (column, operator, value).
Predicate = tuple[str, str, Any]

_CMP = {"gt": ">", "gte": ">=", "lt": "<", "lte": "<="}
_COLUMN_SET = set(COLUMNS)


def compile_predicates(preds: Iterable[Predicate]) -> tuple[str, list[Any]]:
    """Predicates -> (WHERE fragment, bound parameters).

    Semantics are fixed here once, for every caller:
      not_null on a boolean  means true  (NULL and 0 both excluded)
      falsy                  means NULL or 0  (absence of a boolean)
      eq on text             is case-insensitive
      comparisons with NULL  are false
    """
    clauses: list[str] = []
    params: list[Any] = []
    for col, op, value in preds:
        if col not in _COLUMN_SET:
            raise ValueError(f"unknown column {col!r}")
        c = f'"{col}"'                       # safe: col is whitelisted above
        if op in (None, "not_null"):
            clauses.append(f"({c} IS NOT NULL AND {c} != 0)" if col in BOOLEAN_FIELDS
                           else f"{c} IS NOT NULL")
        elif op == "is_null":
            clauses.append(f"{c} IS NULL")
        elif op == "falsy":
            clauses.append(f"({c} IS NULL OR {c} = 0)")
        elif op == "eq":
            if isinstance(value, bool):
                clauses.append(f"COALESCE({c}, 0) = ?")
                params.append(int(value))
            elif isinstance(value, str):
                clauses.append(f"LOWER({c}) = LOWER(?)")
                params.append(value)
            else:
                clauses.append(f"{c} = ?")
                params.append(value)
        elif op == "ne":
            clauses.append(f"({c} IS NULL OR {c} != ?)")
            params.append(value)
        elif op in _CMP:
            clauses.append(f"{c} {_CMP[op]} ?")
            params.append(value)
        elif op == "between":
            lo, hi = value
            clauses.append(f"{c} BETWEEN ? AND ?")
            params.extend([lo, hi])
        elif op == "contains":
            esc = str(value).replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            clauses.append(f"LOWER({c}) LIKE LOWER(?) ESCAPE '\\'")
            params.append(f"%{esc}%")
        elif op == "in":
            vals = list(value or [])
            if not vals:
                clauses.append("0")
            else:
                clauses.append(f"{c} IN ({','.join('?' * len(vals))})")
                params.extend(vals)
        else:
            raise ValueError(f"unsupported operator {op!r}")
    return (" AND ".join(clauses) or "1"), params


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
        """Every row. Used for determinacy checks and tests."""
        return [dict(r) for r in self.conn.execute("SELECT * FROM contracts ORDER BY contract_id")]

    def select(self, preds: Iterable[Predicate] = (), limit: int | None = None) -> list[dict[str, Any]]:
        """Rows matching every predicate, evaluated by SQLite over the full table."""
        where, params = compile_predicates(preds)
        sql = f"SELECT * FROM contracts WHERE {where} ORDER BY contract_id"
        if limit is not None:
            sql += " LIMIT ?"
            params = [*params, int(limit)]
        return [dict(r) for r in self.conn.execute(sql, params)]

    def count_where(self, preds: Iterable[Predicate] = ()) -> int:
        where, params = compile_predicates(preds)
        return self.conn.execute(f"SELECT COUNT(*) AS n FROM contracts WHERE {where}", params).fetchone()["n"]

    def sum_where(self, column: str, preds: Iterable[Predicate] = ()) -> tuple[float | None, int]:
        """(SUM, number of non-null values) of a numeric column over matching rows."""
        if column not in _COLUMN_SET:
            raise ValueError(f"unknown column {column!r}")
        where, params = compile_predicates(preds)
        r = self.conn.execute(
            f'SELECT SUM("{column}") AS s, COUNT("{column}") AS n FROM contracts WHERE {where}', params
        ).fetchone()
        return r["s"], r["n"]

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
        to fix. Column names come from the registry, never from a request.
        """
        total = self.count()
        out: dict[str, float] = {}
        for c in COLUMNS:
            if c in ("contract_id", "source_path"):
                continue
            nulls = self.count_where([(c, "is_null", None)])
            out[c] = (total - nulls) / total if total else 0.0
        return out

    def close(self) -> None:
        self.conn.close()


def _coerce(v: Any) -> Any:
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (dict, list)):
        return json.dumps(v)
    return v
