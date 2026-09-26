"""Typed schemas shared across ingestion, indexing, routing and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


class QueryType(str, Enum):
    """Query taxonomy. The router maps a question onto exactly one of these.

    SEMANTIC is the only type naive top-k retrieval can serve correctly.
    The rest require computation over the structured layer -- that is the
    entire thesis of this system.
    """
    SEMANTIC = "semantic"            # "what does the indemnification clause say"
    AGGREGATE = "aggregate"          # "how many contracts auto-renew"
    ABSENCE = "absence"              # "which contracts have no liability cap"
    NUMERIC = "numeric"              # "penalties above $100,000"
    TEMPORAL = "temporal"            # "expiring in Q1 2026"
    MULTIHOP = "multihop"            # "vendors under SOWs amending Delaware MSAs"
    ENUMERATE = "enumerate"          # "list every contract governed by New York law"


@dataclass
class Citation:
    """A verifiable pointer back to source text.

    Every structured fact carries one. A reviewer can check any claim the
    system makes by looking at the exact character span it came from.
    """
    contract_id: str
    start: int
    end: int
    snippet: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Fact:
    """One extracted field with provenance and extractor confidence."""
    name: str
    value: Any
    citation: Citation | None
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "confidence": self.confidence,
            "citation": self.citation.to_dict() if self.citation else None,
        }


@dataclass
class ExtractedContract:
    """Structured representation of one contract, built only from its prose."""
    contract_id: str
    source_path: str
    text: str
    facts: dict[str, Fact] = field(default_factory=dict)

    def value(self, name: str, default: Any = None) -> Any:
        f = self.facts.get(name)
        return f.value if f is not None else default

    def citation(self, name: str) -> Citation | None:
        f = self.facts.get(name)
        return f.citation if f else None

    def to_row(self) -> dict[str, Any]:
        """Flatten to a row for the structured store."""
        row: dict[str, Any] = {"contract_id": self.contract_id, "source_path": self.source_path}
        for name, f in self.facts.items():
            row[name] = f.value
        return row


@dataclass
class AnswerItem:
    """One contract in an answer set, with the citation that justifies it."""
    contract_id: str
    reason: str
    citation: Citation | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_id": self.contract_id,
            "reason": self.reason,
            "citation": self.citation.to_dict() if self.citation else None,
        }


@dataclass
class Answer:
    """A routed answer.

    `complete` is the field that matters most and that naive RAG cannot set
    honestly: it records whether the engine examined the full corpus or only
    a retrieved sample. A count derived from a top-k sample is not a count.
    """
    query: str
    query_type: QueryType
    engine: str
    value: Any                       # scalar for aggregates, list for sets
    items: list[AnswerItem] = field(default_factory=list)
    complete: bool = False
    scanned: int = 0
    explanation: str = ""
    abstained: bool = False
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "query_type": self.query_type.value,
            "engine": self.engine,
            "value": self.value,
            "items": [i.to_dict() for i in self.items],
            "complete": self.complete,
            "scanned": self.scanned,
            "explanation": self.explanation,
            "abstained": self.abstained,
            "latency_ms": round(self.latency_ms, 2),
        }
