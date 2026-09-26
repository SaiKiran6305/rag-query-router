"""Wiring: corpus on disk -> extraction -> structured store + retrieval index."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.baseline.naive_rag import NaiveRAGBaseline
from app.engines.core import dispatch
from app.index.retrieval import HybridIndex, chunk_contract
from app.index.structured import StructuredStore
from app.ingest.extract import Extractor, get_extractor
from app.models import Answer
from app.router.classify import QueryPlan, QueryRouter


@dataclass
class SystemStats:
    contracts: int
    chunks: int
    extractor: str
    embedding_backend: str


class ContractIntelligence:
    """The assembled system. Construct once, query many times."""

    def __init__(self, corpus_dir: str | Path, extractor: str = "deterministic",
                 db_path: str = ":memory:", build_index: bool = True,
                 embeddings: str | None = None):
        self.corpus_dir = Path(corpus_dir)
        self.extractor: Extractor = get_extractor(extractor)
        self.store = StructuredStore(db_path)
        self.router = QueryRouter()
        self.index: HybridIndex | None = None
        self.baseline: NaiveRAGBaseline | None = None
        self._chunk_count = 0
        self._embeddings_pref = embeddings
        self._build(build_index)

    def _build(self, build_index: bool) -> None:
        docs = self.extractor.extract_dir(self.corpus_dir)
        self.store.load(docs)

        if build_index:
            chunks = []
            for d in docs:
                chunks.extend(chunk_contract(d.contract_id, d.text))
            self._chunk_count = len(chunks)
            self.index = HybridIndex(self._embeddings_pref)
            self.index.build(chunks)
            self.baseline = NaiveRAGBaseline()

    # -- query paths -------------------------------------------------------
    def plan(self, query: str) -> QueryPlan:
        return self.router.route(query)

    def ask(self, query: str) -> Answer:
        """Routed path: classify, then dispatch to an engine that can answer."""
        return dispatch(self.router.route(query), self.store, self.index)

    def ask_baseline(self, query: str) -> Answer:
        """Naive path: same question, retrieve-then-read."""
        if self.baseline is None or self.index is None:
            raise RuntimeError("index not built; construct with build_index=True")
        return self.baseline.answer(self.router.route(query), self.store, self.index)

    def stats(self) -> SystemStats:
        return SystemStats(
            contracts=self.store.count(),
            chunks=self._chunk_count,
            extractor=self.extractor.name,
            embedding_backend=self.index.embed.kind if self.index else "none",
        )
