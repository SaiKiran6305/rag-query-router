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
from app.router.classify import QueryPlan
from app.router.llm import HybridRouter, build_router


@dataclass
class SystemStats:
    contracts: int
    chunks: int
    extractor: str
    embedding_backend: str
    router: str
    llm_calls: int
    llm_rejections: int


class ContractIntelligence:
    """The assembled system. Construct once, query many times."""

    def __init__(self, corpus_dir: str | Path, extractor: str = "deterministic",
                 db_path: str = ":memory:", build_index: bool = True,
                 embeddings: str | None = None, router: HybridRouter | None = None):
        self.corpus_dir = Path(corpus_dir)
        self.extractor: Extractor = get_extractor(extractor)
        self.store = StructuredStore(db_path)
        self.router = router or build_router()
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

    def ask(self, query: str, plan: QueryPlan | None = None) -> Answer:
        """Routed path: classify, then dispatch to an engine that can answer."""
        return dispatch(plan or self.plan(query), self.store, self.index)

    def ask_baseline(self, query: str, plan: QueryPlan | None = None) -> Answer:
        """Naive path: same question, same plan, retrieve-then-read."""
        if self.baseline is None or self.index is None:
            raise RuntimeError("index not built; construct with build_index=True")
        return self.baseline.answer(plan or self.plan(query), self.store, self.index)

    def stats(self) -> SystemStats:
        planner = getattr(self.router, "planner", None)
        return SystemStats(
            contracts=self.store.count(),
            chunks=self._chunk_count,
            extractor=self.extractor.name,
            embedding_backend=self.index.embed.kind if self.index else "none",
            router=getattr(self.router, "kind", "rules"),
            llm_calls=planner.calls if planner else 0,
            llm_rejections=planner.rejections if planner else 0,
        )
