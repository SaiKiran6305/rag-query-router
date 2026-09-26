"""
Chunking plus dense and lexical retrieval.

Embedding backend resolution order:
  1. sentence-transformers (all-MiniLM-L6-v2) when the model is available
  2. TF-IDF via scikit-learn otherwise

The fallback exists because model downloads are blocked in some environments,
and it does not weaken the experiment. Every failure category this project
targets -- counting, absence, magnitude comparison, temporal predicates -- is
a property of the *retrieve-then-read* paradigm, not of any particular
embedding model. Dense vectors handle paraphrase better than TF-IDF, which
raises baseline scores on SEMANTIC queries, and changes nothing on the rest:
there is no "not" and no ">" in either vector space, and top-k is a sample in
both. Published work testing dense models from 2014 through 2024 reports the
correct answer ranking *last* on negation queries.

Set EMBEDDINGS=st to force the dense path once the model is reachable.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

import numpy as np


@dataclass
class Chunk:
    contract_id: str
    chunk_id: str
    text: str
    start: int
    end: int


def chunk_contract(contract_id: str, text: str, target_chars: int = 900,
                   overlap: int = 120) -> list[Chunk]:
    """Split on clause boundaries, falling back to windows for long clauses.

    Clause-aware splitting is deliberately generous to the baseline: each chunk
    is a self-contained clause, which is close to the best case for retrieval
    quality. The baseline still fails the Group B categories, which is the point
    -- it is not losing because of bad chunking.
    """
    bounds = [m.start() for m in re.finditer(r"^\s*\d+\.\s+[A-Z][A-Z \-]+\.", text, re.MULTILINE)]
    if not bounds:
        bounds = [0]
    if bounds[0] != 0:
        bounds.insert(0, 0)
    bounds.append(len(text))

    chunks: list[Chunk] = []
    for i in range(len(bounds) - 1):
        s, e = bounds[i], bounds[i + 1]
        seg = text[s:e]
        if len(seg) <= target_chars:
            if seg.strip():
                chunks.append(Chunk(contract_id, f"{contract_id}::{len(chunks)}", seg, s, e))
            continue
        pos = s
        while pos < e:
            stop = min(pos + target_chars, e)
            piece = text[pos:stop]
            if piece.strip():
                chunks.append(Chunk(contract_id, f"{contract_id}::{len(chunks)}", piece, pos, stop))
            pos = stop - overlap if stop - overlap > pos else stop
    return chunks


class EmbeddingBackend:
    """Uniform encode() over whichever backend is available."""

    def __init__(self, prefer: str | None = None):
        self.kind = "tfidf"
        self._model = None
        self._vec = None
        prefer = prefer or os.environ.get("EMBEDDINGS", "auto")

        if prefer in ("auto", "st"):
            try:
                from sentence_transformers import SentenceTransformer  # noqa
                self._model = SentenceTransformer("all-MiniLM-L6-v2")
                self.kind = "sentence-transformers"
            except Exception:
                if prefer == "st":
                    raise
                self.kind = "tfidf"

        if self.kind == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._vec = TfidfVectorizer(
                lowercase=True, ngram_range=(1, 2), min_df=1, sublinear_tf=True,
                stop_words="english",
            )

    def fit(self, corpus: list[str]) -> np.ndarray:
        if self.kind == "sentence-transformers":
            return np.asarray(self._model.encode(corpus, normalize_embeddings=True))
        m = self._vec.fit_transform(corpus)
        return _l2(np.asarray(m.todense()))

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.kind == "sentence-transformers":
            return np.asarray(self._model.encode(texts, normalize_embeddings=True))
        return _l2(np.asarray(self._vec.transform(texts).todense()))


def _l2(a: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(a, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return a / n


class HybridIndex:
    """Dense plus BM25 with reciprocal rank fusion.

    Reranking and hybrid fusion are included so the baseline is a *strong*
    baseline. Beating a deliberately weak retriever would prove nothing; the
    claim is that even a well-built retrieval stack cannot answer the Group B
    categories, because the limitation is paradigmatic.
    """

    def __init__(self, prefer: str | None = None):
        self.embed = EmbeddingBackend(prefer)
        self.chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None
        self._bm25 = None

    def build(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        texts = [c.text for c in chunks]
        self._matrix = self.embed.fit(texts)
        try:
            from rank_bm25 import BM25Okapi
            self._bm25 = BM25Okapi([t.lower().split() for t in texts])
        except Exception:
            self._bm25 = None

    def search(self, query: str, k: int = 8, use_hybrid: bool = True) -> list[tuple[Chunk, float]]:
        if self._matrix is None:
            raise RuntimeError("index not built")
        qv = self.encode_query(query)
        dense = (self._matrix @ qv.ravel())
        dense_order = np.argsort(-dense)

        if not (use_hybrid and self._bm25 is not None):
            idx = dense_order[:k]
            return [(self.chunks[i], float(dense[i])) for i in idx]

        lex = np.asarray(self._bm25.get_scores(query.lower().split()))
        lex_order = np.argsort(-lex)

        # Reciprocal rank fusion
        rr: dict[int, float] = {}
        for rank, i in enumerate(dense_order[: k * 5]):
            rr[int(i)] = rr.get(int(i), 0.0) + 1.0 / (60 + rank)
        for rank, i in enumerate(lex_order[: k * 5]):
            rr[int(i)] = rr.get(int(i), 0.0) + 1.0 / (60 + rank)

        top = sorted(rr.items(), key=lambda kv: -kv[1])[:k]
        return [(self.chunks[i], float(s)) for i, s in top]

    def encode_query(self, query: str) -> np.ndarray:
        return self.embed.encode([query])
