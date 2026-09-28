"""
Naive RAG baseline: chunk, embed, retrieve top-k, answer from context.

This baseline is deliberately generous, because beating a weak one would prove
nothing.

Three concessions are made to it:

  1. Clause-aware chunking. Each chunk is a semantically complete clause --
     close to the best case for retrieval quality.
  2. Hybrid retrieval with reciprocal rank fusion (dense + BM25), not dense
     alone. This is the strong 2026 pattern, not a naive one.
  3. Perfect reading comprehension over whatever it retrieved. Instead of
     asking an LLM to read the chunks, the baseline is handed the already
     extracted structured facts for the contracts its chunks came from, and
     applies the router's plan to them through the *same* SQL predicate
     compiler the routed engines use -- so the two sides cannot disagree about
     what a predicate means, only about which rows they looked at.

Concession 3 is the important one. It means the baseline never makes a reading
error, never hallucinates, and never misparses an amount. Every failure it
records is attributable to retrieval alone -- the top-k set simply did not
contain what the question needed. A real LLM pipeline would score at or below
these numbers, never above, so the measured gap is a lower bound on the gap.

What it cannot do, structurally:
  - count, because it sees k chunks and not the corpus
  - detect absence, because absent text cannot be retrieved
  - compare magnitudes, because retrieval ranks by similarity not by value
  - traverse, because one retrieval pass cannot use its own output as input
"""

from __future__ import annotations

import time
from typing import Any

from app.index.retrieval import HybridIndex
from app.index.structured import StructuredStore
from app.models import Answer, AnswerItem, QueryType
from app.router.classify import BOOLEAN_FIELDS, QueryPlan

DEFAULT_K = 8


class NaiveRAGBaseline:
    name = "naive_rag"

    def __init__(self, k: int = DEFAULT_K):
        self.k = k

    def answer(self, plan: QueryPlan, store: StructuredStore, index: HybridIndex) -> Answer:
        t0 = time.perf_counter()

        hits = index.search(plan.query, k=self.k)

        # Contracts represented in the retrieved window, order preserved.
        seen: list[str] = []
        for chunk, _ in hits:
            if chunk.contract_id not in seen:
                seen.append(chunk.contract_id)

        f, op, v = plan.field, plan.operator, plan.value

        # The pipeline applies the question's predicate to its retrieved window.
        # This is what "the LLM reads the context and answers" amounts to when
        # the reading is perfect.
        preds: list[tuple[str, str, Any]] = [("contract_id", "in", seen)]
        preds += [(col, "eq", val) for col, val in (plan.filters or {}).items()]
        if plan.query_type == QueryType.ABSENCE and f:
            preds.append((f, "falsy" if f in BOOLEAN_FIELDS else "is_null", None))
        elif plan.query_type != QueryType.SEMANTIC and f is not None and op is not None:
            preds.append((f, op, v))

        order = {cid: i for i, cid in enumerate(seen)}
        matched = sorted(store.select(preds), key=lambda r: order[r["contract_id"]])

        if plan.aggregation == "count" or plan.query_type == QueryType.AGGREGATE:
            value: Any = len(matched)
        else:
            value = [r["contract_id"] for r in matched]

        return Answer(
            query=plan.query,
            query_type=plan.query_type,
            engine=self.name,
            value=value,
            items=[
                AnswerItem(contract_id=r["contract_id"], reason=f"in top-{self.k} retrieved window")
                for r in matched[:50]
            ],
            # Never complete: the window is a sample of the corpus by construction.
            complete=False,
            scanned=len(seen),
            explanation=(
                f"Retrieved top {self.k} chunks spanning {len(seen)} contracts out of "
                f"{store.count()} in the corpus, then applied the predicate to that window. "
                f"Any count reported here is a count of the window, not of the corpus."
            ),
            latency_ms=(time.perf_counter() - t0) * 1000,
        )
