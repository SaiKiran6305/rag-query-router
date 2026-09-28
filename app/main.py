"""
FastAPI application. Single deployable: API on /api/*, static UI on /.

One process, one container, one URL -- no CORS, no second deployment. The
frontend is built and served by the same app, which is the right topology for
a demo someone should be able to open and use in ninety seconds.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.pipeline import ContractIntelligence

ROOT = Path(__file__).resolve().parent.parent
CORPUS = Path(os.environ.get("CORPUS_DIR", ROOT / "data" / "corpus"))
STATIC = ROOT / "static"

state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Built once at startup: extraction and index construction are not per-request work.
    state["system"] = ContractIntelligence(
        CORPUS / "contracts",
        extractor=os.environ.get("EXTRACTOR", "deterministic"),
        embeddings=os.environ.get("EMBEDDINGS"),
    )
    yield
    state.clear()


app = FastAPI(
    title="Contract Intelligence",
    description="Query routing over structured contract facts. Answers the "
                "question categories top-k retrieval cannot.",
    version="1.0.0",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    query: str
    compare: bool = True


def _system() -> ContractIntelligence:
    s = state.get("system")
    if s is None:
        raise HTTPException(503, "system not initialised")
    return s


@app.get("/api/health")
def health() -> dict[str, Any]:
    s = state.get("system")
    if s is None:
        return {"status": "starting"}
    st = s.stats()
    return {
        "status": "ok",
        "contracts": st.contracts,
        "chunks": st.chunks,
        "extractor": st.extractor,
        "embedding_backend": st.embedding_backend,
        "router": st.router,
        "llm_calls": st.llm_calls,
        "llm_rejections": st.llm_rejections,
    }


@app.post("/api/ask")
def ask(req: AskRequest) -> dict[str, Any]:
    """Route a query, optionally alongside the naive baseline for comparison."""
    s = _system()
    if not req.query.strip():
        raise HTTPException(400, "empty query")

    plan = s.plan(req.query)            # routed once; both paths share the plan
    routed = s.ask(req.query, plan)
    out: dict[str, Any] = {"plan": plan.to_dict(), "routed": routed.to_dict()}
    if req.compare:
        out["baseline"] = s.ask_baseline(req.query, plan).to_dict()
    return out


@app.get("/api/schema")
def schema() -> list[dict[str, Any]]:
    """Every field the system can answer questions about, with its type and meaning."""
    from app.schema import FIELDS, OPERATORS_BY_KIND
    return [
        {"field": f.name, "kind": f.kind, "description": f.description,
         "values": list(f.values), "operators": sorted(OPERATORS_BY_KIND[f.kind])}
        for f in FIELDS if f.synonyms
    ]


@app.get("/api/plan")
def explain_plan(q: str = Query(..., description="query to classify")) -> dict[str, Any]:
    """Routing decision only. Useful for debugging misroutes without executing."""
    return _system().plan(q).to_dict()


@app.get("/api/contract/{contract_id}")
def contract(contract_id: str) -> dict[str, Any]:
    s = _system()
    row = s.store.get(contract_id)
    if not row:
        raise HTTPException(404, f"unknown contract {contract_id}")
    text = Path(row["source_path"]).read_text(encoding="utf-8")
    return {"facts": row, "text": text}


@app.get("/api/coverage")
def coverage() -> dict[str, float]:
    """Per-field extraction coverage. The number the absence engine gates on."""
    return _system().store.field_coverage()


@app.get("/api/eval")
def eval_report() -> JSONResponse:
    p = ROOT / "eval_report.json"
    if not p.exists():
        raise HTTPException(404, "no eval report; run `python -m app.eval.run_eval` first")
    return JSONResponse(content=__import__("json").loads(p.read_text(encoding="utf-8")))


@app.get("/api/examples")
def examples() -> list[dict[str, str]]:
    """Curated queries, one per failure category, for the UI."""
    return [
        {"q": "How many contracts have auto renewal?", "cat": "counting",
         "why": "A count needs the corpus. Retrieval returns a sample of it."},
        {"q": "Which contracts have no liability cap?", "cat": "absence",
         "why": "Absent text cannot be retrieved; explicit disclaimers rank with their opposite."},
        {"q": "Which contracts have penalties above $100,000?", "cat": "magnitude",
         "why": "Embeddings encode that a number is present, not how large it is."},
        {"q": "List contracts expiring in Q1 2026", "cat": "temporal",
         "why": "Vector space has no ordering over dates."},
        {"q": "Which vendors are on statements of work issued under California master agreements?",
         "cat": "multihop",
         "why": "The second hop consumes the first hop's output."},
        {"q": "List every contract governed by Delaware law", "cat": "completeness",
         "why": "Top-k cannot signal that its list is partial."},
        {"q": "How many Texas contracts renew on their own?", "cat": "paraphrase",
         "why": "No 'auto-renew' keyword, plus a scope filter. The plan bar shows how it was read."},
        {"q": "What expires before June 2026?", "cat": "before/after",
         "why": "A date comparison, not a date window. Vector space has no 'before'."},
        {"q": "What does the indemnification clause say?", "cat": "control",
         "why": "Retrieval is the right tool here. Both systems score the same."},
        {"q": "What is the weather in Dallas?", "cat": "out of scope",
         "why": "Nothing in the contracts answers this, so the routed system declines instead of guessing."},
    ]


if STATIC.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(str(STATIC / "index.html"))
