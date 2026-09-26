# rag-query-router

**Top k retrieval cannot count, cannot detect absence, and cannot compare magnitudes. This is the layer that can.**

A document question answering system that classifies each query by type and dispatches it to an engine capable of answering it, instead of sending every question down a retrieval path regardless of whether retrieval is able to answer it.

Measured against a naive RAG baseline on 24 golden queries over 200 contracts:

| Category | n | Routed | Naive RAG | Delta |
|---|---:|---:|---:|---:|
| aggregate (counting) | 4 | **1.000** | 0.000 | +1.000 |
| numeric (magnitude) | 4 | **1.000** | 0.056 | +0.945 |
| absence (negation) | 5 | **1.000** | 0.058 | +0.942 |
| multihop (traversal) | 3 | **1.000** | 0.269 | +0.731 |
| temporal (dates) | 3 | **1.000** | 0.275 | +0.725 |
| enumerate (completeness) | 2 | **1.000** | 0.280 | +0.720 |
| semantic *(control)* | 3 | 0.667 | 0.667 | +0.000 |
| **Structural mean** | | **1.000** | **0.156** | **+0.844** |

The semantic row is the control, and it is supposed to be a tie. Both systems route those queries down the same retrieval path, because retrieval is the correct tool for open ended clause lookup. A benchmark where the proposed system wins every category is a benchmark someone rigged. The claim here is narrower: retrieval is fine at the one thing retrieval is for, and structurally incapable at five things it is routinely asked to do anyway.

Complete answers: **21/24 routed, 0/24 baseline.** Latency p50 1.64 ms, p95 3.03 ms.

---

## The problem

Embedding retrieval has failure modes that are properties of the paradigm, not of model quality. Three root causes produce all of them.

**Embeddings measure topical similarity, not answer-hood.** A passage that talks *about* your question outranks the passage that *answers* it.

**Vector space has no logical or numerical operators.** There is no `not` and no `>`. "Contracts with a liability cap" and "contracts without a liability cap" produce nearly identical vectors, because the word "not" contributes almost nothing. Published work testing dense models from 2014 through 2024 reports the correct answer ranking **last** on negation queries — not low, last. Similarly, embeddings encode that a number is present, not its magnitude: `1,000,000` and `1,500,000` are not ordered.

**Top k is a sample, not a scan.** Any question whose answer depends on the whole corpus cannot be answered from five chunks. Raising k to 100 does not fix it; you need k equal to everything, at which point you are not retrieving, you are scanning and computing.

These are not tuning problems. Better chunking, hybrid search and reranking genuinely help with vocabulary mismatch and answer-hood — and this project uses all three in the baseline. They do nothing for counting, absence, or magnitude.

## The architecture

```
contracts ──> extraction ──> structured store (SQLite) ─┐
    │         (typed facts,     full corpus scans        │
    │          span citations)                           │
    │                                                    ├──> engine ──> answer
    └───────> chunking ────> hybrid index ───────────────┘              + citations
                             (dense + BM25, RRF)         ▲              + complete flag
                                                         │
                              query ──> router ──────────┘
                                        (type, field, operator, value)
```

The router emits an inspectable plan — type, field, operator, value, and the lexical signals that produced them. When an answer is wrong you can see whether the router misread the question or an engine mishandled a correct plan. That separation is what makes the system debuggable rather than a black box you re-prompt at.

**Engines.** `aggregate` computes counts, sums and averages over a full scan. `absence` enumerates then checks. `numeric` applies typed predicates. `temporal` filters typed dates. `multihop` performs dependent traversal. `enumerate` returns guaranteed-complete listings. `semantic` is ordinary retrieval, used only where it is correct.

**The `complete` flag.** Every engine except `semantic` sets `complete=True`, because every engine except `semantic` examined the full corpus. This single field is the honest difference between the two systems: a count taken over eight retrieved chunks is not a count, and nothing in a top k pipeline can tell the user so.

## Two design decisions worth explaining

**Absence is genuinely ambiguous, and the system says so.** A null value can mean the clause is truly absent, or that extraction missed it. Treating those as identical would be exactly the overconfidence this project argues against. So fields carry a companion *basis* column recording why a value is absent (`omitted`, `explicit_unlimited`, `unparsed`), and the absence engine gates on **determinacy** — the fraction of rows where the answer is settled — rather than on raw value coverage. Below an 80% floor it abstains and explains why. `test_absence_abstains_without_determinacy` removes the basis signal and asserts the abstention fires.

This matters concretely. In this corpus 84 contracts have no liability cap, in two textual shapes: 39 where the clause is omitted entirely, and 45 where a clause is present stating that no limitation applies. Semantic search surfaces neither. The first has no text to match. The second sits nearest in embedding space to the contracts that *do* cap, because it is a liability clause full of liability vocabulary. The system reports both, with the breakdown, and cites the disclaimer text for the 45 while correctly citing nothing for the 39 — there is no span to point at.

**The baseline is deliberately strong.** Beating a weak baseline would prove nothing, so it gets three concessions: clause-aware chunking (each chunk is a complete clause), hybrid dense plus BM25 retrieval with reciprocal rank fusion, and **perfect reading comprehension** over whatever it retrieved. Rather than asking an LLM to read the chunks, it is handed the already extracted structured facts for the contracts its chunks came from. It never misreads, never hallucinates, never misparses an amount. Every failure it records is attributable to retrieval alone. A real LLM pipeline would score at or below these numbers, never above — **the measured gap is a lower bound.**

## Running it

```bash
pip install -r requirements.txt
python data/generate_corpus.py --n 200 --seed 7   # corpus + ground truth
python -m pytest tests/ -q                        # 18 tests
python -m app.eval.run_eval --corpus data/corpus  # the table above
uvicorn app.main:app --port 8000                  # UI at localhost:8000
```

No API key and no model download required for any of the above. `make docker` builds a single container serving API and UI on one port.

The UI runs both systems on the same question side by side and shows the routing plan, the completeness badge, per item citations with character offsets, and latency.

## Evaluation methodology

Ground truth is **constructed, not annotated**. The corpus generator writes each contract's facts to a hidden record and renders prose from it, so "84 contracts have no liability cap" is true by construction with zero label noise. Golden queries are then computed from those records, so regenerating with a different seed updates every expected answer automatically.

The prose is not a fixed template — amounts appear as `$500,000`, `$500K` and `Five Hundred Thousand Dollars ($500,000)`; dates in three formats; each clause has multiple phrasings; and absence takes two distinct textual forms. Extraction has to parse language rather than match a pattern.

Scoring: counts by exact match, sets by precision/recall/F1, and semantic controls by **precision@k only**. Recall is the wrong metric for a clause lookup — "what does the indemnification clause say" is well answered by eight relevant contracts, and scoring it against all 139 contracts containing the clause would report 0.04 for a retrieval that was 100% relevant. Abstentions score 0 but are tallied separately, so declining to answer is never conflated with answering wrongly.

CI runs the suite and gates the build at 0.95 structural accuracy, so a change that quietly degrades the engines cannot merge while this README claims 1.000.

## Limitations

**Extraction is pattern based in the default path.** It scores 100% on all 16 fields here because it was written against this corpus's drafting conventions. That number describes the extractor's fit to this corpus, not its generality — on real contracts with unseen phrasing, recall degrades. `LLMExtractor` exists for that case and emits the identical `Fact` schema, so routing, engines and eval are untouched by the swap. The honest reading of the results table is that it isolates *routing and execution*, having removed extraction as a variable.

**Synthetic corpus.** Real contracts are messier: defined terms, cross references, exhibits, inconsistent numbering, OCR noise. A CUAD adapter is the natural next step, and its 41 expert-annotated clause types would let extraction be measured against expert labels rather than construction.

**Rule based routing.** Classification signals here — negation markers, comparison operators, aggregation verbs — are lexically explicit in English, so a rule layer is accurate, free, instant and auditable. Paraphrase-heavy production traffic would need an LLM classifier behind the same interface. The failure mode to watch is a misroute sending a structural query to the semantic engine, which is why the plan is exposed at `/api/plan`.

**TF-IDF fallback in this environment.** Model downloads were unavailable, so the baseline uses TF-IDF. Dense embeddings handle paraphrase better and would raise the semantic control row — and change nothing structural, since there is no `not` and no `>` in either vector space, and top k is a sample in both. Set `EMBEDDINGS=st` to force the dense path.

## What I would build next

Calibrated confidence on the semantic path, with reliability diagrams and expected calibration error, so "80% confident" can be verified as correct 80% of the time. Fine tuning an open model for extraction — ContractEval found open models exhibit an *avoidance problem* on legal clause extraction, answering "no related clause" when the clause demonstrably exists, which is measurable headroom against a public baseline. And a temporal layer for contract versions, since "what does our current agreement say" over six amendments is the same structural failure in a different dimension.

## Layout

```
app/
  models.py           typed schemas, citations, Answer with complete flag
  pipeline.py         wiring
  main.py             FastAPI, single deployable
  ingest/extract.py   extraction, deterministic + LLM providers
  index/structured.py SQLite fact store, full scans, determinacy
  index/retrieval.py  chunking, embeddings with fallback, hybrid + RRF
  router/classify.py  query -> inspectable plan
  engines/core.py     seven engines + dispatcher
  baseline/naive_rag.py
  eval/golden.py      golden set derived from ground truth
  eval/run_eval.py    scoring, per-category report, CI gate
data/generate_corpus.py
static/index.html     side by side comparison UI
tests/test_system.py  correctness + characterisation
```

**References:** [Predictable Failure Modes of RAG Retrieval](https://towardsdatascience.com/embeddings-arent-magic-the-predictable-failure-modes-of-rag-retrieval-enterprise-document-intelligence-vol-1-2/) · [ContractEval (arXiv 2508.03080)](https://arxiv.org/abs/2508.03080) · [CUAD](https://www.atticusprojectai.org/cuad)
