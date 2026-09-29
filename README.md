# rag-query-router

**Top k retrieval cannot count, cannot detect absence, and cannot compare magnitudes. This is the layer that can.**

A document question answering system that classifies each query by type and dispatches it to an engine capable of answering it, instead of sending every question down a retrieval path regardless of whether retrieval is able to answer it.

Measured against a naive RAG baseline on 24 golden queries over 200 contracts:

| Category | n | Routed | Naive RAG | Delta |
|---|---:|---:|---:|---:|
| aggregate (counting) | 4 | **1.000** | 0.000 | +1.000 |
| absence (negation) | 5 | **1.000** | 0.035 | +0.965 |
| numeric (magnitude) | 4 | **1.000** | 0.100 | +0.900 |
| multihop (traversal) | 3 | **1.000** | 0.321 | +0.679 |
| temporal (dates) | 3 | **1.000** | 0.324 | +0.676 |
| enumerate (completeness) | 2 | **1.000** | 0.377 | +0.623 |
| semantic *(control)* | 3 | 0.917 | 0.917 | +0.000 |
| **Structural mean** | | **1.000** | **0.193** | **+0.807** |

The semantic row is the control, and it is supposed to be a tie. Both systems route those queries down the same retrieval path, because retrieval is the correct tool for open ended clause lookup. A benchmark where the proposed system wins every category is a benchmark someone rigged. The claim here is narrower: retrieval is fine at the one thing retrieval is for, and structurally incapable at five things it is routinely asked to do anyway.

Complete answers: **21/24 routed, 0/24 baseline.** Latency p50 1.00 ms, p95 2.20 ms.

> **These numbers moved in v1.1, in the baseline's favour.** The baseline's BM25 tokenizer split on whitespace, so the heading `INDEMNIFICATION.` became the token `indemnification.` and never matched a query. Fixing it made the baseline stronger (structural 0.156 → 0.193) and the gap smaller (+0.844 → +0.807). A benchmark whose comparison is only as strong as its weakest bug is worth fixing even when the fix costs you points. Details in [What changed in v1.1](#what-changed-in-v11).

The golden set measures *fit* — its questions use the phrasings the router was written for. How the router does on questions phrased the way people actually ask them is measured separately, with a held-out set, in [Robustness](#robustness-paraphrased-and-out-of-scope-questions).

### ▶ Live demo — **[rag-query-router.onrender.com](https://rag-query-router.onrender.com)**

Ask it *"which contracts have no liability cap?"* and watch both systems answer side by side. Then ask it *"what is the weather in Dallas?"* and watch the routed system decline.

> Hosted on a free tier that sleeps when idle, so the **first load takes 30–50 seconds** while the container wakes and rebuilds its index. Everything after that is single-digit milliseconds.

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/SaiKiran6305/rag-query-router)

Deploys in one click from the checked-in `render.yaml`. `railway.json` and `fly.toml` are included too, and the Dockerfile binds `$PORT` so it runs unmodified on any of them. Nothing to configure: no API key, no model download, no external services.

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
    │         (typed facts,    parameterized SQL,       │
    │          span citations)  full-table scans        │
    │                                                   ├──> engine ──> answer
    └───────> chunking ────> hybrid index ──────────────┘              + citations
                             (TF-IDF/MiniLM + BM25, RRF)  ▲              + complete flag
                                                          │              + or: declined, and why
            query ──> rule router ──(doubtful fall-through)──> LLM planner ──> validate
                        │                                                       │
                        └──────────────── plan (type, field, operator, value, ◄─┘
                                                scope filters, confidence, tier)

                      app/schema.py: one registry of fields, types, synonyms and
                      legal operators, read by the router, the SQL compiler and the
                      LLM validator alike
```

The router emits an inspectable plan — type, field, operator, value, scope filters, the lexical signals that produced them, any defaults it assumed, and a confidence. When an answer is wrong you can see whether the router misread the question or an engine mishandled a correct plan. That separation is what makes the system debuggable rather than a black box you re-prompt at.

**Engines.** `aggregate` computes counts, sums, averages and percentages in SQL. `absence` enumerates then checks. `numeric` applies typed comparisons. `temporal` handles before, after, on-or-before, on-or-after and between over typed dates, including quarters, months and relative ranges ("next 90 days"). `multihop` performs dependent traversal with parent and child type constraints. `enumerate` returns guaranteed-complete listings. `semantic` is ordinary retrieval, used only where it is correct — and now declines when it cannot be.

**The `complete` flag.** Every structural engine sets `complete=True`, because every structural engine answers through one parameterized query over the whole table. This single field is the honest difference between the two systems: a count taken over eight retrieved chunks is not a count, and nothing in a top k pipeline can tell the user so. `complete` means every row was checked; when some rows have no extracted value for the field, the explanation says how many.

**One predicate compiler, both sides of the benchmark.** Engines and the baseline both answer through `StructuredStore.select()`, which compiles a plan to a parameterized `WHERE` clause with column names checked against the schema registry and operators against a whitelist. The two systems therefore cannot disagree about what a predicate *means*, only about which rows they looked at — and user text only ever reaches SQLite as a bound parameter (`test_user_text_is_only_ever_a_parameter`).

## Design decisions worth explaining

> Fuller rationale for the non-obvious decisions in this codebase — including the
> wrong version of each, which looks more reasonable than the right one — is in
> [DESIGN_NOTES.md](DESIGN_NOTES.md).

**Absence is genuinely ambiguous, and the system says so.** A null value can mean the clause is truly absent, or that extraction missed it. Treating those as identical would be exactly the overconfidence this project argues against. So fields carry a companion *basis* column recording why a value is absent (`omitted`, `explicit_unlimited`, `unparsed`, `unknown`), and the absence engine gates on **determinacy** — the fraction of rows where the answer is settled — rather than on raw value coverage. Below an 80% floor it abstains and explains why.

This matters concretely. In this corpus 84 contracts have no liability cap, in two textual shapes: 39 where the clause is omitted entirely, and 45 where a clause is present stating that no limitation applies. Semantic search surfaces neither. The first has no text to match. The second sits nearest in embedding space to the contracts that *do* cap, because it is a liability clause full of liability vocabulary. The system reports both, with the breakdown, and cites the disclaimer text for the 45 while correctly citing nothing for the 39 — there is no span to point at.

**"Omitted" has to be earned.** The determinacy gate is only as honest as the extractor's claim that a clause is omitted. Run on real contracts (below), the extractor marked 490 of 510 contracts "omitted" because it could not find its expected headings — and lawyers had found a liability clause in 255 of them. The absence engine would have reported all 490 as uncapped, labelled complete. The extractor now claims absence only when it can see the document's clause structure; otherwise the basis is `unknown`, determinacy drops, and the engine abstains. False "omitted" claims on CUAD fell from 255 to 13.

**Semantic has to earn its route, and may decline.** Falling through to retrieval used to be the default for anything unrecognised, so *"what is the weather in Dallas?"* returned eight arbitrary contracts. Now the semantic engine declines when the question asks for a count (a sample is not a count) or when nothing in the corpus is relevant. The relevance gate uses **raw similarity, not the fused score**: reciprocal rank fusion depends only on rank, so *"best biryani recipe"* receives the same top fused score (0.033) as *"what does the indemnification clause say?"* and a higher one than *"explain the confidentiality obligations"* (0.017). Raw cosine separates them cleanly (0.000 vs 0.145–0.203).

**The baseline is deliberately strong.** Beating a weak baseline would prove nothing, so it gets three concessions: clause-aware chunking (each chunk is a complete clause), hybrid dense plus BM25 retrieval with reciprocal rank fusion, and **perfect reading comprehension** over whatever it retrieved. Rather than asking an LLM to read the chunks, it is handed the already extracted structured facts for the contracts its chunks came from, and applies the router's plan to them through the same SQL compiler the engines use. It never misreads, never hallucinates, never misparses an amount. Every failure it records is attributable to retrieval alone. A real LLM pipeline would score at or below these numbers, never above — **the measured gap is a lower bound.**

## Robustness: paraphrased and out-of-scope questions

The golden questions use the phrasings the rules were written for. `app/eval/paraphrase.py` asks the same kinds of questions the way people do — *"which agreements are uncapped?"*, *"contracts that started in 2025"*, *"how many Texas contracts auto-renew?"* — plus questions the corpus cannot answer, and scores three layers separately:

- **route** — right engine?
- **plan** — right engine, field *and* operator?
- **answer** — correct against ground truth?

Route can be right while the answer is wrong, and that is the dangerous case: *"contracts that started in 2025"* reached the temporal engine, filtered the **expiry** date, returned 21 instead of 78, and was labelled a complete scan. The report counts these as **confident-wrong**.

The set was written in two splits **before any router change**: a dev split the rules were tuned against, and a held-out split that was never used for tuning.

| | route | plan | answer | confident-wrong | out-of-scope declined |
|---|---:|---:|---:|---:|---:|
| dev, before | 0.743 | 0.629 | 0.602 | 9 | 0 of 5 |
| dev, after | 1.000 | 1.000 | 1.000 | 0 | 5 of 5 |
| **held-out, before** | 0.731 | 0.615 | 0.593 | 6 | 0 of 4 |
| **held-out, after** | **0.923** | **0.885** | **0.869** | **4** | **1 of 4** |

The dev row is 1.000 because it was tuned against; the held-out row is the honest number. Remaining held-out failures, deliberately left unfixed so that number stays honest:

- *"which contracts **cannot** be assigned?"* — "cannot" is not a negation word, so it lists the opposite set
- *"agreements that **began** in 2024"* — "began" is not mapped to the effective date
- *"the **second half** of 2026"* — not parsed as a date window
- *"SOWs that **fall under** … master services agreements"* — not recognised as multi-hop
- three out-of-scope questions still answered, including *"how many **vendors** are headquartered in Europe?"*, which matches the vendor field and counts every contract

To fix these honestly: fix them, then write a *new* held-out split. Both splits were written by the same author as the rules, which flatters them; paraphrases from other people, or sampled from an LLM, would be a harder test.

### The LLM tier

`ROUTER=hybrid` (with `OPENAI_API_KEY`) adds an LLM planner behind the same interface. It is consulted **only** when the rules fall through to semantic with a reason for doubt — a count, negation, comparison or date they could not map, or no evidence that this is a clause-reading question — so cost scales with how unusual the traffic is, not how much of it there is. Calls are cached.

The LLM never writes SQL and never answers. It proposes a plan as JSON, and `validate()` checks every part against the schema registry before anything runs: the field must exist, the operator must be legal for the field's type, values are coerced and range-checked, states and agreement types must match their closed vocabularies. A hallucinated `employee_count` column is rejected there, not discovered as a wrong answer later. The LLM may also say the question is outside the schema, and the system declines.

What this tier **cannot** fix is a confident rule misroute: if the rules produce a structural plan, the LLM is never asked. *"How many vendors have more than 500 staff?"* is pinned by `test_confident_misroute_is_not_escalated` as a known limit. The whole tier is tested with an injected fake, so the suite needs no key.

## On real contracts (CUAD)

`app/ingest/cuad.py` exports the [CUAD](https://www.atticusprojectai.org/cuad) dataset — 510 real commercial contracts with lawyers' labels for 41 clause types — into this system's corpus layout, so everything runs on it unchanged (`make cuad`, then `CORPUS_DIR=data/cuad uvicorn app.main:app`). `app/eval/extraction_eval.py` scores extraction against the labels. CUAD marks *whether* a clause is present, so fields are scored on detection:

| CUAD category | labelled present | precision | recall |
|---|---:|---:|---:|
| Governing Law | 437 | 0.994 | **0.723** |
| Expiration Date | 413 | 0.927 | 0.092 |
| Cap On Liability | 275 | 1.000 | 0.073 |
| Renewal Term | 176 | 1.000 | 0.062 |
| Insurance | 166 | 1.000 | 0.054 |
| Anti-Assignment | 374 | 1.000 | 0.051 |
| Liquidated Damages | 61 | 0.000 | 0.000 |

The deterministic extractor scores 19/19 fields at 100% on the synthetic corpus and **5–9% recall** on most fields of real contracts. When it claims a clause it is almost always right; it simply cannot find most of them, because it was written against the synthetic corpus's numbered-heading convention and 436 of CUAD's 510 contracts use none. That is the measured case for the `LLMExtractor` (`python -m app.eval.extraction_eval --cuad data/cuad --extractor llm`), which emits the identical schema so nothing downstream changes.

Two things the system now does right on real contracts that it did not before: absence questions **abstain** instead of reporting extraction misses as findings (determinacy 5% for liability, 8% for insurance), and the index **builds at all** — the TF-IDF matrix used to be densified, which on CUAD's 35,616 chunks would need 205 GB. It is now sparse: 13 seconds, 615 MB.

## Running it

```bash
pip install -r requirements.txt
python data/generate_corpus.py --n 200 --seed 7       # corpus + ground truth
python -m pytest tests/ -q                            # 71 tests
python -m app.eval.run_eval --corpus data/corpus      # the tables above
python -m app.eval.paraphrase --show holdout          # robustness, with failures
python -m app.eval.extraction_eval --synthetic data/corpus
uvicorn app.main:app --port 8000                      # UI at localhost:8000

make cuad && make cuad-eval                           # real contracts (18 MB download)
ROUTER=hybrid OPENAI_API_KEY=... uvicorn app.main:app # with the LLM tier
```

No API key and no model download required for anything except the two optional LLM paths. `make docker` builds a single container serving API and UI on one port.

The UI runs both systems on the same question side by side and shows the routing plan (tier, confidence, scope, assumptions), the completeness or *declined* badge, per item citations with character offsets — retrieved passages for semantic answers — and latency. `/api/schema` lists every field the system can answer about.

## Evaluation methodology

Ground truth is **constructed, not annotated**. The corpus generator writes each contract's facts to a hidden record and renders prose from it, so "84 contracts have no liability cap" is true by construction with zero label noise. Golden queries are then computed from those records, so regenerating with a different seed updates every expected answer automatically.

The prose is not a fixed template — amounts appear as `$500,000`, `$500K` and `Five Hundred Thousand Dollars ($500,000)`; dates in three formats; each clause has multiple phrasings; and absence takes two distinct textual forms. Extraction has to parse language rather than match a pattern.

Scoring: counts by exact match, sets by precision/recall/F1 (an empty answer to a question whose true answer is empty scores 1.0 — it used to score 0), and semantic controls by **precision@k only**. Recall is the wrong metric for a clause lookup — "what does the indemnification clause say" is well answered by eight relevant contracts, and scoring it against all 139 contracts containing the clause would report 0.04 for a retrieval that was 100% relevant. Abstentions score 0 but are tallied separately, so declining to answer is never conflated with answering wrongly.

That precision@k is measured at the *contract* level, which is generous: a retrieved passage counts if its contract has the clause anywhere. The report now also prints **passage-level** precision — is the retrieved passage the clause asked about? With the TF-IDF fallback it is **0.333**: indemnification 8/8, but confidentiality and termination 0/8, because the liability clause literally says "confidentiality … obligations" and the confidentiality clause literally says "termination". This is the lexical-retrieval weakness dense embeddings exist to fix, and the number to watch with `EMBEDDINGS=st`.

CI runs the suite and gates the build at 0.95 structural accuracy and 0.95 on the dev paraphrase split, so a change that quietly degrades the engines or the router cannot merge. The held-out split is reported, never gated — gating it would invite tuning against it.

## Limitations

**Extraction is pattern based in the default path.** 19/19 fields at 100% here, 5–9% recall on real contracts (see CUAD above). The honest reading of the main results table is that it isolates *routing and execution*, having removed extraction as a variable. `LLMExtractor` is the path for real contracts and emits the identical `Fact` schema.

**Synthetic corpus for the headline numbers.** Real contracts are messier: defined terms, cross references, exhibits, inconsistent numbering, OCR noise. The CUAD adapter makes that measurable; the headline benchmark still uses constructed ground truth because CUAD labels presence, not typed values.

**Rule based routing, with an optional LLM tier.** Rules are accurate, free, instant and auditable on explicit phrasings, and miss paraphrases they were not written for (held-out answer accuracy 0.869). The LLM tier catches fall-throughs but not confident misroutes.

**Compound questions are limited.** A question may add a governing-law or agreement-type scope to its main predicate (*"Texas contracts that auto-renew"*), but not combine two arbitrary predicates (*"penalties over $100K expiring in 2026"*).

**TF-IDF fallback in this environment.** Model downloads were unavailable, so retrieval uses TF-IDF. Dense embeddings handle paraphrase better and would raise passage-level precision on the semantic control — and change nothing structural, since there is no `not` and no `>` in either vector space, and top k is a sample in both. Set `EMBEDDINGS=st` to force the dense path, and re-tune `SEMANTIC_MIN_RELEVANCE`: the out-of-scope gate is calibrated for TF-IDF, and on real contracts even "weather" finds force-majeure clauses.

## What changed in v1.1

- **Robustness eval** with dev and held-out splits, scored on route, plan and answer; out-of-scope questions must be declined.
- **Router:** before/after/on-or-before/on-or-after dates, quarters by name, months, relative ranges; start-date synonyms; lexicon gaps (uncapped, unlimited liability, renew on their own, state names, agreement types, indemnity); scope filters; confidence and unresolved-intent on every plan. Two existing bugs fixed: *"no less than $50,000"* routed to absence, and *"total number of"* summed dollar amounts instead of counting.
- **Schema registry** (`app/schema.py`): one declaration per field, from which the SQL DDL, the lexicon, the type sets, the LLM prompt and the LLM validator are derived.
- **Parameterized SQL** for every engine and the baseline; the unused f-string `query(where=...)` is gone.
- **Semantic engine** returns the passages it retrieved and declines counts and irrelevant questions.
- **LLM router tier** with schema validation, tested without a key.
- **CUAD adapter** and extraction eval; `unknown` instead of `omitted` on unreadable documents; boolean absence gated on determinacy too; sparse TF-IDF.
- **Fixed** the baseline's BM25 tokenizer, canonical agreement type names (`"Statement Of Work"` → `"Statement of Work"`), and the empty-set scoring bug.

## What I would build next

Paraphrases from outside the author, and a fresh held-out split after fixing the current one's failures. Running `LLMExtractor` against CUAD to put a real number on its recall. Calibrated confidence on the semantic path, with reliability diagrams and expected calibration error, so "80% confident" can be verified as correct 80% of the time. Fine tuning an open model for extraction — ContractEval found open models exhibit an *avoidance problem* on legal clause extraction, answering "no related clause" when the clause demonstrably exists, which is measurable headroom against a public baseline and exactly the failure the `unknown` basis now guards against. And a temporal layer for contract versions, since "what does our current agreement say" over six amendments is the same structural failure in a different dimension.

## Layout

```
app/
  schema.py            field registry: types, synonyms, legal operators, LLM catalog
  models.py            typed schemas, citations, Answer with complete flag
  pipeline.py          wiring
  main.py              FastAPI, single deployable
  ingest/extract.py    extraction, deterministic + LLM providers
  ingest/cuad.py       CUAD -> corpus layout + expert labels
  index/structured.py  SQLite fact store, parameterized predicate compiler
  index/retrieval.py   chunking, embeddings with fallback, hybrid + RRF, relevance
  router/classify.py   rule router: query -> inspectable plan
  router/llm.py        optional LLM tier: propose, validate, escalate
  engines/core.py      seven engines + dispatcher
  baseline/naive_rag.py
  eval/golden.py       golden set derived from ground truth
  eval/paraphrase.py   robustness: dev/held-out paraphrases + out-of-scope
  eval/extraction_eval.py  per-field extraction accuracy, synthetic and CUAD
  eval/run_eval.py     scoring, per-category report, CI gates
data/generate_corpus.py
static/index.html      side by side comparison UI
tests/test_system.py   correctness + characterisation
tests/test_robustness.py  router, SQL compiler, abstention, LLM tier, CUAD
DESIGN_NOTES.md        why the non-obvious decisions are what they are
```

**References:** [Predictable Failure Modes of RAG Retrieval](https://towardsdatascience.com/embeddings-arent-magic-the-predictable-failure-modes-of-rag-retrieval-enterprise-document-intelligence-vol-1-2/) · [ContractEval (arXiv 2508.03080)](https://arxiv.org/abs/2508.03080) · [CUAD](https://www.atticusprojectai.org/cuad)
