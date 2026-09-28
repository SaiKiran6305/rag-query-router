# Design notes

Six decisions in this codebase are not obvious from reading the code, and each
one has a wrong version that looks more reasonable than the right one. This
documents why each is the way it is.

---

## 1. The absence engine gates on determinacy, not coverage

**Location:** `app/engines/core.py`, `_determinacy()` and `AbsenceEngine.run()`

**The problem.** "Which contracts have no liability cap?" is answered from null
values in the structured store. But a null is ambiguous. It can mean:

- the contract genuinely has no cap (the answer the user wants), or
- extraction failed to parse a cap that is there (a bug, reported as a finding)

A system that cannot tell these apart and answers anyway is reporting its own
failures as facts about the world. That is precisely the overconfidence this
project exists to argue against, so the engine has to refuse in that case.

**The obvious fix, which is wrong.** Gate on *field coverage* -- the fraction of
rows with a non-null value -- and abstain when it is low. This fails badly here:

```
liability_cap_usd coverage = 58%
```

58% looks like broken extraction. It is not. It is 42% of contracts genuinely
having no cap, which is the thing being asked about. Gating on coverage makes
the engine abstain on exactly the question it can answer perfectly, and the
abstention rate rises as the true answer set grows. The metric is anti-correlated
with the thing it is supposed to measure.

**What it does instead.** Fields carry a companion *basis* column recording
**why** a value is absent:

| basis | meaning | null is... |
|---|---|---|
| `capped` | clause found, amount parsed | not null |
| `explicit_unlimited` | clause found, states no limit applies | a determined finding |
| `omitted` | no liability clause in the document | a determined finding |
| `unparsed` | clause found, amount could not be read | an extraction failure |
| `unknown` | document structure not recognisable (see section 5) | undetermined |

Determinacy is the fraction of rows whose basis is *not* `unparsed`. Here that is
100%, so the engine answers. Strip the basis column and it abstains -- which is
the correct behaviour, and is what `test_absence_abstains_without_determinacy`
asserts by removing `BASIS_FIELDS` at runtime.

**The generalisation.** Any field where "no value" is a meaningful answer needs a
basis column. `confidentiality_years` has one for the same reason. Fields where
null only ever means "we did not extract it" fall back to coverage, which is
correct for them.

**Cost.** Extraction must distinguish "absent" from "present but unreadable",
which is more work than returning `None` for both. That is the price of being
able to answer absence questions at all.

---

## 2. Inclusive comparators are matched before strict ones

**Location:** `app/router/classify.py`, the `COMPARATORS` list

**The bug this prevents.** Pattern order in a list of regexes is usually
cosmetic. Here it silently changes answers:

```python
COMPARATORS = [
    (r"...greater than|above|over...", "gt"),    # if this is first
    (r"...at least|at or above...",    "gte"),   # this never matches
]
```

The query *"liability capped at or above $2,000,000"* contains the substring
**"above"**. With strict patterns first it routes to `gt`, and every contract
sitting exactly on $2,000,000 is dropped from the result.

**Why it is dangerous rather than merely wrong.** The failure is silent and
plausible. You get a smaller answer set, not an error. Measured on the golden
set this scored `P=1.00 R=0.47` -- perfect precision, half the recall. Precision
staying at 1.00 is what makes it easy to miss: every returned row is correct, so
spot-checking the output finds nothing wrong. Only comparison against a known
total exposes it.

**The fix.** Inclusive forms (`gte`, `lte`) are tested first, because they
contain their strict counterparts as substrings and never the reverse.
`test_inclusive_comparator_beats_strict` pins all four operators so a future
reordering fails the build.

**Generalisation.** In any ordered pattern matcher, a pattern that is a superstring
of another must be tested first. The same trap exists in the field lexicon:
`liability cap` is listed before a bare `liability`, for identical reasons.

---

## 3. Semantic controls are scored on precision@k, not F1

**Location:** `app/eval/run_eval.py`, the `answer_kind == "retrieval"` branch

**Why the semantic queries are in the benchmark at all.** Both systems route
them down the same retrieval path, so they must tie. That tie is the credibility
anchor of the whole results table: it shows the comparison is not constructed to
win. A benchmark where the proposed system wins every category is one someone
rigged.

**The scoring bug this fixes.** Semantic queries were originally scored as sets,
against every contract containing the clause. For *"what does the indemnification
clause say?"*:

```
retrieved 8 contracts, all 8 relevant
ground truth set = 139 contracts
precision 1.00 - recall 0.02 - F1 0.04
```

A retrieval that was **100% relevant** scored 0.04. That is not a weak result,
it is the wrong measurement: recall against the full set penalises the system for
not returning all 139 contracts, which no user asking that question wants.

**The fix.** Retrieval-style queries are scored on precision@k only -- of what was
returned, how much was relevant. Recall is reported as `None` rather than 0, so
the report does not imply a number that was never measured.

**Why this is not tilting the scale.** The change raises both systems identically
(0.059 -> 0.667 each) and leaves the delta at exactly zero. It corrects a
methodology error in the control group, and the control still controls. If it had
moved the routed system and not the baseline, it would be cheating.

**The general point.** Choosing a metric that cannot express success for the task
is a more common failure than choosing a model that cannot achieve it. Ask what
a perfect answer looks like before picking the number that scores it.

**The limit of this fix.** Contract-level precision@k is still generous: a
retrieved passage counts if its *contract* has the clause anywhere, and 139 of
200 contracts have an indemnification clause. Once the semantic engine began
showing the passages it retrieved, it was visible that "what does the
indemnification clause say?" returned eight TERM clauses -- scored 1.00. The
cause was a BM25 tokenizer that split on whitespace, so `INDEMNIFICATION.` never
matched `indemnification`. The report now also prints passage-level precision
(is the retrieved passage the clause asked about?), which is the number that
would have caught it.

---

## 4. The abstention gate uses raw similarity, not the fused score

**Location:** `app/index/retrieval.py`, `HybridIndex.relevance()`, and
`SemanticEngine.run()` in `app/engines/core.py`

**The problem.** Every question the router did not recognise fell through to
retrieval, and retrieval always returns k results. *"What is the weather in
Dallas?"* returned eight contracts. The engine needs to decline when nothing in
the corpus is relevant.

**The obvious fix, which is wrong.** Threshold the score `search()` already
returns. That score comes from reciprocal rank fusion, and RRF is a function of
*rank only*: `1/(60 + rank)` from each retriever, summed. The top hit gets about
`2/61` whether it is a perfect match or merely the least-bad chunk:

```
                                              top fused   top raw cosine
best biryani recipe                             0.0333        0.000
what does the indemnification clause say?       0.0333        0.145
explain the confidentiality obligations         0.0167        0.203
```

A gate on the fused score would reject the confidentiality question and accept
the biryani recipe.

**What it does instead.** `relevance()` returns the best raw cosine similarity
between the query and any chunk, which still carries magnitude. The engine also
declines any question that asks for a count it could not map to a field, since
a sample is not a count however relevant it is.

**Cost.** The threshold is backend- and corpus-dependent. It is calibrated for
TF-IDF here; dense embeddings never score exactly zero, and on real contracts
even "weather" finds force-majeure clauses.

---

## 5. "Omitted" has to be earned

**Location:** `app/ingest/extract.py`, `MIN_HEADINGS_FOR_ABSENCE`, and the
determinacy gate in `AbsenceEngine`

**The problem.** Decision 1 made the absence engine trust `basis = omitted` as a
settled finding. That is only honest if the extractor's "omitted" is honest.

**How it failed.** On CUAD's 510 real contracts the deterministic extractor
marked 490 liability clauses `omitted` -- not because the clauses were missing,
but because it looks for numbered headings like `4. LIMITATION OF LIABILITY.`
and 436 of the 510 contracts have none. Lawyers had found a liability clause in
255 of those 490. The absence engine, seeing 96% determinacy, would have
answered "which contracts have no liability cap?" with 490 contracts, labelled
complete.

**The obvious fix, which is wrong.** Lower the confidence of every null, or
raise the determinacy floor. Both make the engine abstain on the synthetic
corpus too, where "omitted" is exactly right.

**What it does instead.** The extractor claims absence only when it can see the
document's clause structure (at least five recognisable headings; every
synthetic contract has six or more). Otherwise the basis is `unknown` and
booleans are `null` rather than `False`. Determinacy on CUAD drops to 5%, the
engine abstains, and false "omitted" claims fall from 255 to 13. Boolean fields
are now gated on determinacy too, because a `null` boolean now means "could not
tell", not "no".

**The general point.** An abstention mechanism inherits the honesty of whatever
feeds it. "I looked and it isn't there" and "I couldn't look" must be different
values all the way down.

---

## 6. Paraphrase evaluation scores route, plan and answer separately, with a held-out split

**Location:** `app/eval/paraphrase.py`

**The problem.** The golden set's phrasings are the ones the rules were written
for, so its 1.000 measures fit. A robustness eval is needed, and it is easy to
build one that flatters.

**Two ways it flatters.** First, scoring only the route: *"contracts that
started in 2025"* routed to the temporal engine -- correct -- then filtered the
expiry date and returned 21 instead of 78, as a complete scan. Seven of the dev
failures before the fix were right-engine-wrong-answer (six date questions and
"how many Texas contracts auto-renew?", which dropped the state). Scoring route, plan (engine + field +
operator) and answer separately exposes them, and the report counts
*confident-wrong* answers: wrong, yet labelled complete.

Second, tuning against the test. The paraphrases were written in two splits
*before* any router change; the rules were tuned against dev failures only, and
the held-out failures are listed in the README rather than fixed. The held-out
answer accuracy (0.593 -> 0.869) is the number to quote; dev (-> 1.000) is not.
CI gates the dev split for regressions and deliberately does not gate the
held-out split, which would invite tuning against it.

**Cost.** One author wrote both the rules and the paraphrases, so both splits
are easier than real traffic. Paraphrases from other people would be harder.

---

## What these have in common

Each is a case where the obvious implementation produces a number that looks
fine. Coverage-gating abstains politely. Strict-first comparators return
precision 1.00. Set-F1 reports a low score you might accept as "retrieval is
just weak here." A fused-score gate looks like a threshold. An extractor's
"omitted" looks like a finding. A route-only robustness score looks like a pass.
None of them throws an error, and all six were found by checking results
against a ground truth -- constructed, or labelled by lawyers -- rather than
against whether the output looked reasonable.

That is the argument for constructed ground truth generally: it is the only
setup where "the system returned 15 rows" can be checked against "the true
answer is 32" rather than against whether the 15 look reasonable.
