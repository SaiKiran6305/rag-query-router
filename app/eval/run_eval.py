"""
Evaluation harness: routed system vs naive RAG, per query category.

Scoring
-------
Scalar answers (counts) are scored by exact match, with relative error recorded
alongside so a near miss is distinguishable from a wild one.

Set answers are scored by precision, recall and F1 against the ground-truth id
set.

An abstention scores 0.0 but is tallied separately. That matters: a system that
declines to answer when it cannot distinguish a genuine absence from an
extraction miss is behaving correctly, and conflating that with a wrong answer
would punish exactly the behaviour this project argues for. The report shows
both numbers.

Run:
    python -m app.eval.run_eval --corpus data/corpus --out eval_report.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import re

from app.eval.golden import GoldenQuery, build_golden_set
from app.models import Answer, QueryType
from app.pipeline import ContractIntelligence


def _as_set(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set)):
        return {str(v) for v in value}
    return set()


def score(answer: Answer, g: GoldenQuery) -> dict[str, Any]:
    if answer.abstained:
        return {"score": 0.0, "abstained": True, "precision": 0.0, "recall": 0.0,
                "exact": False, "detail": "abstained"}

    if g.answer_kind == "retrieval":
        # Scored on precision@k only. Recall is the wrong metric for a clause
        # lookup: "what does the indemnification clause say" is answered well by
        # eight relevant contracts, and penalising a system for not returning all
        # 139 contracts that contain the clause would measure the wrong thing.
        # Set-F1 here would report 0.04 for a retrieval that was 100% relevant.
        got_set = _as_set(answer.value)
        want_set = set(g.expected)
        if not got_set:
            return {"score": 0.0, "abstained": False, "exact": False,
                    "precision": 0.0, "recall": 0.0, "detail": "returned nothing"}
        prec = len(got_set & want_set) / len(got_set)
        return {"score": round(prec, 4), "abstained": False, "exact": prec == 1.0,
                "precision": round(prec, 4), "recall": None,
                "n_got": len(got_set),
                "detail": f"precision@{len(got_set)}={prec:.2f}"}

    if g.answer_kind == "scalar":
        got = answer.value if isinstance(answer.value, (int, float)) else len(_as_set(answer.value))
        want = g.expected
        exact = got == want
        rel_err = abs(got - want) / max(abs(want), 1)
        return {"score": 1.0 if exact else 0.0, "abstained": False,
                "exact": exact, "got": got, "want": want,
                "relative_error": round(rel_err, 4),
                "precision": 1.0 if exact else 0.0, "recall": 1.0 if exact else 0.0,
                "detail": f"got {got}, want {want}"}

    got_set = _as_set(answer.value)
    want_set = set(g.expected)
    if not got_set and not want_set:
        # "None match" is a correct answer when none match. Scoring it 0 (as
        # P = 0/0 -> 0 used to) punished the right answer to an empty question.
        return {"score": 1.0, "abstained": False, "exact": True, "precision": 1.0, "recall": 1.0,
                "n_got": 0, "n_want": 0, "detail": "P=1.00 R=1.00 F1=1.00 (0 returned, 0 true)"}
    tp = len(got_set & want_set)
    prec = tp / len(got_set) if got_set else 0.0
    rec = tp / len(want_set) if want_set else (1.0 if not got_set else 0.0)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"score": round(f1, 4), "abstained": False, "exact": got_set == want_set,
            "precision": round(prec, 4), "recall": round(rec, 4),
            "n_got": len(got_set), "n_want": len(want_set),
            "detail": f"P={prec:.2f} R={rec:.2f} F1={f1:.2f} ({len(got_set)} returned, {len(want_set)} true)"}


def run(corpus_dir: Path, embeddings: str | None = None) -> dict[str, Any]:
    system = ContractIntelligence(corpus_dir / "contracts", embeddings=embeddings)
    golden = build_golden_set(corpus_dir / "ground_truth.json")

    rows: list[dict[str, Any]] = []
    lat = {"routed": [], "baseline": []}

    for g in golden:
        t0 = time.perf_counter()
        routed = system.ask(g.query)
        lat["routed"].append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        base = system.ask_baseline(g.query)
        lat["baseline"].append((time.perf_counter() - t0) * 1000)

        rows.append({
            "query": g.query,
            "category": g.category.value,
            "tags": g.tags,
            "note": g.note,
            "routed": {**score(routed, g), "engine": routed.engine,
                       "complete": routed.complete, "scanned": routed.scanned},
            "baseline": {**score(base, g), "engine": base.engine,
                         "complete": base.complete, "scanned": base.scanned},
        })

    by_cat: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"routed": [], "baseline": []})
    for r in rows:
        by_cat[r["category"]]["routed"].append(r["routed"]["score"])
        by_cat[r["category"]]["baseline"].append(r["baseline"]["score"])

    category_summary = {
        cat: {
            "n": len(v["routed"]),
            "routed": round(sum(v["routed"]) / len(v["routed"]), 4),
            "baseline": round(sum(v["baseline"]) / len(v["baseline"]), 4),
        }
        for cat, v in sorted(by_cat.items())
    }

    overall = {
        "routed": round(sum(r["routed"]["score"] for r in rows) / len(rows), 4),
        "baseline": round(sum(r["baseline"]["score"] for r in rows) / len(rows), 4),
    }

    # Group B = every category where retrieval is structurally incapable.
    group_b = [c for c in category_summary if c != QueryType.SEMANTIC.value]
    gb = {
        "routed": round(statistics.mean(category_summary[c]["routed"] for c in group_b), 4),
        "baseline": round(statistics.mean(category_summary[c]["baseline"] for c in group_b), 4),
    }

    # Passage-level check for the semantic control. The main score counts a
    # retrieved contract as relevant if it *has* the clause anywhere, which is
    # generous: 139 of 200 contracts have an indemnification clause, so almost
    # any passage "counts". Here a hit counts only if the retrieved passage IS
    # the clause asked about. Both systems share retrieval, so one number
    # describes both.
    passage: dict[str, float] = {}
    for g in golden:
        if g.answer_kind == "retrieval" and g.clause and system.index is not None:
            hits = system.index.search(g.query, k=8)
            head = re.compile(rf"^\s*\d+\.\s+{g.clause}\b", re.I)
            passage[g.query] = round(sum(bool(head.match(c.text)) for c, _ in hits) / max(len(hits), 1), 4)

    from app.eval.paraphrase import build_cases, evaluate as evaluate_paraphrases
    robustness = evaluate_paraphrases(system, build_cases(corpus_dir / "ground_truth.json"))

    st = system.stats()
    return {
        "system": {
            "contracts": st.contracts, "chunks": st.chunks,
            "extractor": st.extractor, "embedding_backend": st.embedding_backend,
        },
        "n_queries": len(rows),
        "overall": overall,
        "group_b_structural": gb,
        "by_category": category_summary,
        "latency_ms": {
            k: {
                "p50": round(statistics.median(v), 2),
                "p95": round(sorted(v)[max(0, int(len(v) * 0.95) - 1)], 2),
                "mean": round(statistics.mean(v), 2),
            } for k, v in lat.items()
        },
        "abstentions": {
            "routed": sum(1 for r in rows if r["routed"]["abstained"]),
            "baseline": sum(1 for r in rows if r["baseline"]["abstained"]),
        },
        "completeness": {
            "routed_complete": sum(1 for r in rows if r["routed"]["complete"]),
            "baseline_complete": sum(1 for r in rows if r["baseline"]["complete"]),
            "note": "Whether the engine examined the full corpus rather than a retrieved sample.",
        },
        "semantic_passage_precision": {
            "per_query": passage,
            "mean": round(statistics.mean(passage.values()), 4) if passage else None,
            "note": "Fraction of the top-8 retrieved passages that are the clause asked about.",
        },
        "robustness": {"dev": robustness["dev"], "holdout": robustness["holdout"]},
        "results": rows,
        "robustness_results": robustness["results"],
    }


def print_report(rep: dict[str, Any]) -> None:
    s = rep["system"]
    print(f"\n{'=' * 74}")
    print("  CONTRACT INTELLIGENCE -- ROUTED vs NAIVE RAG")
    print(f"{'=' * 74}")
    print(f"  corpus {s['contracts']} contracts / {s['chunks']} chunks   "
          f"extractor={s['extractor']}   embeddings={s['embedding_backend']}")
    print(f"  {rep['n_queries']} golden queries\n")

    print(f"  {'category':14} {'n':>3}  {'routed':>8} {'baseline':>9}   delta")
    print(f"  {'-' * 56}")
    for cat, v in rep["by_category"].items():
        d = v["routed"] - v["baseline"]
        mark = "  <- control" if cat == "semantic" else ""
        print(f"  {cat:14} {v['n']:>3}  {v['routed']:>8.3f} {v['baseline']:>9.3f}   {d:+.3f}{mark}")

    print(f"  {'-' * 56}")
    o, g = rep["overall"], rep["group_b_structural"]
    print(f"  {'OVERALL':14} {rep['n_queries']:>3}  {o['routed']:>8.3f} {o['baseline']:>9.3f}   "
          f"{o['routed'] - o['baseline']:+.3f}")
    print(f"  {'structural':14} {'':>3}  {g['routed']:>8.3f} {g['baseline']:>9.3f}   "
          f"{g['routed'] - g['baseline']:+.3f}")
    print("  (structural = mean across all categories except the semantic control)\n")

    lat = rep["latency_ms"]
    print(f"  latency  routed p50 {lat['routed']['p50']:>7.2f}ms  p95 {lat['routed']['p95']:>7.2f}ms")
    print(f"           naive  p50 {lat['baseline']['p50']:>7.2f}ms  p95 {lat['baseline']['p95']:>7.2f}ms")
    c = rep["completeness"]
    print(f"  complete answers   routed {c['routed_complete']}/{rep['n_queries']}   "
          f"naive {c['baseline_complete']}/{rep['n_queries']}")
    print(f"  abstentions        routed {rep['abstentions']['routed']}   "
          f"naive {rep['abstentions']['baseline']}\n")

    sp = rep.get("semantic_passage_precision") or {}
    if sp.get("per_query"):
        print(f"  semantic control, passage-level precision@8 (shared by both systems): {sp['mean']:.3f}")
        for q, v in sp["per_query"].items():
            print(f"    {v:.3f}  {q}")
        print()

    rob = rep.get("robustness")
    if rob:
        print("  robustness (paraphrased + out-of-scope questions, routed system):")
        for split in ("dev", "holdout"):
            r = rob[split]
            print(f"    {split:8} route {r['route_accuracy']:.3f}  plan {r['plan_accuracy']:.3f}  "
                  f"answer {r['answer_accuracy']:.3f}  confident-wrong {r['confident_wrong']}  "
                  f"out-of-scope declined {r['abstain_rate']:.0%}")
        print()

    worst = sorted(rep["results"], key=lambda r: r["baseline"]["score"])[:5]
    print("  Widest gaps:")
    for r in worst:
        print(f"    {r['baseline']['score']:.2f} -> {r['routed']['score']:.2f}  {r['query'][:58]}")
    print()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    p.add_argument("--out", type=Path, default=Path("eval_report.json"))
    p.add_argument("--embeddings", default=None, help="'st' to force sentence-transformers")
    p.add_argument("--fail-under", type=float, default=None,
                   help="exit 1 if structural routed score falls below this (CI gate)")
    p.add_argument("--robustness-fail-under", type=float, default=None,
                   help="exit 1 if *dev* paraphrase answer accuracy falls below this (CI gate). "
                        "The held-out split is reported, never gated: gating it would invite "
                        "tuning against it, which is what it exists to prevent.")
    a = p.parse_args()

    rep = run(a.corpus, a.embeddings)
    a.out.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print_report(rep)
    print(f"  full report -> {a.out}\n")

    if a.fail_under is not None and rep["group_b_structural"]["routed"] < a.fail_under:
        raise SystemExit(
            f"FAIL: structural score {rep['group_b_structural']['routed']:.3f} < {a.fail_under}"
        )
    dev = rep["robustness"]["dev"]["answer_accuracy"]
    if a.robustness_fail_under is not None and dev < a.robustness_fail_under:
        raise SystemExit(f"FAIL: dev paraphrase answer accuracy {dev:.3f} < {a.robustness_fail_under}")


if __name__ == "__main__":
    main()
