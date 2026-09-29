"""
Extraction accuracy, measured rather than assumed.

Two modes:

  synthetic   every extracted field against the constructed ground truth.
              Replaces the README's old hand-written "100% on all 16 fields"
              with a number the code produces.

  cuad        against lawyers' labels on 510 real contracts. CUAD marks whether
              a clause is present, not its typed value, so each field is scored
              on *detection*: when the experts say the clause is there, did the
              extractor find it (recall), and when the extractor claims it, is
              it there (precision)? Governing law is also checked on value.

The CUAD mode also reports the number that matters most for this system's
honesty: how often the extractor says a liability clause is *omitted* when the
experts found one. The absence engine treats basis="omitted" as a settled fact,
so every false "omitted" becomes a wrong answer to "which contracts have no
liability cap?" that is still labelled complete.

Run:
    python -m app.eval.extraction_eval --synthetic data/corpus
    python -m app.eval.extraction_eval --cuad data/cuad [--extractor llm]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

from app.ingest.extract import get_extractor


def _synthetic_expected(t: dict) -> dict[str, Any]:
    exp = {k: t[k] for k in (
        "vendor", "agreement_type", "governing_law", "effective_date", "expiry_date",
        "term_months", "auto_renew", "renewal_notice_days", "liability_cap_usd",
        "late_penalty_usd", "termination_notice_days", "insurance_required",
        "insurance_min_usd", "indemnification", "assignment_allowed",
        "confidentiality_years", "amends_contract_id")}
    exp["liability_cap_basis"] = t["liability_cap_style"]
    exp["confidentiality_basis"] = "present" if t["confidentiality_years"] else "omitted"
    return exp


def eval_synthetic(corpus: Path, extractor: str = "deterministic") -> dict[str, Any]:
    truth = {t["contract_id"]: t for t in json.loads((corpus / "ground_truth.json").read_text())}
    docs = get_extractor(extractor).extract_dir(corpus / "contracts")
    per_field: dict[str, list[bool]] = {}
    misses: dict[str, list[str]] = {}
    for d in docs:
        for f, want in _synthetic_expected(truth[d.contract_id]).items():
            got = d.value(f)
            ok = (bool(got) == want) if isinstance(want, bool) else (got == want)
            per_field.setdefault(f, []).append(ok)
            if not ok:
                misses.setdefault(f, []).append(f"{d.contract_id}: got {got!r}, want {want!r}")
    acc = {f: round(sum(v) / len(v), 4) for f, v in per_field.items()}
    return {"mode": "synthetic", "extractor": extractor, "contracts": len(docs),
            "fields": len(acc), "accuracy": acc,
            "fields_at_100": sum(1 for v in acc.values() if v == 1.0),
            "examples_of_misses": {f: m[:3] for f, m in misses.items()}}


# CUAD category -> does the extracted contract claim that clause is present?
DETECTORS: dict[str, Callable[[Any], bool]] = {
    "Governing Law": lambda d: d.value("governing_law") is not None,
    "Cap On Liability": lambda d: d.value("liability_cap_basis") in ("capped", "unparsed"),
    "Uncapped Liability": lambda d: d.value("liability_cap_basis") == "explicit_unlimited",
    "Insurance": lambda d: bool(d.value("insurance_required")),
    "Liquidated Damages": lambda d: d.value("late_penalty_usd") is not None,
    "Renewal Term": lambda d: bool(d.value("auto_renew")),
    "Anti-Assignment": lambda d: d.value("assignment_allowed") is False,
    "Effective Date": lambda d: d.value("effective_date") is not None,
    "Expiration Date": lambda d: d.value("expiry_date") is not None,
    "Notice Period To Terminate Renewal": lambda d: d.value("renewal_notice_days") is not None,
}


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return round(p, 3), round(r, 3), round(2 * p * r / (p + r), 3) if p + r else 0.0


def eval_cuad(cuad_dir: Path, extractor: str = "deterministic", limit: int | None = None) -> dict[str, Any]:
    labels: dict[str, dict] = json.loads((cuad_dir / "cuad_labels.json").read_text())
    ext = get_extractor(extractor)
    ids = sorted(labels)[:limit]
    docs = [ext.extract(cid, (cuad_dir / "contracts" / f"{cid}.txt").read_text(encoding="utf-8"),
                        str(cuad_dir / "contracts" / f"{cid}.txt")) for cid in ids]

    per_cat: dict[str, dict[str, Any]] = {}
    for cat, detect in DETECTORS.items():
        tp = fp = fn = tn = 0
        for d in docs:
            present = bool(labels[d.contract_id]["labels"].get(cat))
            claimed = detect(d)
            tp += present and claimed
            fp += (not present) and claimed
            fn += present and not claimed
            tn += (not present) and not claimed
        p, r, f1 = _prf(tp, fp, fn)
        per_cat[cat] = {"labelled_present": tp + fn, "precision": p, "recall": r, "f1": f1}

    # Governing law value: does the extracted state appear in the expert span?
    law_ok = law_n = 0
    for d in docs:
        spans = labels[d.contract_id]["labels"].get("Governing Law") or []
        got = d.value("governing_law")
        if spans and got:
            law_n += 1
            law_ok += got.lower() in " ".join(spans).lower()

    # The honesty number: "omitted" claims contradicted by the experts.
    omitted = [d for d in docs if d.value("liability_cap_basis") == "omitted"]
    false_omitted = [d for d in omitted if labels[d.contract_id]["labels"].get("Cap On Liability")
                     or labels[d.contract_id]["labels"].get("Uncapped Liability")]
    return {
        "mode": "cuad", "extractor": extractor, "contracts": len(docs),
        "detection": per_cat,
        "governing_law_value_accuracy": round(law_ok / law_n, 3) if law_n else None,
        "liability_basis_omitted": len(omitted),
        "liability_basis_omitted_but_experts_found_clause": len(false_omitted),
    }


def print_report(rep: dict[str, Any]) -> None:
    print(f"\n  extraction eval -- {rep['mode']}, extractor={rep['extractor']}, {rep['contracts']} contracts\n")
    if rep["mode"] == "synthetic":
        for f, a in sorted(rep["accuracy"].items(), key=lambda kv: kv[1]):
            print(f"    {a:6.3f}  {f}")
        print(f"\n  {rep['fields_at_100']}/{rep['fields']} fields at 100%")
        for f, ex in rep["examples_of_misses"].items():
            print(f"    miss {f}: {ex[0]}")
    else:
        print(f"    {'CUAD category':38} {'present':>7} {'prec':>6} {'recall':>6} {'F1':>6}")
        for cat, v in rep["detection"].items():
            print(f"    {cat:38} {v['labelled_present']:>7} {v['precision']:>6.3f} {v['recall']:>6.3f} {v['f1']:>6.3f}")
        print(f"\n    governing law value accuracy (when both found): {rep['governing_law_value_accuracy']}")
        print(f"    liability basis 'omitted': {rep['liability_basis_omitted']} contracts, of which "
              f"{rep['liability_basis_omitted_but_experts_found_clause']} have a liability clause per CUAD")
    print()


def main() -> None:
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--synthetic", type=Path, help="corpus dir with ground_truth.json")
    g.add_argument("--cuad", type=Path, help="dir written by app.ingest.cuad")
    p.add_argument("--extractor", default="deterministic", choices=["deterministic", "llm"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args()
    rep = eval_synthetic(a.synthetic, a.extractor) if a.synthetic else eval_cuad(a.cuad, a.extractor, a.limit)
    print_report(rep)
    if a.out:
        a.out.write_text(json.dumps(rep, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
