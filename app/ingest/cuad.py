"""
CUAD adapter: real contracts, expert labels, same pipeline.

CUAD (Contract Understanding Atticus Dataset, https://www.atticusprojectai.org/cuad)
is 510 commercial contracts from SEC filings, annotated by lawyers for 41 clause
categories. Each label marks *where* a clause is, not a typed value, so it
answers "does this contract have a liability cap?" but not "what is the cap?".

This module does two things:

  export   writes CUAD's contract text into the same layout as the synthetic
           corpus (<out>/contracts/*.txt), so every part of this system --
           extraction, the fact store, routing, engines, the UI -- runs on real
           contracts unchanged:  CORPUS_DIR=data/cuad uvicorn app.main:app

  labels   writes <out>/cuad_labels.json: per contract, the expert-marked spans
           for the categories that overlap this schema. app/eval/extraction_eval.py
           scores an extractor against them.

Get the data (18 MB, public, from the CUAD GitHub repository):

    make cuad            # or:
    curl -L -o /tmp/cuad.zip https://github.com/TheAtticusProject/cuad/raw/main/data.zip
    unzip -o /tmp/cuad.zip -d /tmp/cuad
    python -m app.ingest.cuad --cuad /tmp/cuad/CUADv1.json --out data/cuad

Honest scope: the deterministic extractor was written against the synthetic
corpus's drafting conventions and is expected to do poorly here -- measuring
exactly how poorly is the point. The LLM extractor is the path for this data.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

# CUAD category -> the schema field it evidences.
CATEGORY_TO_FIELD: dict[str, str] = {
    "Governing Law": "governing_law",
    "Cap On Liability": "liability_cap_usd",
    "Uncapped Liability": "liability_cap_basis",
    "Insurance": "insurance_required",
    "Liquidated Damages": "late_penalty_usd",
    "Renewal Term": "auto_renew",
    "Anti-Assignment": "assignment_allowed",
    "Effective Date": "effective_date",
    "Expiration Date": "expiry_date",
    "Notice Period To Terminate Renewal": "renewal_notice_days",
}


@dataclass
class CuadContract:
    contract_id: str
    title: str
    text: str
    labels: dict[str, list[str]] = field(default_factory=dict)   # category -> answer spans


def safe_id(title: str, taken: set[str]) -> str:
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", title).strip("_")[:100] or "contract"
    cid, n = base, 2
    while cid in taken:
        cid, n = f"{base}_{n}", n + 1
    taken.add(cid)
    return cid


def load_cuad(json_path: str | Path, limit: int | None = None) -> list[CuadContract]:
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))["data"]
    taken: set[str] = set()
    out: list[CuadContract] = []
    for doc in data[:limit]:
        para = doc["paragraphs"][0]
        labels: dict[str, list[str]] = {}
        for qa in para["qas"]:
            category = qa["id"].split("__")[-1]
            if category in CATEGORY_TO_FIELD:
                labels[category] = [a["text"] for a in qa.get("answers", [])]
        out.append(CuadContract(safe_id(doc["title"], taken), doc["title"], para["context"], labels))
    return out


def export(json_path: str | Path, out_dir: str | Path, limit: int | None = None) -> int:
    out = Path(out_dir)
    (out / "contracts").mkdir(parents=True, exist_ok=True)
    contracts = load_cuad(json_path, limit)
    for c in contracts:
        (out / "contracts" / f"{c.contract_id}.txt").write_text(c.text, encoding="utf-8")
    (out / "cuad_labels.json").write_text(json.dumps(
        {c.contract_id: {"title": c.title, "labels": c.labels} for c in contracts}, indent=1),
        encoding="utf-8")
    return len(contracts)


def main() -> None:
    p = argparse.ArgumentParser(description="Export CUAD into this system's corpus layout.")
    p.add_argument("--cuad", type=Path, required=True, help="path to CUADv1.json")
    p.add_argument("--out", type=Path, default=Path("data/cuad"))
    p.add_argument("--limit", type=int, default=None, help="export only the first N contracts")
    a = p.parse_args()
    n = export(a.cuad, a.out, a.limit)
    print(f"exported {n} CUAD contracts -> {a.out}/contracts, labels -> {a.out}/cuad_labels.json")


if __name__ == "__main__":
    main()
