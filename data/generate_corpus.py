"""
Synthetic contract corpus generator with exact, constructed ground truth.

Why synthetic: the eval needs zero label noise. Every fact in the ground-truth
record is one we placed, so "47 contracts have auto-renewal" is true by
construction rather than by annotator agreement. For real contracts, the CUAD
adapter in app/ingest/cuad.py exports CUAD's 510 expert-annotated contracts into
the same corpus layout, and app/eval/extraction_eval.py scores against its labels.

Design constraint that matters: the prose is generated with multiple phrasing
variants per clause, amounts appear in mixed formats ($500,000 / $500K / five
hundred thousand dollars), and dates vary in format. Extraction therefore has
to actually parse language rather than match a fixed template. Absence is
modelled two ways -- clause omitted entirely, or an explicit "no limitation"
statement -- because both occur in real contracts and the second is the case
that defeats semantic retrieval.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, asdict
from datetime import date, timedelta
from pathlib import Path

VENDORS = [
    "Northwind Logistics LLC", "Cascade Analytics Inc.", "Brightpath Media Group",
    "Ironvale Manufacturing Co.", "Solstice Data Services", "Kestrel Security Partners",
    "Vantage Freight Systems", "Lumen Health Technologies", "Redoak Consulting Group",
    "Pinnacle Facilities Management", "Aurora Cloud Systems", "Quarry Point Industrial",
    "Blue Harbor Staffing", "Meridian Payment Solutions", "Tallgrass Energy Services",
    "Copperline Software Ltd.", "Fairmount Engineering", "Sable Creek Transport",
    "Halcyon Research Group", "Windward Supply Co.", "Granite Ridge Contractors",
    "Silverline Telecom", "Oakfield Pharmaceuticals", "Driftwood Marketing",
    "Ember Grid Utilities", "Pellham Legal Services", "Cobalt Aerospace Parts",
    "Ridgeway Insurance Brokers", "Thornton Food Distribution", "Vireo Biotech",
]

CUSTOMER = "Atlas Holdings Corporation"

STATES = ["Delaware", "New York", "California", "Texas", "Illinois", "Massachusetts"]

AGREEMENT_TYPES = [
    ("Master Services Agreement", "MSA"),
    ("Software License Agreement", "SLA"),
    ("Supply Agreement", "SUP"),
    ("Statement of Work", "SOW"),
    ("Maintenance and Support Agreement", "MSP"),
    ("Professional Services Agreement", "PSA"),
]


@dataclass
class ContractTruth:
    """The hidden ground truth. Never shown to the extractor."""
    contract_id: str
    title: str
    vendor: str
    customer: str
    agreement_type: str
    effective_date: str
    expiry_date: str
    term_months: int
    auto_renew: bool
    renewal_notice_days: int | None
    liability_cap_usd: int | None          # None => no cap (omitted or explicitly unlimited)
    liability_cap_style: str               # "capped" | "omitted" | "explicit_unlimited"
    late_penalty_usd: int | None
    termination_notice_days: int
    insurance_required: bool
    insurance_min_usd: int | None
    indemnification: bool
    assignment_allowed: bool
    confidentiality_years: int | None
    governing_law: str
    amends_contract_id: str | None         # supports multi-hop queries


def _fmt_money(n: int, rng: random.Random) -> str:
    """Mixed formats so extraction cannot rely on one pattern."""
    style = rng.choice(["plain", "short", "words", "plain", "plain"])
    if style == "short" and n >= 1000 and n % 1000 == 0:
        if n >= 1_000_000 and n % 1_000_000 == 0:
            return f"${n // 1_000_000}M"
        return f"${n // 1000}K"
    if style == "words":
        words = {
            250_000: "Two Hundred Fifty Thousand Dollars ($250,000)",
            500_000: "Five Hundred Thousand Dollars ($500,000)",
            750_000: "Seven Hundred Fifty Thousand Dollars ($750,000)",
            1_000_000: "One Million Dollars ($1,000,000)",
            2_000_000: "Two Million Dollars ($2,000,000)",
        }
        if n in words:
            return words[n]
    return f"${n:,}"


def _fmt_date(d: date, rng: random.Random) -> str:
    style = rng.choice(["long", "long", "slash", "iso"])
    if style == "long":
        return d.strftime("%B %-d, %Y")
    if style == "slash":
        return d.strftime("%m/%d/%Y")
    return d.isoformat()


def _make_truth(i: int, rng: random.Random, prior_ids: list[str]) -> ContractTruth:
    atype, code = rng.choice(AGREEMENT_TYPES)
    vendor = rng.choice(VENDORS)
    eff = date(2024, 1, 1) + timedelta(days=rng.randint(0, 900))
    term = rng.choice([12, 12, 24, 24, 36, 48, 60])
    expiry = eff + timedelta(days=int(term * 30.44))

    auto_renew = rng.random() < 0.42

    # Liability cap: three realistic states. "explicit_unlimited" is the case
    # that breaks semantic retrieval -- the clause is present and reads like a
    # liability clause, but the answer to "which contracts lack a cap" is yes.
    roll = rng.random()
    if roll < 0.58:
        cap_style = "capped"
        cap = rng.choice([250_000, 500_000, 750_000, 1_000_000, 2_000_000, 5_000_000])
    elif roll < 0.82:
        cap_style = "omitted"
        cap = None
    else:
        cap_style = "explicit_unlimited"
        cap = None

    has_penalty = rng.random() < 0.55
    penalty = rng.choice([10_000, 25_000, 50_000, 100_000, 250_000, 500_000, 750_000]) if has_penalty else None

    insurance = rng.random() < 0.62
    ins_min = rng.choice([1_000_000, 2_000_000, 5_000_000]) if insurance else None

    return ContractTruth(
        contract_id=f"{code}-{2024 + (i % 3)}-{i:04d}",
        title=f"{atype} between {CUSTOMER} and {vendor}",
        vendor=vendor,
        customer=CUSTOMER,
        agreement_type=atype,
        effective_date=eff.isoformat(),
        expiry_date=expiry.isoformat(),
        term_months=term,
        auto_renew=auto_renew,
        renewal_notice_days=rng.choice([30, 60, 90]) if auto_renew else None,
        liability_cap_usd=cap,
        liability_cap_style=cap_style,
        late_penalty_usd=penalty,
        termination_notice_days=rng.choice([15, 30, 30, 60, 90]),
        insurance_required=insurance,
        insurance_min_usd=ins_min,
        indemnification=rng.random() < 0.71,
        assignment_allowed=rng.random() < 0.35,
        confidentiality_years=rng.choice([2, 3, 5, 5, 7]) if rng.random() < 0.8 else None,
        governing_law=rng.choice(STATES),
        amends_contract_id=(
            rng.choice(prior_ids) if prior_ids and atype == "Statement of Work" and rng.random() < 0.5 else None
        ),
    )


def _render(t: ContractTruth, rng: random.Random) -> str:
    """Render contract prose. The extractor sees only this."""
    eff = date.fromisoformat(t.effective_date)
    exp = date.fromisoformat(t.expiry_date)
    s: list[str] = []

    s.append(f"{t.agreement_type.upper()}\n")
    s.append(f"Contract Reference: {t.contract_id}\n")

    s.append(
        f"This {t.agreement_type} (the \"Agreement\") is entered into as of "
        f"{_fmt_date(eff, rng)} (the \"Effective Date\") by and between {t.customer}, "
        f"a Delaware corporation (\"Customer\"), and {t.vendor} (\"Supplier\").\n"
    )

    if t.amends_contract_id:
        s.append(
            f"1. INCORPORATION. This Agreement is issued under and incorporates by reference "
            f"the terms of the Master Services Agreement bearing reference {t.amends_contract_id}. "
            f"In the event of conflict, the terms of the master agreement control.\n"
        )

    # Term and renewal
    if t.auto_renew:
        variant = rng.choice([
            f"2. TERM. The initial term commences on the Effective Date and continues for "
            f"{t.term_months} months, expiring {_fmt_date(exp, rng)}. This Agreement shall "
            f"automatically renew for successive periods of equal length unless either party "
            f"delivers written notice of non-renewal not less than {t.renewal_notice_days} days "
            f"prior to the end of the then-current term.",
            f"2. TERM AND RENEWAL. The term of this Agreement is {t.term_months} months from the "
            f"Effective Date, ending {_fmt_date(exp, rng)}. Upon expiration the Agreement renews "
            f"automatically on the same terms for an additional {t.term_months} month period, "
            f"provided that either party may prevent such renewal by giving {t.renewal_notice_days} "
            f"days advance written notice.",
        ])
    else:
        variant = rng.choice([
            f"2. TERM. This Agreement begins on the Effective Date and expires on "
            f"{_fmt_date(exp, rng)}, a period of {t.term_months} months. The Agreement does not "
            f"renew automatically. Any extension requires a written amendment executed by both parties.",
            f"2. TERM. The Agreement remains in effect for {t.term_months} months from the Effective "
            f"Date, terminating {_fmt_date(exp, rng)} unless earlier terminated. Continuation beyond "
            f"the stated expiration requires mutual written agreement.",
        ])
    s.append(variant + "\n")

    # Payment and penalties
    if t.late_penalty_usd:
        s.append(
            f"3. PAYMENT. Customer shall pay undisputed invoices within {rng.choice([30, 45, 60])} days "
            f"of receipt. Failure to deliver the services within the agreed schedule shall entitle "
            f"Customer to liquidated damages of {_fmt_money(t.late_penalty_usd, rng)} per occurrence, "
            f"which the parties agree is a reasonable estimate of actual harm and not a penalty.\n"
        )
    else:
        s.append(
            f"3. PAYMENT. Customer shall pay undisputed invoices within {rng.choice([30, 45, 60])} days "
            f"of receipt. Disputed amounts shall be resolved under Section 9 before becoming due.\n"
        )

    # Liability -- the interesting one
    if t.liability_cap_style == "capped":
        s.append(
            f"4. LIMITATION OF LIABILITY. Except for breaches of confidentiality and "
            f"indemnification obligations, the aggregate liability of either party arising out of "
            f"or relating to this Agreement shall not exceed {_fmt_money(t.liability_cap_usd, rng)}. "
            f"Neither party shall be liable for indirect, incidental, or consequential damages.\n"
        )
    elif t.liability_cap_style == "explicit_unlimited":
        s.append(
            f"4. LIABILITY. The parties acknowledge that no monetary limitation shall apply to "
            f"liability arising under this Agreement. Each party remains fully responsible for all "
            f"direct damages caused by its acts or omissions without cap or limitation.\n"
        )
    # omitted => no liability section at all

    if t.indemnification:
        s.append(
            f"5. INDEMNIFICATION. Supplier shall defend, indemnify and hold harmless Customer from "
            f"third party claims arising from Supplier's negligence, willful misconduct, or "
            f"infringement of intellectual property rights.\n"
        )

    if t.insurance_required:
        s.append(
            f"6. INSURANCE. Supplier shall maintain commercial general liability insurance with "
            f"limits of not less than {_fmt_money(t.insurance_min_usd, rng)} per occurrence and "
            f"shall furnish Customer with a certificate of insurance evidencing such coverage prior "
            f"to commencing work, naming Customer as an additional insured.\n"
        )

    if t.confidentiality_years:
        s.append(
            f"7. CONFIDENTIALITY. Each party shall protect the Confidential Information of the other "
            f"and shall not disclose it to third parties for a period of {t.confidentiality_years} "
            f"years following termination or expiration of this Agreement.\n"
        )

    s.append(
        f"8. TERMINATION. Either party may terminate this Agreement for convenience upon "
        f"{t.termination_notice_days} days prior written notice. Either party may terminate "
        f"immediately for material breach that remains uncured for thirty (30) days after notice.\n"
    )

    if t.assignment_allowed:
        s.append(
            "9. ASSIGNMENT. Either party may assign this Agreement to a successor in interest in "
            "connection with a merger, acquisition, or sale of substantially all assets, with notice "
            "to the other party.\n"
        )
    else:
        s.append(
            "9. ASSIGNMENT. Neither party may assign or transfer this Agreement, in whole or in part, "
            "without the prior written consent of the other party, which consent may be withheld in "
            "its sole discretion.\n"
        )

    s.append(
        f"10. GOVERNING LAW. This Agreement shall be governed by and construed in accordance with "
        f"the laws of the State of {t.governing_law}, without regard to its conflict of laws "
        f"principles. The parties consent to exclusive jurisdiction of the state and federal courts "
        f"located therein.\n"
    )

    s.append(
        f"IN WITNESS WHEREOF, the parties have executed this Agreement as of the Effective Date.\n\n"
        f"{t.customer}                    {t.vendor}\n"
        f"By: _____________________       By: _____________________\n"
    )

    return "\n".join(s)


def generate(n: int, seed: int, out_dir: Path) -> None:
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    docs_dir = out_dir / "contracts"
    docs_dir.mkdir(exist_ok=True)

    truths: list[ContractTruth] = []
    prior_ids: list[str] = []

    for i in range(n):
        t = _make_truth(i, rng, prior_ids)
        truths.append(t)
        if t.agreement_type == "Master Services Agreement":
            prior_ids.append(t.contract_id)
        (docs_dir / f"{t.contract_id}.txt").write_text(_render(t, rng), encoding="utf-8")

    with (out_dir / "ground_truth.json").open("w", encoding="utf-8") as f:
        json.dump([asdict(t) for t in truths], f, indent=2)

    # Quick corpus summary so the numbers in the README are never guessed.
    summary = {
        "n_contracts": len(truths),
        "auto_renew": sum(t.auto_renew for t in truths),
        "no_liability_cap": sum(t.liability_cap_usd is None for t in truths),
        "  of_which_omitted": sum(t.liability_cap_style == "omitted" for t in truths),
        "  of_which_explicit_unlimited": sum(t.liability_cap_style == "explicit_unlimited" for t in truths),
        "no_insurance_requirement": sum(not t.insurance_required for t in truths),
        "penalty_over_100k": sum(bool(t.late_penalty_usd and t.late_penalty_usd > 100_000) for t in truths),
        "amendments_with_parent": sum(t.amends_contract_id is not None for t in truths),
    }
    (out_dir / "corpus_stats.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", type=Path, default=Path(__file__).parent / "corpus")
    a = p.parse_args()
    generate(a.n, a.seed, a.out)
