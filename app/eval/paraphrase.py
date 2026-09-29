"""
Robustness eval: paraphrased questions and out-of-scope questions.

Why this exists
---------------
The golden set in golden.py uses the phrasings the router's rules were written
for, so its 1.000 measures *fit*, not generality. This module asks the same
kinds of questions the way a person actually would ("which agreements are
uncapped?", "contracts that started in 2025") and scores three things
separately:

  route   -- did the router pick the right engine?
  plan    -- right engine AND right field AND right operator?
  answer  -- is the final answer correct against ground truth?

Route can be right while the answer is wrong (the plan picked the wrong field
or the wrong date window), and that case is the dangerous one: the engine still
reports `complete=True`. Scoring the three layers separately is what exposes it.

Two splits, written together before any router change:

  dev      -- the router was tuned against these failures
  holdout  -- never used for tuning; the honest generalisation number

Both splits were written by the same author as the rules, which flatters them.
Paraphrases written by other people (or sampled from an LLM) would be a harder
and fairer test; see README.

Out-of-scope questions (weather, recipes, facts the corpus does not contain)
are scored on whether the system *declines*. Returning eight arbitrary
contracts for "what is the weather in Dallas" is a wrong answer, not a sample.

Run:
    python -m app.eval.paraphrase --corpus data/corpus
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from app.eval.golden import GoldenQuery
from app.models import QueryType as T

DEV, HOLDOUT = "dev", "holdout"


@dataclass
class Case:
    query: str
    split: str
    category: T | None                 # None => out of scope, must abstain
    field: str | None = None           # expected plan field
    operator: str | None = None        # expected plan operator
    expected: Any = None               # computed from ground truth
    answer_kind: str = "set"           # "scalar" | "set" | "retrieval" | "abstain"

    def as_golden(self) -> GoldenQuery:
        return GoldenQuery(self.query, self.category or T.SEMANTIC, self.expected, self.answer_kind)


def build_cases(ground_truth_path: str | Path) -> list[Case]:
    gt: list[dict] = json.loads(Path(ground_truth_path).read_text(encoding="utf-8"))

    def ids(pred: Callable[[dict], bool]) -> set[str]:
        return {t["contract_id"] for t in gt if pred(t)}

    def count(pred: Callable[[dict], bool]) -> int:
        return len(ids(pred))

    def children_of(pred: Callable[[dict], bool]) -> set[str]:
        parents = ids(pred)
        return ids(lambda t: t["amends_contract_id"] in parents)

    caps = [t["liability_cap_usd"] for t in gt if t["liability_cap_usd"] is not None]
    notice = [t["termination_notice_days"] for t in gt if t["termination_notice_days"] is not None]
    n = len(gt)

    no_cap = lambda t: t["liability_cap_usd"] is None                      # noqa: E731
    has_liability_clause = lambda t: t["liability_cap_style"] != "omitted"  # noqa: E731

    C: list[Case] = [
        # ================================================================ DEV
        # --- aggregate
        Case("What number of agreements auto-renew?", DEV, T.AGGREGATE, "auto_renew", "eq",
             count(lambda t: t["auto_renew"]), "scalar"),
        Case("Count the deals that renew on their own", DEV, T.AGGREGATE, "auto_renew", "eq",
             count(lambda t: t["auto_renew"]), "scalar"),
        Case("How many agreements need the supplier to carry insurance?", DEV, T.AGGREGATE,
             "insurance_required", "eq", count(lambda t: t["insurance_required"]), "scalar"),
        Case("Total number of contracts with an indemnity clause", DEV, T.AGGREGATE,
             "indemnification", "eq", count(lambda t: t["indemnification"]), "scalar"),
        Case("What is the average liability cap?", DEV, T.AGGREGATE, "liability_cap_usd", "not_null",
             round(sum(caps) / len(caps), 2), "scalar"),
        Case("What percentage of contracts can be assigned?", DEV, T.AGGREGATE,
             "assignment_allowed", "eq", round(100.0 * count(lambda t: t["assignment_allowed"]) / n, 1),
             "scalar"),
        Case("How many Texas contracts auto-renew?", DEV, T.AGGREGATE, "auto_renew", "eq",
             count(lambda t: t["auto_renew"] and t["governing_law"] == "Texas"), "scalar"),
        # --- absence
        Case("Which agreements are uncapped?", DEV, T.ABSENCE, "liability_cap_usd", "is_null",
             ids(no_cap)),
        Case("Are there contracts where liability is unlimited?", DEV, T.ABSENCE,
             "liability_cap_usd", "is_null", ids(no_cap)),
        Case("Show contracts missing a cap on liability", DEV, T.ABSENCE, "liability_cap_usd",
             "is_null", ids(no_cap)),
        Case("Which contracts don't have an indemnification section?", DEV, T.ABSENCE,
             "indemnification", "is_null", ids(lambda t: not t["indemnification"])),
        Case("List agreements with no insurance requirement", DEV, T.ABSENCE,
             "insurance_required", "is_null", ids(lambda t: not t["insurance_required"])),
        Case("How many contracts lack a confidentiality term?", DEV, T.ABSENCE,
             "confidentiality_years", "is_null", count(lambda t: t["confidentiality_years"] is None),
             "scalar"),
        # --- numeric
        Case("Contracts with liquidated damages over $100K", DEV, T.NUMERIC, "late_penalty_usd", "gt",
             ids(lambda t: (t["late_penalty_usd"] or 0) > 100_000)),
        Case("Which deals have a liability cap of at least $2M?", DEV, T.NUMERIC,
             "liability_cap_usd", "gte", ids(lambda t: (t["liability_cap_usd"] or 0) >= 2_000_000)),
        Case("Which contracts need more than 30 days notice to terminate?", DEV, T.NUMERIC,
             "termination_notice_days", "gt", ids(lambda t: t["termination_notice_days"] > 30)),
        Case("Contracts where the late penalty is below $50,000", DEV, T.NUMERIC, "late_penalty_usd",
             "lt", ids(lambda t: t["late_penalty_usd"] is not None and t["late_penalty_usd"] < 50_000)),
        Case("Which agreements require insurance coverage of at least $2,000,000?", DEV, T.NUMERIC,
             "insurance_min_usd", "gte",
             ids(lambda t: t["insurance_min_usd"] is not None and t["insurance_min_usd"] >= 2_000_000)),
        Case("Contracts with a term longer than 24 months", DEV, T.NUMERIC, "term_months", "gt",
             ids(lambda t: t["term_months"] > 24)),
        # --- temporal
        Case("Contracts that started in 2025", DEV, T.TEMPORAL, "effective_date", "between",
             ids(lambda t: t["effective_date"].startswith("2025"))),
        Case("What expires before June 2026?", DEV, T.TEMPORAL, "expiry_date", "lt",
             ids(lambda t: t["expiry_date"] < "2026-06-01")),
        Case("Which contracts expire after March 1, 2027?", DEV, T.TEMPORAL, "expiry_date", "gt",
             ids(lambda t: t["expiry_date"] > "2027-03-01")),
        Case("Contracts ending in the first quarter of 2026", DEV, T.TEMPORAL, "expiry_date", "between",
             ids(lambda t: "2026-01-01" <= t["expiry_date"] <= "2026-03-31")),
        Case("How many contracts expire in 2028?", DEV, T.TEMPORAL, "expiry_date", "between",
             count(lambda t: t["expiry_date"].startswith("2028")), "scalar"),
        Case("Which contracts were signed between January 2024 and June 2024?", DEV, T.TEMPORAL,
             "effective_date", "between",
             ids(lambda t: "2024-01-01" <= t["effective_date"] <= "2024-06-30")),
        Case("Contracts expiring in March 2026", DEV, T.TEMPORAL, "expiry_date", "between",
             ids(lambda t: "2026-03-01" <= t["expiry_date"] <= "2026-03-31")),
        # --- multihop
        Case("Which statements of work were issued under a Delaware master agreement?", DEV,
             T.MULTIHOP, None, None, children_of(lambda t: t["governing_law"] == "Delaware")),
        Case("Find SOWs referencing Texas-governed MSAs", DEV, T.MULTIHOP, None, None,
             children_of(lambda t: t["governing_law"] == "Texas")),
        # --- enumerate
        Case("Show me all Delaware-governed contracts", DEV, T.ENUMERATE, "governing_law", "eq",
             ids(lambda t: t["governing_law"] == "Delaware")),
        Case("Which contracts are under Texas law?", DEV, T.ENUMERATE, "governing_law", "eq",
             ids(lambda t: t["governing_law"] == "Texas")),
        Case("List all agreements with an indemnification clause", DEV, T.ENUMERATE,
             "indemnification", "eq", ids(lambda t: t["indemnification"])),
        Case("Which contracts are Statements of Work?", DEV, T.ENUMERATE, "agreement_type", "eq",
             ids(lambda t: t["agreement_type"] == "Statement of Work")),
        # --- semantic (must answer, not abstain)
        Case("Summarize the insurance requirements", DEV, T.SEMANTIC, None, None,
             ids(lambda t: t["insurance_required"]), "retrieval"),
        Case("Explain the limitation of liability language", DEV, T.SEMANTIC, None, None,
             ids(has_liability_clause), "retrieval"),
        Case("What do the liquidated damages provisions say?", DEV, T.SEMANTIC, None, None,
             ids(lambda t: t["late_penalty_usd"] is not None), "retrieval"),
        # --- out of scope (must abstain)
        Case("What is the weather in Dallas?", DEV, None, answer_kind="abstain"),
        Case("Best biryani recipe", DEV, None, answer_kind="abstain"),
        Case("How many companies have no employees?", DEV, None, answer_kind="abstain"),
        Case("Who won the world cup?", DEV, None, answer_kind="abstain"),
        Case("What is the capital of France?", DEV, None, answer_kind="abstain"),

        # ============================================================ HOLDOUT
        # Written alongside the dev split, before any router change, and never
        # used to tune the rules. Treat this as the generalisation number.
        Case("How many contracts renew automatically?", HOLDOUT, T.AGGREGATE, "auto_renew", "eq",
             count(lambda t: t["auto_renew"]), "scalar"),
        Case("How many of our agreements have an indemnity provision?", HOLDOUT, T.AGGREGATE,
             "indemnification", "eq", count(lambda t: t["indemnification"]), "scalar"),
        Case("Number of contracts that allow the vendor to assign them", HOLDOUT, T.AGGREGATE,
             "assignment_allowed", "eq", count(lambda t: t["assignment_allowed"]), "scalar"),
        Case("What's the average termination notice period?", HOLDOUT, T.AGGREGATE,
             "termination_notice_days", "not_null", round(sum(notice) / len(notice), 2), "scalar"),
        Case("How many California contracts require insurance?", HOLDOUT, T.AGGREGATE,
             "insurance_required", "eq",
             count(lambda t: t["insurance_required"] and t["governing_law"] == "California"), "scalar"),
        Case("Which contracts have unlimited liability?", HOLDOUT, T.ABSENCE, "liability_cap_usd",
             "is_null", ids(no_cap)),
        Case("Which contracts cannot be assigned?", HOLDOUT, T.ABSENCE, "assignment_allowed",
             "is_null", ids(lambda t: not t["assignment_allowed"])),
        Case("Contracts without an insurance clause", HOLDOUT, T.ABSENCE, "insurance_required",
             "is_null", ids(lambda t: not t["insurance_required"])),
        Case("How many agreements have no cap on liability?", HOLDOUT, T.ABSENCE,
             "liability_cap_usd", "is_null", count(no_cap), "scalar"),
        Case("Which contracts have a liability cap above $1M?", HOLDOUT, T.NUMERIC,
             "liability_cap_usd", "gt", ids(lambda t: (t["liability_cap_usd"] or 0) > 1_000_000)),
        Case("Contracts with penalties of $250,000 or more", HOLDOUT, T.NUMERIC, "late_penalty_usd",
             "gte", ids(lambda t: (t["late_penalty_usd"] or 0) >= 250_000)),
        Case("Agreements requiring at least 60 days termination notice", HOLDOUT, T.NUMERIC,
             "termination_notice_days", "gte", ids(lambda t: t["termination_notice_days"] >= 60)),
        Case("Contracts with a confidentiality period longer than 3 years", HOLDOUT, T.NUMERIC,
             "confidentiality_years", "gt",
             ids(lambda t: t["confidentiality_years"] is not None and t["confidentiality_years"] > 3)),
        Case("Which contracts expire before 2026?", HOLDOUT, T.TEMPORAL, "expiry_date", "lt",
             ids(lambda t: t["expiry_date"] < "2026-01-01")),
        Case("Contracts that expire in Q3 2027", HOLDOUT, T.TEMPORAL, "expiry_date", "between",
             ids(lambda t: "2027-07-01" <= t["expiry_date"] <= "2027-09-30")),
        Case("How many agreements began in 2024?", HOLDOUT, T.TEMPORAL, "effective_date", "between",
             count(lambda t: t["effective_date"].startswith("2024")), "scalar"),
        Case("Which contracts expire on or after January 1, 2028?", HOLDOUT, T.TEMPORAL,
             "expiry_date", "gte", ids(lambda t: t["expiry_date"] >= "2028-01-01")),
        Case("List contracts expiring during the second half of 2026", HOLDOUT, T.TEMPORAL,
             "expiry_date", "between", ids(lambda t: "2026-07-01" <= t["expiry_date"] <= "2026-12-31")),
        Case("Which SOWs fall under Illinois-governed master services agreements?", HOLDOUT,
             T.MULTIHOP, None, None, children_of(lambda t: t["governing_law"] == "Illinois")),
        Case("Statements of work that reference a New York master agreement", HOLDOUT, T.MULTIHOP,
             None, None, children_of(lambda t: t["governing_law"] == "New York")),
        Case("List every contract governed by California law", HOLDOUT, T.ENUMERATE, "governing_law",
             "eq", ids(lambda t: t["governing_law"] == "California")),
        Case("Show all Massachusetts contracts", HOLDOUT, T.ENUMERATE, "governing_law", "eq",
             ids(lambda t: t["governing_law"] == "Massachusetts")),
        Case("Which agreements are Master Services Agreements?", HOLDOUT, T.ENUMERATE,
             "agreement_type", "eq", ids(lambda t: t["agreement_type"] == "Master Services Agreement")),
        Case("What does the confidentiality section require?", HOLDOUT, T.SEMANTIC, None, None,
             ids(lambda t: t["confidentiality_years"] is not None), "retrieval"),
        Case("Explain how termination for convenience works", HOLDOUT, T.SEMANTIC, None, None,
             ids(lambda t: True), "retrieval"),
        Case("Describe the insurance obligations on the supplier", HOLDOUT, T.SEMANTIC, None, None,
             ids(lambda t: t["insurance_required"]), "retrieval"),
        Case("What is the stock price of Atlas Holdings?", HOLDOUT, None, answer_kind="abstain"),
        Case("Tell me a joke", HOLDOUT, None, answer_kind="abstain"),
        Case("How many vendors are headquartered in Europe?", HOLDOUT, None, answer_kind="abstain"),
        Case("Which employees work remotely?", HOLDOUT, None, answer_kind="abstain"),
    ]
    return C


def evaluate(system, cases: list[Case]) -> dict[str, Any]:
    from app.eval.run_eval import score

    rows = []
    for c in cases:
        plan = system.plan(c.query)
        ans = system.ask(c.query)
        if c.category is None:
            ok = bool(ans.abstained)
            rows.append({"query": c.query, "split": c.split, "kind": "out_of_scope",
                         "route": ok, "plan": ok, "answer": 1.0 if ok else 0.0,
                         "got_type": plan.query_type.value, "abstained": ans.abstained})
            continue
        route = plan.query_type == c.category
        plan_ok = route and (c.field is None or plan.field == c.field) and \
            (c.operator is None or plan.operator == c.operator)
        s = score(ans, c.as_golden())
        rows.append({"query": c.query, "split": c.split, "kind": "in_scope",
                     "want_type": c.category.value, "got_type": plan.query_type.value,
                     "want_plan": f"{c.field} {c.operator}", "got_plan": f"{plan.field} {plan.operator}",
                     "route": route, "plan": plan_ok, "answer": s["score"],
                     "abstained": ans.abstained, "complete": ans.complete, "detail": s["detail"]})

    def summarise(split: str) -> dict[str, Any]:
        ins = [r for r in rows if r["split"] == split and r["kind"] == "in_scope"]
        oos = [r for r in rows if r["split"] == split and r["kind"] == "out_of_scope"]
        # a wrong answer the system still labelled a complete scan
        confident_wrong = [r for r in ins if r["answer"] < 1.0 and r.get("complete")]
        return {
            "n_in_scope": len(ins),
            "route_accuracy": round(statistics.mean(r["route"] for r in ins), 3),
            "plan_accuracy": round(statistics.mean(r["plan"] for r in ins), 3),
            "answer_accuracy": round(statistics.mean(r["answer"] for r in ins), 3),
            "exact_answers": sum(1 for r in ins if r["answer"] == 1.0),
            "confident_wrong": len(confident_wrong),
            "n_out_of_scope": len(oos),
            "abstain_rate": round(statistics.mean(r["answer"] for r in oos), 3) if oos else None,
        }

    return {"dev": summarise(DEV), "holdout": summarise(HOLDOUT), "results": rows}


def print_report(rep: dict[str, Any], show_failures: str | None = DEV) -> None:
    print(f"\n  {'split':8} {'route':>6} {'plan':>6} {'answer':>7} {'exact':>7} "
          f"{'confident-wrong':>16} {'out-of-scope declined':>22}")
    for split in (DEV, HOLDOUT):
        s = rep[split]
        oos = f"{s['abstain_rate']:.0%} of {s['n_out_of_scope']}" if s["n_out_of_scope"] else "-"
        print(f"  {split:8} {s['route_accuracy']:>6.3f} {s['plan_accuracy']:>6.3f} "
              f"{s['answer_accuracy']:>7.3f} {s['exact_answers']:>3}/{s['n_in_scope']:<3} "
              f"{s['confident_wrong']:>16} {oos:>22}")
    print("  (confident-wrong = wrong answer the engine still labelled a complete scan)\n")
    if show_failures:
        fails = [r for r in rep["results"] if r["split"] == show_failures and r["answer"] < 1.0]
        if fails:
            print(f"  {show_failures} failures:")
            for r in fails:
                if r["kind"] == "out_of_scope":
                    print(f"    [not declined -> {r['got_type']}] {r['query']}")
                else:
                    print(f"    [{r['want_type']}:{r['want_plan']} -> {r['got_type']}:{r['got_plan']}] "
                          f"{r['query']}  ({r['detail']})")
            print()


def main() -> None:
    from app.pipeline import ContractIntelligence

    p = argparse.ArgumentParser()
    p.add_argument("--corpus", type=Path, default=Path("data/corpus"))
    p.add_argument("--embeddings", default=None)
    p.add_argument("--show", default=DEV, help="which split's failures to print ('dev', 'holdout', 'none')")
    p.add_argument("--fail-under", type=float, default=None,
                   help="exit 1 if dev answer accuracy falls below this (CI gate)")
    a = p.parse_args()

    system = ContractIntelligence(a.corpus / "contracts", embeddings=a.embeddings)
    rep = evaluate(system, build_cases(a.corpus / "ground_truth.json"))
    print_report(rep, None if a.show == "none" else a.show)
    if a.fail_under is not None and rep[DEV]["answer_accuracy"] < a.fail_under:
        raise SystemExit(f"FAIL: dev paraphrase answer accuracy "
                         f"{rep[DEV]['answer_accuracy']:.3f} < {a.fail_under}")


if __name__ == "__main__":
    main()
