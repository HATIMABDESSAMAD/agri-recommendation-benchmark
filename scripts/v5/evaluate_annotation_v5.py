"""Evaluate revised questions after all saved predictions are complete."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.rescore_saved_questions import LABELS, METHODS, read_jsonl, score

ROOT = Path(__file__).resolve().parents[2]
LIVE = ROOT / "results/annotation_v5_live"
OLD_MATCHER = ROOT / "results/s5_matcher_single_model_deepseek_exploratory"
OLD_REASONER = ROOT / "results/s5_reasoner_single_model_deepseek_exploratory"
OUT = LIVE / "evaluation"
CHANGED = {"G01", "G04", "G06", "G08"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_decision(path: Path) -> tuple[str | None, dict]:
    item = json.loads(path.read_text(encoding="utf-8"))
    decision = item.get("decision") or {}
    label = decision.get("label")
    return label if label in LABELS else None, item


def main() -> None:
    blind_path = ROOT / "data/private/annotation_v5_blind.jsonl"
    gold_path = ROOT / "data/private/annotation_v5_gold.jsonl"
    prior_path = ROOT / "data/private/original_questions_blind.jsonl"
    blind = read_jsonl(blind_path)
    gold = read_jsonl(gold_path)
    prior = {row["id"]: row for row in read_jsonl(prior_path)}
    if len(blind) != 15 or len(gold) != 15:
        raise ValueError("Expected 15 questions")
    if [(q["id"], q["question"]) for q in blind] != [(q["id"], q["question"]) for q in gold]:
        raise ValueError("Blind and gold question text mismatch")
    if {q["id"] for q in blind if q["question"] != prior[q["id"]]["question"]} != CHANGED:
        raise ValueError("Changed question set differs from run manifest")
    manifest = json.loads((LIVE / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest["blind_sha256"] != digest(blind_path) or manifest["gold_opened_by_runner"]:
        raise ValueError("Run manifest does not prove blind input identity")
    rows = []
    for q in gold:
        qid = q["id"]
        changed = qid in CHANGED
        direct_path = LIVE / "direct/decisions" / f"{qid}.json"
        direct_raw = LIVE / "direct/raw" / f"{qid}_direct.json"
        matcher_root = LIVE / "matcher" if changed else OLD_MATCHER
        reasoner_root = LIVE / "reasoner" if changed else OLD_REASONER
        matcher_path = matcher_root / "decisions" / f"{qid}.json"
        reasoner_path = reasoner_root / "decisions" / f"{qid}.json"
        for path in (direct_path, direct_raw, matcher_path, reasoner_path):
            if not path.exists():
                raise FileNotFoundError(path)
        raw = json.loads(direct_raw.read_text(encoding="utf-8"))
        if raw.get("question_id") != qid or raw.get("model") != manifest["model"]:
            raise ValueError(f"Direct run identity mismatch for {qid}")
        query_path = matcher_root / "s5queries" / f"{qid}.json"
        if query_path.exists():
            query = json.loads(query_path.read_text(encoding="utf-8"))
            if query.get("original_question") != q["question"]:
                raise ValueError(f"S5Query question mismatch for {qid}")
        direct_label, direct = load_decision(direct_path)
        matcher_label, matcher = load_decision(matcher_path)
        reasoner_label, reasoner = load_decision(reasoner_path)
        if matcher.get("question_id") != qid or reasoner.get("question_id") != qid:
            raise ValueError(f"Decision ID mismatch for {qid}")
        if reasoner_label and query_path.exists() and reasoner.get("s5query_sha256") != digest(query_path):
            raise ValueError(f"Reasoner S5Query hash mismatch for {qid}")
        rows.append({"question_id": qid, "question": q["question"], "gold_label": q["gold_label"],
                     "changed_question": changed, "direct_label": direct_label,
                     "s5_matcher_label": matcher_label, "s5_reasoner_label": reasoner_label,
                     "direct_correct": direct_label == q["gold_label"],
                     "s5_matcher_correct": matcher_label == q["gold_label"],
                     "s5_reasoner_correct": reasoner_label == q["gold_label"],
                     "s5_matcher_rule_ids": ",".join((matcher.get("decision") or {}).get("rule_ids") or []),
                     "s5_reasoner_rule_ids": ",".join((reasoner.get("decision") or {}).get("rule_ids") or []),
                     "s5_reasoner_cited_pages": ",".join(map(str, (reasoner.get("decision") or {}).get("cited_pages") or [])),
                     "direct_error": str(direct.get("error") or ""),
                     "s5_matcher_error": str(matcher.get("error") or ""),
                     "s5_reasoner_error": str(reasoner.get("error") or "")})
    ledger = json.loads((LIVE / "budget_ledger.json").read_text(encoding="utf-8"))
    raw_files = {"direct": LIVE / "direct/raw", "s5query": LIVE / "matcher/raw_s5query",
                 "s5_reasoner": LIVE / "reasoner/raw"}
    new_costs = {method: sum(float(json.loads(path.read_text(encoding="utf-8")).get("cost_usd") or 0)
                             for path in folder.glob("*.json")) for method, folder in raw_files.items()}
    if abs(sum(new_costs.values()) - ledger["observed_usd"]) > 1e-8:
        raise ValueError("Raw response costs differ from budget ledger")
    report = {"study": "retrospective_exploratory_revised_questions", "model": manifest["model"],
              "gold_sha256": digest(gold_path), "blind_sha256": digest(blind_path),
              "n_questions": len(rows), "changed_question_ids": sorted(CHANGED),
              "methods": {method: score(rows, method) for method in METHODS},
              "new_api_cost_usd": new_costs, "new_api_cost_total_usd": ledger["observed_usd"],
              "budget_cap_usd": ledger["cap_usd"],
              "direct_user_format_matches_previous_run": manifest["direct_user_format_matches_old_run"],
              "gold_sent_to_model": False,
              "note": "Class metrics do not establish agronomic content correctness; saved unchanged S5Query and reasoner predictions are reused."}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "evaluation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (OUT / "questions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"n": len(rows), "cost_usd": ledger["observed_usd"],
                      "methods": {method: {"accuracy": report["methods"][method]["accuracy_all"],
                                           "macro_f1": report["methods"][method]["macro_f1"],
                                           "invalid": report["methods"][method]["invalid"]}
                                  for method in METHODS}}, indent=2))


if __name__ == "__main__":
    main()
