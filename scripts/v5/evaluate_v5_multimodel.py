"""Evaluate completed V5 model predictions after generation has ended."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.rescore_saved_questions import LABELS, METHODS, read_jsonl, score

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "results/annotation_v5_multimodel"
OUT = RUN / "evaluation"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_decision(path: Path) -> tuple[str | None, dict]:
    item = json.loads(path.read_text(encoding="utf-8"))
    label = (item.get("decision") or {}).get("label")
    return label if label in LABELS else None, item


def main() -> None:
    gold_path = ROOT / "data/private/annotation_v5_gold.jsonl"
    blind_path = ROOT / "data/private/annotation_v5_blind.jsonl"
    manifest = json.loads((RUN / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest["gold_opened_by_runner"] or manifest["blind_sha256"] != sha(blind_path):
        raise ValueError("Blind run provenance mismatch")
    gold = read_jsonl(gold_path)
    blind = read_jsonl(blind_path)
    if len(gold) != 15 or [(x["id"], x["question"]) for x in gold] != [(x["id"], x["question"]) for x in blind]:
        raise ValueError("Gold and blind questions mismatch")
    if {x["gold_label"] for x in gold} != set(LABELS):
        raise ValueError("Unexpected gold labels")
    rows, summaries = [], {}
    for model_id in manifest["models"]:
        folder = RUN / model_id.replace("/", "__")
        if (folder / "model_blocked.json").exists():
            summaries[model_id] = {"status": "BLOCKED", "detail": json.loads((folder / "model_blocked.json").read_text(encoding="utf-8"))}
            continue
        model_rows = []
        for question in gold:
            qid = question["id"]
            paths = {"direct": folder / "direct/decisions" / f"{qid}.json",
                     "s5_matcher": folder / "matcher/decisions" / f"{qid}.json",
                     "s5_reasoner": folder / "reasoner/decisions" / f"{qid}.json"}
            if any(not path.exists() for path in paths.values()):
                raise FileNotFoundError(f"Incomplete predictions for {model_id} {qid}")
            labels, items = {}, {}
            for method, path in paths.items():
                labels[method], items[method] = read_decision(path)
                if items[method].get("question_id") != qid:
                    raise ValueError(f"Decision question ID mismatch: {model_id} {qid} {method}")
            query_path = folder / "matcher/s5queries" / f"{qid}.json"
            if query_path.exists() and json.loads(query_path.read_text(encoding="utf-8")).get("original_question") != question["question"]:
                raise ValueError(f"S5Query question mismatch: {model_id} {qid}")
            model_rows.append({"model": model_id, "question_id": qid,
                "question": question["question"], "gold_label": question["gold_label"],
                **{f"{method}_label": labels[method] for method in METHODS},
                **{f"{method}_correct": labels[method] == question["gold_label"] for method in METHODS},
                "matcher_rule_ids": ",".join((items["s5_matcher"].get("decision") or {}).get("rule_ids") or []),
                "reasoner_rule_ids": ",".join((items["s5_reasoner"].get("decision") or {}).get("rule_ids") or []),
                "reasoner_pages": ",".join(map(str, (items["s5_reasoner"].get("decision") or {}).get("cited_pages") or [])),
                **{f"{method}_error": str(items[method].get("error") or "") for method in METHODS}})
        costs = {"compat": 0.0, "direct": 0.0, "s5query": 0.0, "s5_reasoner": 0.0}
        for path in folder.rglob("*.json"):
            if path.parent.name not in {"raw", "raw_s5query"} and path.name != "preflight_compat.json":
                continue
            item = json.loads(path.read_text(encoding="utf-8"))
            method = item.get("method")
            if method in costs:
                costs[method] += float(item.get("cost_usd") or 0)
        summaries[model_id] = {"status": "COMPLETE", "metrics": {method: score(model_rows, method) for method in METHODS},
                               "api_cost_usd": costs,
                               "method_cost_usd": {"direct": costs["direct"],
                                   "s5_matcher": costs["s5query"],
                                   "s5_reasoner": costs["s5query"] + costs["s5_reasoner"]}}
        rows.extend(model_rows)
    ledger = json.loads((RUN / "budget_ledger.json").read_text(encoding="utf-8"))
    summed = sum(sum(item["api_cost_usd"].values()) for item in summaries.values() if item["status"] == "COMPLETE")
    if abs(summed - ledger["observed_usd"]) > 1e-7:
        raise ValueError("Saved API costs differ from ledger")
    report = {"study": "retrospective_exploratory_v5_multimodel",
              "gold_sha256": sha(gold_path), "blind_sha256": sha(blind_path),
              "n_questions_per_model": 15, "models": summaries,
              "total_observed_api_cost_usd": ledger["observed_usd"],
              "budget_cap_usd": ledger["cap_usd"], "gold_sent_to_models": False,
              "note": "Invalid predictions count as errors. Class metrics do not validate answer content."}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "evaluation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if rows:
        with (OUT / "questions.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({"cost_usd": ledger["observed_usd"], "models": {
        model: {method: (round(item["metrics"][method]["accuracy_all"], 4),
                         item["metrics"][method]["invalid"]) for method in METHODS}
        for model, item in summaries.items() if item["status"] == "COMPLETE"}}, indent=2))


if __name__ == "__main__":
    main()
