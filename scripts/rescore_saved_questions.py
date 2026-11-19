"""Rescore saved predictions only when the question text matches exactly.

This script never calls a model and never alters saved predictions or annotations.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


LABELS = ("ANSWER", "ASK", "ABSTAIN")
METHODS = ("direct", "s5_matcher", "s5_reasoner")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def score(rows: list[dict], method: str) -> dict:
    matrix = {gold: {pred: 0 for pred in (*LABELS, "INVALID")} for gold in LABELS}
    for row in rows:
        gold = row["gold_label"]
        pred = row[f"{method}_label"]
        matrix[gold][pred if pred in LABELS else "INVALID"] += 1
    n = len(rows)
    correct = sum(matrix[label][label] for label in LABELS)
    recalls, f1s = {}, []
    for label in LABELS:
        tp = matrix[label][label]
        support = sum(matrix[label].values())
        predicted = sum(matrix[gold][label] for gold in LABELS)
        recall = tp / support if support else None
        precision = tp / predicted if predicted else 0.0
        recalls[label] = recall
        f1s.append(2 * precision * recall / (precision + recall) if recall and precision else 0.0)
    return {"n": n, "valid": n - sum(matrix[label]["INVALID"] for label in LABELS),
            "invalid": sum(matrix[label]["INVALID"] for label in LABELS),
            "accuracy_all": correct / n if n else None,
            "macro_f1": sum(f1s) / 3,
            "recall": recalls,
            "confusion_matrix": matrix,
            "false_answer_on_ask_or_abstain": sum(matrix[label]["ANSWER"] for label in ("ASK", "ABSTAIN"))}


def saved_direct(path: Path) -> tuple[str | None, float]:
    item = json.loads(path.read_text(encoding="utf-8"))
    try:
        content = item["raw"]["choices"][0]["message"]["content"].strip()
        if content.startswith("```json"):
            content = content[7:].strip().removesuffix("```").strip()
        label = json.loads(content).get("label")
    except (KeyError, IndexError, TypeError, ValueError):
        label = None
    return label if label in LABELS else None, float(item.get("cost_usd") or 0)


def saved_decision(path: Path) -> str | None:
    item = json.loads(path.read_text(encoding="utf-8"))
    label = (item.get("decision") or {}).get("label")
    return label if label in LABELS else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline rescore with exact question matching")
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--blind", type=Path, required=True)
    parser.add_argument("--prior-questions", type=Path, required=True)
    parser.add_argument("--direct-dir", type=Path, required=True)
    parser.add_argument("--matcher-dir", type=Path, required=True)
    parser.add_argument("--reasoner-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    gold = read_jsonl(args.gold)
    blind = read_jsonl(args.blind)
    prior = read_jsonl(args.prior_questions)
    if len(gold) != 15 or len(blind) != 15 or len(prior) != 15:
        raise ValueError("Expected exactly 15 annotated, 15 blind and 15 prior questions")
    if len({q["id"] for q in gold}) != 15 or {q["id"] for q in gold} != {q["id"] for q in prior}:
        raise ValueError("Question IDs are missing or duplicated")
    if [(q["id"], q["question"]) for q in gold] != [(q["id"], q["question"]) for q in blind]:
        raise ValueError("Gold and blind question texts differ")
    old = {q["id"]: q for q in prior}
    rows = []
    for question in gold:
        qid = question["id"]
        label = question["gold_label"]
        if label not in LABELS:
            raise ValueError(f"Unknown gold label for {qid}: {label}")
        direct_path = args.direct_dir / f"{qid}_direct.json"
        matcher_path = args.matcher_dir / "decisions" / f"{qid}.json"
        reasoner_path = args.reasoner_dir / "decisions" / f"{qid}.json"
        for path in (direct_path, matcher_path, reasoner_path):
            if not path.exists():
                raise FileNotFoundError(path)
        direct, cost = saved_direct(direct_path)
        same_question = question["question"] == old[qid]["question"]
        rows.append({"question_id": qid, "gold_label": label,
                     "question_unchanged": same_question,
                     "direct_label": direct,
                     "s5_matcher_label": saved_decision(matcher_path),
                     "s5_reasoner_label": saved_decision(reasoner_path),
                     "direct_saved_cost_usd": cost})
    comparable = [row for row in rows if row["question_unchanged"]]
    report = {"kind": "offline_retrospective_rescore", "gold_sha256": sha(args.gold),
              "blind_sha256": sha(args.blind),
              "prior_questions_sha256": sha(args.prior_questions),
              "n_gold": len(gold), "n_exact_question_matches": len(comparable),
              "changed_question_ids": [row["question_id"] for row in rows if not row["question_unchanged"]],
              "comparable": {method: score(comparable, method) for method in METHODS},
              "saved_direct_api_cost_usd_all_15": sum(row["direct_saved_cost_usd"] for row in rows),
              "note": "Only exact question matches are scored. No new API calls or predictions were made."}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "rescore.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (args.out / "questions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"exact_matches": len(comparable),
                      "changed": report["changed_question_ids"],
                      "accuracy": {method: report["comparable"][method]["accuracy_all"]
                                   for method in METHODS}}, indent=2))


if __name__ == "__main__":
    main()
