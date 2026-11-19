"""Verify the published V5 decision benchmark from public saved files only.

No model is called and no result file is written. The historical scoring
function is imported unchanged from rescore_saved_questions.py.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from math import isclose
from pathlib import Path

from rescore_saved_questions import LABELS, METHODS, score


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/v5"
INPUTS = DATA / "inputs"
SAVED = DATA / "saved/results"
SUMMARY = ROOT / "reports/v5/summary.json"
CHANGED = {"G01", "G04", "G06", "G08"}
MODEL_DIRS = {
    "Qwen/Qwen3.5-9B": "Qwen__Qwen3.5-9B",
    "Qwen/Qwen3.5-27B": "Qwen__Qwen3.5-27B",
}


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def same_number(actual: float, expected: float, name: str) -> None:
    require(isclose(actual, expected, rel_tol=0, abs_tol=1e-11),
            f"{name}: {actual!r} != {expected!r}")


def rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def decision_label(path: Path, question_id: str) -> str | None:
    item = read(path)
    require(item.get("question_id") == question_id, f"Wrong question ID in {path}")
    value = (item.get("decision") or {}).get("label")
    return value if value in LABELS else None


def cost(path: Path) -> float:
    return float(read(path).get("cost_usd") or 0)


def compare_metrics(computed: dict, saved: dict, name: str) -> None:
    for field in ("n", "valid", "invalid", "confusion_matrix", "false_answer_on_ask_or_abstain"):
        require(computed[field] == saved[field], f"{name}: {field} differs")
    for field in ("accuracy_all", "macro_f1"):
        same_number(computed[field], saved[field], f"{name}: {field}")
    for label in LABELS:
        same_number(computed["recall"][label], saved["recall"][label],
                    f"{name}: recall {label}")


def verify_decisions(model: str, question: dict, labels: dict, changed: set[str]) -> None:
    qid = question["id"]
    if model in MODEL_DIRS:
        base = SAVED / "annotation_v5_multimodel" / MODEL_DIRS[model]
        paths = {
            "direct": base / "direct/decisions" / f"{qid}.json",
            "s5_matcher": base / "matcher/decisions" / f"{qid}.json",
            "s5_reasoner": base / "reasoner/decisions" / f"{qid}.json",
        }
        query_path = base / "matcher/s5queries" / f"{qid}.json"
    else:
        live = SAVED / "annotation_v5_live"
        matcher = (live / "matcher" if qid in changed else
                   SAVED / "s5_matcher_single_model_deepseek_exploratory")
        reasoner = (live / "reasoner" if qid in changed else
                    SAVED / "s5_reasoner_single_model_deepseek_exploratory")
        paths = {
            "direct": live / "direct/decisions" / f"{qid}.json",
            "s5_matcher": matcher / "decisions" / f"{qid}.json",
            "s5_reasoner": reasoner / "decisions" / f"{qid}.json",
        }
        query_path = matcher / "s5queries" / f"{qid}.json"
    for method, path in paths.items():
        require(decision_label(path, qid) == labels[method],
                f"{model} {qid} {method}: per-item decision differs")
    if query_path.exists():
        require(read(query_path).get("original_question") == question["question"],
                f"{model} {qid}: saved S5Query question differs")


def verify_costs(model: str, question_ids: list[str], saved_costs: dict,
                 changed: set[str]) -> None:
    if model in MODEL_DIRS:
        base = SAVED / "annotation_v5_multimodel" / MODEL_DIRS[model]
        direct = sum(cost(base / "direct/raw" / f"{qid}.json") for qid in question_ids)
        query = sum(cost(base / "matcher/raw_s5query" / f"{qid}.json") for qid in question_ids)
        reasoner = sum(cost(path) for path in (base / "reasoner/raw").glob("*.json"))
    else:
        live = SAVED / "annotation_v5_live"
        old_matcher = SAVED / "s5_matcher_single_model_deepseek_exploratory"
        old_reasoner = SAVED / "s5_reasoner_single_model_deepseek_exploratory"
        direct = sum(cost(live / "direct/raw" / f"{qid}_direct.json") for qid in question_ids)
        query = sum(cost((live / "matcher" if qid in changed else old_matcher) /
                         "raw_s5query" / f"{qid}.json") for qid in question_ids)
        reasoner = sum(cost(path) for qid in question_ids
                       if (path := (live / "reasoner" if qid in changed else old_reasoner) /
                           "raw" / f"{qid}.json").exists())
    for method, value in (("direct", direct), ("s5_matcher", query),
                          ("s5_reasoner", query + reasoner)):
        same_number(value, saved_costs[method], f"{model}: {method} cost")


def main() -> None:
    gold_path = INPUTS / "annotation_v5_gold.jsonl"
    blind_path = INPUTS / "annotation_v5_blind.jsonl"
    rules_path = INPUTS / "verified_s5_rules.json"
    original_path = INPUTS / "original_questions_blind.jsonl"
    comparison_path = SAVED / "annotation_v5_three_model_comparison/comparison.json"
    questions_path = SAVED / "annotation_v5_three_model_comparison/questions.csv"
    comparison = read(comparison_path)
    summary = read(SUMMARY)
    gold = jsonl(gold_path)
    blind = jsonl(blind_path)
    original = {item["id"]: item for item in jsonl(original_path)}
    require(len(gold) == len(blind) == len(original) == 15, "Expected 15 questions")
    require(Counter(item["gold_label"] for item in gold) == Counter({label: 5 for label in LABELS}),
            "Expected five questions in each reference class")
    require([(q["id"], q["question"]) for q in gold] ==
            [(q["id"], q["question"]) for q in blind], "Gold and blind questions differ")
    require({q["id"] for q in gold if q["question"] != original[q["id"]]["question"]} == CHANGED,
            "The V5 changed-question set differs")
    for path, expected in ((gold_path, comparison["gold_sha256"]),
                           (blind_path, comparison["blind_sha256"])):
        require(sha(path) == expected, f"Input hash differs: {path}")
    require(sha(rules_path) == read(SAVED / "annotation_v5_live/run_manifest.json")["rules_sha256"],
            "Final S5 rule hash differs")
    require(summary["source_comparison_sha256"] == sha(comparison_path) and
            summary["source_question_rows_sha256"] == sha(questions_path) and
            summary["source_gold_sha256"] == sha(gold_path), "Public summary source hashes differ")
    saved_rows = rows(questions_path)
    require(len(saved_rows) == 45, "Expected 45 model-question rows")
    by_id = {q["id"]: q for q in gold}
    require(set(by_id) == {row["question_id"] for row in saved_rows}, "Question IDs differ")
    source_qwen = read(SAVED / "annotation_v5_multimodel/evaluation/evaluation.json")
    source_deepseek = read(SAVED / "annotation_v5_live/evaluation/evaluation.json")
    selected = tuple(comparison["models"])
    require(len(selected) == 3, "Expected three selected models")
    output = []
    for model in selected:
        model_rows = [row for row in saved_rows if row["model"] == model]
        require(len(model_rows) == 15 and {row["question_id"] for row in model_rows} == set(by_id),
                f"Incomplete rows for {model}")
        source = (source_qwen["models"][model]["metrics"] if model in MODEL_DIRS else
                  source_deepseek["methods"])
        for row in model_rows:
            question = by_id[row["question_id"]]
            require(row["gold_label"] == question["gold_label"],
                    f"{model} {row['question_id']}: reference label differs")
            labels = {method: row[f"{method}_label"] or None for method in METHODS}
            verify_decisions(model, question, labels, CHANGED)
        for method in METHODS:
            computed = score(model_rows, method)
            compare_metrics(computed, source[method], f"{model} {method} source")
            compare_metrics(computed, comparison["models"][model]["metrics"][method],
                            f"{model} {method} comparison")
            published = summary["models"][model]["methods"][method]
            same_number(computed["accuracy_all"], published["accuracy"],
                        f"{model} {method} published accuracy")
            same_number(computed["macro_f1"], published["macro_f1"],
                        f"{model} {method} published macro-F1")
            require(computed["confusion_matrix"] == published["confusion_matrix"] and
                    computed["invalid"] == published["invalid"],
                    f"{model} {method}: published counts differ")
            output.append((model, method, computed["accuracy_all"], computed["macro_f1"],
                           computed["invalid"]))
        verify_costs(model, list(by_id), comparison["models"][model]["method_cost_usd"], CHANGED)
        for method in METHODS:
            same_number(comparison["models"][model]["method_cost_usd"][method],
                        summary["models"][model]["methods"][method]["observed_api_cost_usd"],
                        f"{model} {method} published cost")
    print("V5 offline reproduction: PASS (15 questions, 3 models, 3 methods; no API calls)")
    print("model | method | accuracy | macro-F1 | invalid")
    for model, method, accuracy, macro_f1, invalid in output:
        print(f"{model} | {method} | {accuracy:.4f} | {macro_f1:.4f} | {invalid}")
    print("Saved per-item decisions, source evaluations, published metrics and API costs agree.")


if __name__ == "__main__":
    main()
