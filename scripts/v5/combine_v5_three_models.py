"""Combine saved V5 evaluations without API calls or changing source results."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
QWEN = ROOT / "results/annotation_v5_multimodel"
DEEPSEEK = ROOT / "results/annotation_v5_live"
OLD_MATCHER = ROOT / "results/s5_matcher_single_model_deepseek_exploratory"
OLD_REASONER = ROOT / "results/s5_reasoner_single_model_deepseek_exploratory"
OUT = ROOT / "results/annotation_v5_three_model_comparison"
METHODS = ("direct", "s5_matcher", "s5_reasoner")
DEEPSEEK_ID = "deepseek-ai/DeepSeek-V4.1-Flash-1017"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def cost(path: Path) -> float:
    return float(load(path).get("cost_usd") or 0)


def main() -> None:
    qwen = load(QWEN / "evaluation/evaluation.json")
    deepseek = load(DEEPSEEK / "evaluation/evaluation.json")
    qwen_manifest = load(QWEN / "run_manifest.json")
    deepseek_manifest = load(DEEPSEEK / "run_manifest.json")
    for key in ("gold_sha256", "blind_sha256"):
        if qwen[key] != deepseek[key]:
            raise ValueError(f"Different {key}")
    for key in ("blind_sha256", "rules_sha256", "pdf_artifact_sha256",
                "direct_system_prompt_sha256", "direct_user_format"):
        if qwen_manifest[key] != deepseek_manifest[key]:
            raise ValueError(f"Different {key}")
    if qwen_manifest["s5query_prompt_sha256"] != deepseek_manifest["s5_query_prompt_sha256"]:
        raise ValueError("Different S5Query prompt")
    if qwen_manifest["s5reasoner_system_prompt_sha256"] != deepseek_manifest["reasoner_system_prompt_sha256"]:
        raise ValueError("Different reasoner prompt")
    old_matcher_manifest = load(OLD_MATCHER / "run_manifest.json")
    if old_matcher_manifest["s5_sha256"] != deepseek_manifest["rules_sha256"] or \
            old_matcher_manifest["s5_query_prompt_sha256"] != deepseek_manifest["s5_query_prompt_sha256"]:
        raise ValueError("Old DeepSeek S5Query provenance mismatch")
    old_reasoner_manifest = load(OLD_REASONER / "frozen_manifest.json")
    if old_reasoner_manifest["system_prompt_sha256"] != deepseek_manifest["reasoner_system_prompt_sha256"]:
        raise ValueError("Old DeepSeek reasoner prompt mismatch")
    catalog = {model["id"]: model for model in load(ROOT / "data/private/study_code/configs__deepinfra_models_snapshot.json")["models"]}
    selected = ("Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-27B", DEEPSEEK_ID)
    questions = list(csv.DictReader((DEEPSEEK / "evaluation/questions.csv").open(encoding="utf-8")))
    if len(questions) != 15:
        raise ValueError("Expected 15 DeepSeek rows")
    changed = set(deepseek["changed_question_ids"])
    deepseek_costs = {"direct": 0.0, "s5query": 0.0, "s5_reasoner": 0.0}
    for row in questions:
        qid = row["question_id"]
        deepseek_costs["direct"] += cost(DEEPSEEK / "direct/raw" / f"{qid}_direct.json")
        matcher_dir = DEEPSEEK / "matcher" if qid in changed else OLD_MATCHER
        reasoner_dir = DEEPSEEK / "reasoner" if qid in changed else OLD_REASONER
        deepseek_costs["s5query"] += cost(matcher_dir / "raw_s5query" / f"{qid}.json")
        reasoner_path = reasoner_dir / "raw" / f"{qid}.json"
        if reasoner_path.exists():
            deepseek_costs["s5_reasoner"] += cost(reasoner_path)
    models = {}
    for model_id in selected:
        info = catalog[model_id]
        active_decode = info["active_b"] if info["active_b"] is not None else info["total_b"]
        active_basis = "published MoE active parameter count" if info["active_b"] is not None else "inferred from dense architecture"
        if model_id == DEEPSEEK_ID:
            metrics = deepseek["methods"]
            costs = {"direct": deepseek_costs["direct"],
                     "s5_matcher": deepseek_costs["s5query"],
                     "s5_reasoner": deepseek_costs["s5query"] + deepseek_costs["s5_reasoner"]}
            source = "results/annotation_v5_live/evaluation/evaluation.json"
        else:
            metrics = qwen["models"][model_id]["metrics"]
            costs = qwen["models"][model_id]["method_cost_usd"]
            source = "results/annotation_v5_multimodel/evaluation/evaluation.json"
        models[model_id] = {"total_parameters_b": info["total_b"],
                            "active_parameters_decode_b": active_decode,
                            "active_parameters_prefill_b": info.get("prefill_active_b"),
                            "active_parameter_basis": active_basis,
                            "parameter_source": "local DeepInfra catalog snapshot, 2026-11-18; Qwen dense architecture confirmed by Hugging Face documentation",
                            "metrics": metrics, "method_cost_usd": costs,
                            "source_evaluation": source}
    qrows = list(csv.DictReader((QWEN / "evaluation/questions.csv").open(encoding="utf-8")))
    combined = []
    for model_id in selected:
        source_rows = questions if model_id == DEEPSEEK_ID else [r for r in qrows if r["model"] == model_id]
        if len(source_rows) != 15:
            raise ValueError(f"Expected 15 question rows for {model_id}")
        for row in source_rows:
            combined.append({"model": model_id, "question_id": row["question_id"],
                "gold_label": row["gold_label"],
                **{f"{method}_label": row.get(f"{method}_label") for method in METHODS},
                **{f"{method}_correct": row.get(f"{method}_correct") for method in METHODS}})
    report = {"study": "retrospective_exploratory_v5_three_model_comparison",
              "gold_sha256": qwen["gold_sha256"], "blind_sha256": qwen["blind_sha256"],
              "n_questions_per_model": 15, "models": models,
              "api_calls_for_combination": 0,
              "note": "DeepSeek costs include its earlier reused S5Query and reasoner calls. Each method's cost is separate; do not sum them because S5Query is shared by matcher and reasoner."}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "comparison.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (OUT / "questions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(combined[0]))
        writer.writeheader()
        writer.writerows(combined)
    print(json.dumps({model: {"parameters_total_b": models[model]["total_parameters_b"],
                              "parameters_active_decode_b": models[model]["active_parameters_decode_b"],
                              "accuracy": {method: models[model]["metrics"][method]["accuracy_all"] for method in METHODS}}
                      for model in selected}, indent=2))


if __name__ == "__main__":
    main()
