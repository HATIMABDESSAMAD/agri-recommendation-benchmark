"""Build aggregate V5 figures from saved evaluations; no model or gold access.

The question-level CSV supplies IDs and labels only to draw the private local
heatmap. The published summary contains aggregate counts and no question text.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from math import sqrt
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CLASSES = ("ANSWER", "ASK", "ABSTAIN")
PREDICTIONS = (*CLASSES, "INVALID")
METHODS = (("direct", "Direct PDF"), ("s5_matcher", "S5Matcher"),
           ("s5_reasoner", "S5Reasoner"))
COLORS = {"direct": "#2677b8", "s5_matcher": "#d48c24", "s5_reasoner": "#268c6c"}


def wilson(correct: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = correct / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0, center - half), min(1, center + half)


def save(fig, out: Path, stem: str) -> None:
    fig.savefig(out / f"{stem}.png", dpi=180, bbox_inches="tight")
    fig.savefig(out / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline V5 aggregate dashboard")
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.comparison.read_text(encoding="utf-8"))
    rows = list(csv.DictReader(args.questions.open(encoding="utf-8")))
    models = list(report["models"])
    if len(models) != 3 or len(rows) != 45 or len({(r["model"], r["question_id"]) for r in rows}) != 45:
        raise ValueError("Expected three models and 15 unique question rows per model")
    if {r["model"] for r in rows} != set(models):
        raise ValueError("Model IDs differ between comparison and question rows")
    question_ids = {r["question_id"] for r in rows}
    if len(question_ids) != 15:
        raise ValueError("Expected the same 15 question IDs for every model")
    gold_by_id = {}
    for row in rows:
        if row["gold_label"] not in CLASSES:
            raise ValueError("Unknown gold label")
        old = gold_by_id.setdefault(row["question_id"], row["gold_label"])
        if old != row["gold_label"]:
            raise ValueError("Gold labels differ across models")
    if Counter(gold_by_id.values()) != Counter({label: 5 for label in CLASSES}):
        raise ValueError("Expected five questions per gold class")
    for model in models:
        subset = [r for r in rows if r["model"] == model]
        if {r["question_id"] for r in subset} != question_ids:
            raise ValueError(f"Incomplete question IDs for {model}")
        for method, _ in METHODS:
            stats = report["models"][model]["metrics"][method]
            matrix = Counter((r["gold_label"], r[f"{method}_label"] or "INVALID") for r in subset)
            for gold in CLASSES:
                for pred in PREDICTIONS:
                    if matrix[(gold, pred)] != stats["confusion_matrix"][gold][pred]:
                        raise ValueError(f"Question rows and matrix differ for {model} {method}")
    names = ["Qwen3.5-9B", "Qwen3.5-27B", "DeepSeek-V4.1"]
    args.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

    # Accuracy, descriptive Wilson intervals, and invalid output counts.
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    x = np.arange(3)
    width = 0.25
    for offset, (method, label) in enumerate(METHODS):
        stats = [report["models"][model]["metrics"][method] for model in models]
        values = [s["accuracy_all"] for s in stats]
        intervals = [wilson(round(s["accuracy_all"] * s["n"]), s["n"]) for s in stats]
        lower = [v - lo for v, (lo, hi) in zip(values, intervals)]
        upper = [hi - v for v, (lo, hi) in zip(values, intervals)]
        pos = x + (offset - 1) * width
        ax.bar(pos, values, width, color=COLORS[method], label=label)
        ax.errorbar(pos, values, yerr=[lower, upper], fmt="none", capsize=3,
                    color="#303030", linewidth=1)
        for p, value, stat in zip(pos, values, stats):
            ax.text(p, value + 0.025, f"{value:.0%}\n({stat['invalid']} invalid)",
                    ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x, names)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Accuracy on 15 V5 questions")
    ax.set_title("Accuracy with descriptive 95% Wilson intervals")
    ax.legend(loc="upper left", ncol=3)
    save(fig, args.out, "accuracy_overview")

    # Nine confusion matrices, explicitly including invalid predictions.
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), constrained_layout=True)
    for i, model in enumerate(models):
        for j, (method, label) in enumerate(METHODS):
            ax = axes[i, j]
            matrix = report["models"][model]["metrics"][method]["confusion_matrix"]
            values = np.array([[matrix[gold][pred] for pred in PREDICTIONS] for gold in CLASSES])
            ax.imshow(values, cmap="Blues", vmin=0, vmax=5)
            for row in range(3):
                for col in range(4):
                    ax.text(col, row, str(values[row, col]), ha="center", va="center",
                            color="white" if values[row, col] >= 3 else "black")
            ax.set_xticks(range(4), PREDICTIONS, rotation=30, ha="right")
            ax.set_yticks(range(3), CLASSES)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Gold")
            ax.set_title(f"{names[i]} | {label}")
    save(fig, args.out, "confusion_matrices")

    # Class recall and Macro-F1 separate general accuracy from ASK failures.
    labels = [f"{name} | {label}" for name in names for method, label in METHODS]
    recall = np.array([[report["models"][model]["metrics"][method]["recall"][gold] or 0
                        for gold in CLASSES] for model in models for method, _ in METHODS])
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    ax.imshow(recall, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(3), CLASSES)
    ax.set_yticks(range(9), labels)
    ax.set_title("Recall by gold class")
    for i in range(9):
        for j in range(3):
            ax.text(j, i, f"{recall[i, j]:.0%}", ha="center", va="center",
                    color="white" if recall[i, j] >= .7 else "black")
    save(fig, args.out, "class_recall")

    # Private/local question-level audit: C correct, W wrong, I invalid.
    ids = sorted({r["question_id"] for r in rows})
    by_key = {(r["model"], r["question_id"]): r for r in rows}
    states = np.zeros((9, 15), dtype=int)
    for i, model in enumerate(models):
        for j, (method, _) in enumerate(METHODS):
            for k, qid in enumerate(ids):
                row = by_key[(model, qid)]
                pred = row[f"{method}_label"]
                states[i * 3 + j, k] = 0 if not pred else 2 if pred == row["gold_label"] else 1
    from matplotlib.colors import ListedColormap
    fig, ax = plt.subplots(figsize=(14, 5.5), constrained_layout=True)
    ax.imshow(states, aspect="auto", cmap=ListedColormap(["#b7bec8", "#d97465", "#5aa67d"]),
              vmin=0, vmax=2)
    ax.set_xticks(range(15), ids)
    ax.set_yticks(range(9), labels)
    ax.set_title("Question outcomes: C correct, W wrong, I invalid")
    for i in range(9):
        for j in range(15):
            ax.text(j, i, ("I", "W", "C")[states[i, j]], ha="center", va="center", fontsize=8)
    save(fig, args.out, "question_outcomes")

    # Cost is observed API spend, including failed calls; no lines join models.
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    for method, label in METHODS:
        costs = [report["models"][model]["method_cost_usd"][method] for model in models]
        accs = [report["models"][model]["metrics"][method]["accuracy_all"] for model in models]
        ax.scatter(costs, accs, color=COLORS[method], s=80, label=label)
        for name, cost, accuracy in zip(names, costs, accs):
            offset = (5, 13) if name == "Qwen3.5-27B" and method == "direct" else \
                     (5, -15) if name == "Qwen3.5-27B" and method == "s5_reasoner" else (5, 4)
            ax.annotate(name, (cost, accuracy), xytext=offset, textcoords="offset points", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Observed API cost per method (USD, log scale)")
    ax.set_ylabel("Accuracy on 15 V5 questions")
    ax.set_ylim(0, 1)
    ax.set_title("Cost versus accuracy; S5Query cost is shared by S5 methods")
    ax.legend()
    ax.grid(alpha=.2)
    save(fig, args.out, "cost_vs_accuracy")

    # Safety and coverage measures are distinct counts and should not be stacked.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    x = np.arange(9)
    false_answer = []
    missed_answer = []
    invalid = []
    for model in models:
        for method, _ in METHODS:
            stats = report["models"][model]["metrics"][method]
            false_answer.append(stats["false_answer_on_ask_or_abstain"])
            missed_answer.append(5 - stats["confusion_matrix"]["ANSWER"]["ANSWER"])
            invalid.append(stats["invalid"])
    axes[0].bar(x - .19, false_answer, .38, label="False ANSWER", color="#c85a50")
    axes[0].bar(x + .19, missed_answer, .38, label="Missed gold ANSWER", color="#7a87a8")
    axes[0].set_ylabel("Questions (out of 15)")
    axes[0].set_title("Class-level safety and missed coverage")
    axes[0].legend()
    axes[1].bar(x, invalid, color="#8a78a8")
    axes[1].set_title("Invalid pipeline outputs")
    axes[1].set_ylabel("Questions (out of 15)")
    for ax in axes:
        ax.set_xticks(x, [f"{name}\n{label}" for name in names for _, label in METHODS],
                      rotation=45, ha="right", fontsize=7)
        ax.set_ylim(0, 8)
    save(fig, args.out, "safety_and_validity")

    # Paired direct versus reasoner outcomes per model; neither includes invalids.
    categories = ("Both correct", "Direct only", "Reasoner only", "Neither correct")
    palette = ("#5aa67d", "#2677b8", "#d48c24", "#b7bec8")
    paired = []
    for model in models:
        subset = [r for r in rows if r["model"] == model]
        counts = [0, 0, 0, 0]
        for row in subset:
            direct = row["direct_label"] == row["gold_label"]
            reasoner = row["s5_reasoner_label"] == row["gold_label"]
            counts[0 if direct and reasoner else 1 if direct else 2 if reasoner else 3] += 1
        paired.append(counts)
    fig, ax = plt.subplots(figsize=(9, 4.3), constrained_layout=True)
    left = np.zeros(3)
    for i, (category, color) in enumerate(zip(categories, palette)):
        values = np.array([counts[i] for counts in paired])
        ax.barh(names, values, left=left, color=color, label=category)
        for row, (start, value) in enumerate(zip(left, values)):
            if value:
                ax.text(start + value / 2, row, str(value), ha="center", va="center")
        left += values
    ax.set_xlim(0, 15)
    ax.set_xlabel("Questions")
    ax.set_title("Paired direct versus S5Reasoner decisions")
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(.5, 1.02), fontsize=8)
    save(fig, args.out, "paired_direct_reasoner")

    # Total size and active decode size are separate axes; MoE is not 552B active.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for ax, key, title in ((axes[0], "total_parameters_b", "Total parameters (billions)"),
                           (axes[1], "active_parameters_decode_b", "Active during generation (billions)")):
        for method, label in METHODS:
            xs = [report["models"][model][key] for model in models]
            ys = [report["models"][model]["metrics"][method]["accuracy_all"] for model in models]
            ax.scatter(xs, ys, s=75, color=COLORS[method], label=label)
            for name, a, b in zip(names, xs, ys):
                ax.annotate(name, (a, b), xytext=(4, 3), textcoords="offset points", fontsize=8)
        ax.set_xlim(left=0)
        ax.set_ylim(0, 1)
        ax.set_xlabel(title)
        ax.set_ylabel("Accuracy")
        ax.grid(alpha=.2)
    axes[0].set_title("Performance versus total model size")
    axes[1].set_title("Performance versus active model size")
    axes[1].set_xticks([0, 9, 16, 27])
    axes[0].legend()
    save(fig, args.out, "accuracy_vs_parameters")

    public = {"study": report["study"], "n_questions_per_model": 15,
              "source_comparison_sha256": hashlib.sha256(args.comparison.read_bytes()).hexdigest(),
              "source_question_rows_sha256": hashlib.sha256(args.questions.read_bytes()).hexdigest(),
              "source_gold_sha256": report["gold_sha256"],
              "models": {model: {"total_parameters_b": data["total_parameters_b"],
                 "active_parameters_decode_b": data["active_parameters_decode_b"],
                 "active_parameters_prefill_b": data["active_parameters_prefill_b"],
                 "active_parameter_basis": data["active_parameter_basis"],
                 "methods": {method: {"accuracy": data["metrics"][method]["accuracy_all"],
                    "macro_f1": data["metrics"][method]["macro_f1"],
                    "recall": data["metrics"][method]["recall"],
                    "confusion_matrix": data["metrics"][method]["confusion_matrix"],
                    "invalid": data["metrics"][method]["invalid"],
                    "false_answer_on_ask_or_abstain": data["metrics"][method]["false_answer_on_ask_or_abstain"],
                    "observed_api_cost_usd": data["method_cost_usd"][method]} for method, _ in METHODS}}
                 for model, data in report["models"].items()},
              "limitations": ["15 previously inspected questions", "class labels do not validate answer content",
                              "invalid outputs count as incorrect", "method costs share S5Query calls",
                              "reviewer identity and adjudication are not documented in the supplied reference and S5 files"]}
    (args.out / "summary.json").write_text(json.dumps(public, indent=2) + "\n", encoding="utf-8")
    print(f"Created 8 chart families and summary in {args.out}")


if __name__ == "__main__":
    main()
