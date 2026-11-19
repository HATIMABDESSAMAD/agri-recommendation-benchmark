"""Plot the saved three-model V5 comparison without API calls."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/annotation_v5_three_model_comparison"
report = json.loads((OUT / "comparison.json").read_text(encoding="utf-8"))
models = list(report["models"])
short = ["Qwen3.5-9B", "Qwen3.5-27B", "DeepSeek-V4.1-Flash-1017"]
methods = (("direct", "Direct PDF"), ("s5_matcher", "S5Matcher"),
           ("s5_reasoner", "S5Reasoner"))
colors = ("#2677b8", "#d48c24", "#268c6c")

fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), constrained_layout=True,
                         gridspec_kw={"width_ratios": [1.35, 1, 1]})
x = np.arange(len(models))
width = 0.25
for offset, ((method, title), color) in enumerate(zip(methods, colors)):
    values = [report["models"][model]["metrics"][method]["accuracy_all"] for model in models]
    positions = x + (offset - 1) * width
    axes[0].bar(positions, values, width, label=title, color=color)
    for pos, value in zip(positions, values):
        axes[0].text(pos, value + 0.015, f"{value:.0%}", ha="center", fontsize=8)
axes[0].set_xticks(x, ["Qwen3.5-9B\n9B total / 9B active*",
                       "Qwen3.5-27B\n27B total / 27B active*",
                       "DeepSeek-V4.1\n552B total / 16B active"])
axes[0].set_ylim(0, 1)
axes[0].set_ylabel("Accuracy on 15 V5 questions")
axes[0].set_title("Accuracy by model and method")
axes[0].legend(fontsize=8)
for ax, key, title in ((axes[1], "total_parameters_b", "Total parameters (B)"),
                       (axes[2], "active_parameters_decode_b", "Active parameters in generation (B)")):
    for (method, label), color in zip(methods, colors):
        xs = [report["models"][model][key] for model in models]
        ys = [report["models"][model]["metrics"][method]["accuracy_all"] for model in models]
        ax.scatter(xs, ys, label=label, color=color, s=65)
        for model_name, value_x, value_y in zip(short, xs, ys):
            ax.annotate(model_name, (value_x, value_y), xytext=(4, 3),
                        textcoords="offset points", fontsize=7)
    ax.set_xlabel(title)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.set_xlim(left=0)
    ax.grid(alpha=0.2)
axes[2].set_xticks([0, 9, 16, 27])
fig.suptitle("V5: three models, three methods; no connecting lines")
fig.text(0.01, 0.005, "* Dense-model active count is inferred from architecture. DeepSeek is MoE: 16B decode, 8B prefill.", fontsize=8)
fig.savefig(OUT / "accuracy_by_parameters.png", dpi=180)
fig.savefig(OUT / "accuracy_by_parameters.svg")
plt.close(fig)

classes = ("ANSWER", "ASK", "ABSTAIN")
columns = (*classes, "INVALID")
fig, axes = plt.subplots(3, 3, figsize=(14, 11), constrained_layout=True)
for row, model in enumerate(models):
    for col, (method, title) in enumerate(methods):
        ax = axes[row, col]
        matrix = np.array([[report["models"][model]["metrics"][method]["confusion_matrix"][gold][pred]
                            for pred in columns] for gold in classes])
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=5)
        for i in range(3):
            for j in range(4):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                        color="white" if matrix[i, j] >= 3 else "black", fontsize=10)
        ax.set_xticks(range(4), columns, rotation=30, ha="right")
        ax.set_yticks(range(3), classes)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Gold")
        ax.set_title(f"{short[row]} | {title}", fontsize=10)
fig.savefig(OUT / "confusion_matrices.png", dpi=180)
fig.savefig(OUT / "confusion_matrices.svg")
plt.close(fig)
print(OUT)
