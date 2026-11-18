# V5 decision benchmark

This public report summarizes a retrospective experiment on 15 owner-reviewed reference questions from the included [garlic PDF](../../docs/Bestfitmanual_MUCASCAPE_Garlic_Sep2019final.pdf). It compares three DeepInfra models and three decision methods. Individual questions, reference files, model responses and API credentials are kept in ignored local directories. The charts and `summary.json` contain aggregate metrics; the question-level audit chart shows IDs and correctness states only.

## Results

Accuracy counts invalid outputs as errors. Macro-F1 averages `ANSWER`, `ASK` and `ABSTAIN` equally.

| Model | Total parameters | Active in generation | Method | Accuracy | Macro-F1 | Invalid |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| Qwen3.5-9B | 9B | about 9B* | Direct PDF | 7/15 (46.7%) | 0.462 | 0 |
| Qwen3.5-9B | 9B | about 9B* | S5Matcher | 4/15 (26.7%) | 0.205 | 7 |
| Qwen3.5-9B | 9B | about 9B* | S5Reasoner | 2/15 (13.3%) | 0.156 | 7 |
| Qwen3.5-27B | 27B | about 27B* | Direct PDF | 10/15 (66.7%) | 0.679 | 1 |
| Qwen3.5-27B | 27B | about 27B* | S5Matcher | 5/15 (33.3%) | 0.185 | 2 |
| Qwen3.5-27B | 27B | about 27B* | S5Reasoner | 10/15 (66.7%) | 0.690 | 2 |
| DeepSeek-V4.1-Flash-1017 | 552B | 16B decode, 8B prefill | Direct PDF | 11/15 (73.3%) | 0.683 | 0 |
| DeepSeek-V4.1-Flash-1017 | 552B | 16B decode, 8B prefill | S5Matcher | 5/15 (33.3%) | 0.175 | 1 |
| DeepSeek-V4.1-Flash-1017 | 552B | 16B decode, 8B prefill | S5Reasoner | 9/15 (60.0%) | 0.590 | 2 |

*The Qwen active values are inferred from the [documented dense architecture](https://huggingface.co/docs/transformers/model_doc/qwen3_5), not separately published active counts. DeepSeek's [MoE architecture](https://deepinfra.com/blog/deepseek-v4-1-flash-model-overview-integration) has 552B total backbone parameters and activates 16B during decoding. Total and active parameters are plotted separately.

## Figures

1. [Accuracy with descriptive intervals](accuracy_overview.png) compares the three methods and marks invalid counts. Wilson intervals describe uncertainty from 15 questions; they do not establish significance between methods.
2. [Nine confusion matrices](confusion_matrices.png) show each model and method, with `INVALID` as an explicit prediction column.
3. [Recall by class](class_recall.png) separates correct answers, clarification questions and abstentions.
4. [Question outcome audit](question_outcomes.png) shows correct, wrong and invalid results for each question ID across all nine model-method combinations.
5. [Observed cost versus accuracy](cost_vs_accuracy.png) shows each method as an unconnected point. The x-axis is logarithmic.
6. [False answers, missed answers and invalid outputs](safety_and_validity.png) separates class-level risk from lost coverage.
7. [Paired direct versus reasoner outcomes](paired_direct_reasoner.png) shows how often each method alone, both, or neither is correct on the same question.
8. [Accuracy versus total and active parameters](accuracy_vs_parameters.png) avoids equating MoE total size with active size.

Each figure is also available as SVG. [Aggregate data](summary.json) supplies the exact counts and values used to build them.

## Interpretation

The 27B S5Reasoner has the highest Macro-F1 (0.690), while DeepSeek direct has the highest accuracy (11/15) and the lowest observed cost for direct PDF calls. All three direct runs produced four class-level false `ANSWER` predictions on questions whose gold class was `ASK` or `ABSTAIN`. S5Reasoner produced one such false answer for the 27B and one for DeepSeek. The valid S5Matcher decisions were all `ABSTAIN`, so its absence of false answers comes with low coverage of answerable and clarification questions.

The S5 methods have a narrower evidence base than the direct method: the supplied S5 file contains fertilization rules, while some questions ask about planting or other information in the full PDF. A correct decision class does not establish that a dose, timing, citation or clarification text is factually correct. Seven invalid S5Query outputs for the 9B account for much of its poor structured-pipeline result. The sample has only 15 previously inspected questions and one PDF; no causal effect of parameter count can be inferred.

## Output validity and reference provenance

`INVALID` is a pipeline outcome, not a fourth reference class. It means the model response could not be accepted under the fixed output contract. The evaluation counts each such outcome as incorrect and shows it in a separate confusion-matrix column. It is unrelated to a change in reference labels.

| Model | Direct PDF | S5Query / S5Matcher | S5Reasoner |
| --- | --- | --- | --- |
| Qwen3.5-9B | 0 | 7 schema failures: unsupported field values or extra fields (G01, G03, G05, G06, G08, G09, G14) | The same 7 S5Query failures; reasoner calls were skipped |
| Qwen3.5-27B | 1 truncated response (G12) | 2 non-verbatim fact provenance spans (G01, G04) | The same 2 S5Query failures; reasoner calls were skipped |
| DeepSeek-V4.1-Flash-1017 | 0 | 1 non-verbatim fact provenance span (G02) | The G02 S5Query failure and one reasoner citation to an unknown S5 rule (G06) |

The project owner attests to personally reviewing the final 12-rule S5 recommendation file and the final 15-question V5 annotation file. [Reference provenance](../../docs/reference_review.md) binds this attestation to the exact local file hashes. The older R16 editing workspace is a different artifact and its unfinished status does not describe the final benchmark inputs. The files do not contain an independent second-review or per-item adjudication log. These metrics describe agreement with the supplied, owner-reviewed reference labels; they do not independently establish the correctness of every agronomic answer.

## Observed API cost

| Model | Direct PDF | S5Matcher | S5Reasoner |
| --- | ---: | ---: | ---: |
| Qwen3.5-9B | $0.030230 | $0.002843 | $0.013931 |
| Qwen3.5-27B | $0.089031 | $0.024792 | $0.082829 |
| DeepSeek-V4.1-Flash-1017 | $0.003426 | $0.003151 | $0.021624 |

S5Matcher cost is the S5Query call; Python matching adds no API cost. S5Reasoner cost includes that same S5Query call plus its own model call, so these method costs must not be summed. Observed costs include invalid calls and may reflect provider caching. The comparison reused saved DeepSeek outputs and made no new API calls. Prices can change; [DeepInfra's pricing page](https://deepinfra.com/pricing) is the current provider source.

## Rebuild

The ignored local inputs are `results/annotation_v5_three_model_comparison/comparison.json` and `questions.csv`. Run:

```powershell
python scripts/build_v5_dashboard.py --comparison results/annotation_v5_three_model_comparison/comparison.json --questions results/annotation_v5_three_model_comparison/questions.csv --out reports/v5
```

This command is offline. The original raw model outputs, S5 file, reference annotations and included PDF remain separate and unchanged. An earlier Qwen-235B run remains preserved locally because its task outputs failed JSON validation; it is not included in this three-model comparison.
