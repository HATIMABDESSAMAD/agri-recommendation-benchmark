# Public V5 benchmark data

This directory makes the published garlic decision benchmark inspectable and its saved scores reproducible without a model account. It contains unchanged copies of the original local inputs and result files. The main offline command is:

```powershell
python scripts/reproduce_v5_offline.py
```

Run it from the repository root. It uses only Python's standard library, reads saved files, verifies hashes and per-item decisions, recomputes accuracy, macro-F1, recall, invalid counts, confusion matrices and observed API costs for the three published models, and compares them to `reports/v5/summary.json`. It writes no files and makes no API calls.

`FILE_MANIFEST.tsv` lists every other file in this directory with its SHA-256 digest for a complete file inventory.

## Inputs

| Path | Meaning |
| --- | --- |
| `inputs/annotation_v5_blind.jsonl` | The 15 questions without reference labels, used for model requests. |
| `inputs/annotation_v5_gold.jsonl` | The same questions with owner-reviewed reference labels and rationales. |
| `inputs/verified_s5_rules.json` | The final 12-rule garlic S5 set supplied to both structured methods. |
| `inputs/original_questions_blind.jsonl` | Earlier question text used to identify the four V5 revisions. |
| `inputs/deepinfra_models_snapshot.json` | Saved model IDs, size and price metadata for the original comparison; it is a historical snapshot. |

The final S5 file has SHA-256 `62fda994ddefae8dc6a5528fca2e9d327502811c1cb14a9a84487366a1e404db`. The V5 gold file has SHA-256 `aedf762f4d32e61f1332c4ecf9effc88d6ad169a97f99d4c52cec0245b2e9729`. The blind file has SHA-256 `27cc67e7697dacd249ed95ae1cb8f816b82bdb078ab09ca87d14b4b7e595c6a2`.

## Saved results

`saved/results/` preserves the original campaign layout so every saved source file can be compared by hash with its local original:

| Campaign | Relevant contents |
| --- | --- |
| `annotation_v5_multimodel/` | Qwen 9B and 27B direct responses, S5Query responses, S5Matcher and S5Reasoner decisions, run manifest, budget ledger, source evaluation and per-item CSV. The excluded Qwen 235B attempt is described in the public report but is not part of the final three-model result. |
| `annotation_v5_live/` | DeepSeek V5 direct responses for all 15 questions, new structured responses for the four revised questions, run manifest, ledger, source evaluation and per-item CSV. |
| `s5_matcher_single_model_deepseek_exploratory/` and `s5_reasoner_single_model_deepseek_exploratory/` | Saved DeepSeek S5Query and reasoner responses reused for the 11 unchanged questions, plus their provenance manifests. |
| `annotation_v5_three_model_comparison/` | Final `comparison.json` and 45-row `questions.csv`, copied unchanged. |
| `direct_baseline_archive/manifest.json` | Earlier direct-baseline identity checked by the V5 runners. |

The `questions.csv` file records a prediction for each model, method and question ID. A blank predicted label represents `INVALID`; it is counted as incorrect, never treated as a fourth reference class. Saved per-item JSON files retain the actual response and error details. S5Matcher and S5Reasoner share the S5Query request, so their method costs must not be summed as independent calls.

## Producer code and prompts

`scripts/v5/` contains copies of the historical V5 model runners, evaluators, combiner and plotting script. The two runner copies change only the import location of the copied `agri_extract/baseline_support.py`; experiment logic, prompts, scores and saved results are unchanged. `scripts/rescore_saved_questions.py` is the unchanged historical scoring helper used by the new offline verifier. The S5Query prompt is `configs/s5_query_system_prompt_v1.txt`, the S5Reasoner prompt is in `agri_extract/s5_reasoner.py`, and the direct baseline prompt is in `agri_extract/baseline_support.py`. The original research run used private path names; `scripts/reproduce_v5_offline.py` is the public path for recomputing the reported saved results. Do not run the historical model runners for offline reproduction: they are API clients.

To rebuild the published charts after installing the analysis dependencies:

```powershell
python scripts/build_v5_dashboard.py --comparison data/v5/saved/results/annotation_v5_three_model_comparison/comparison.json --questions data/v5/saved/results/annotation_v5_three_model_comparison/questions.csv --out reports/v5
```

The chart builder makes no API calls. It writes derived figures and `summary.json`, so use a separate `--out` directory when checking byte-for-byte preservation of the published files.

## Scope and limits

The source is one 24-page garlic PDF and 15 previously inspected questions. The direct method saw the prepared text of the whole PDF; the S5 methods saw the selected fertilizer rules. The owner attested to reviewing the final S5 and V5 annotation files; an independent second review and per-item adjudication log are not documented. Correct decision labels do not prove that response text, dose or citation is agronomically correct. The public saved files reproduce the **recorded** benchmark; they do not recreate the exact external model service or a new independent experiment.

The exact P4 prepared PDF artifact was not added here because a prepublication scan found email addresses in its extracted source text. The original licensed PDF, preparation code and document configuration are public, so the document can be prepared locally; parser versions and layout behavior may change. This omission limits byte-for-byte replay of the original model request. No credential values were found in the files copied into this directory.
