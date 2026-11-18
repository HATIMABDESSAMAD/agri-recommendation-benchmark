# Annotation guide: fertilization recommendations

This guide governs the full-PDF extraction gold templates for each crop. The separately owner-reviewed final garlic S5 file and V5 question annotations used in the decision benchmark are different inputs; see [reference provenance](reference_review.md). The extraction templates' review status should not be used to infer the status of those final benchmark inputs.

Annotate each original PDF page by page. LLM output is not a gold standard. Mark every page as reviewed, including pages without recommendations. Preserve annotator disagreements until adjudication. The original PDFs must remain unchanged.

## Atomic unit

Create one rule for each independent application or prohibition. Two products, doses, or distinct application times produce two rules when the document prescribes two actions. A split dose produces a rule for each application, linked to the same operation in S5. Several conditions governing a single application remain one rule.

## Fields

- **Action and modality:** Transcribe the prescription and distinguish MUST, SHOULD, MAY, MUST_NOT, and UNSPECIFIED. Do not turn trial observations or described farmer practices into advice.
- **Product, nutrient, and quantity:** Preserve the exact wording, unit, basis (commercial product, N, P, etc.), and interval. A missing quantity remains `null`, never zero.
- **Conditions:** Represent ALL, ANY, NOT, strict thresholds, and inclusive bounds explicitly. Conditions inherited from a heading receive `INHERITED_FROM_CONTEXT` and a source citation.
- **Exceptions:** Store them separately from conditions. An explicit prohibition is more than an absence of advice.
- **Timing:** Record dates, stages, and frequencies as written. Do not convert a local calendar without supporting evidence.
- **Tables and notes:** Link a cell to its row, column, caption, and page. Cite a note-based exception separately.
- **Repetitions and cross-references:** Preserve occurrences when they differ; link rules after review. Do not infer an unstated hierarchy among passages.
- **Ambiguity:** Retain the original passage and mark `AMBIGUOUS`. Do not silently choose one value when source numbers conflict.
- **Difficulty:** Use only categories actually present in `difficulty_tags` (table, note, AND/OR, threshold, dependency, etc.). An absent category is not evaluated.

## Quality control

A gold set becomes `ADJUDICATED` only after every page has been reviewed and disagreements have documented resolutions. Record pages without rules. Choose development, validation, and test splits before tuning matching thresholds. New documents start as `unassigned`.

Use the hash-named `artifacts/gold/<document_id>_<hash8>.json` template, which follows `GoldAgriculturalRuleSet`. Add human annotator names and set `reviewed_pages` to every PDF page exactly once after reviewing them. Each rule must include a quotation from the original PDF with the correct page. Record each disagreement in `annotation_disagreements` and resolve it with a written `resolution` before setting `status` to `ADJUDICATED`. When only one person annotates, document the limitation in `single_annotator_limitation`. Do not copy model output into gold without an independent source review.

Check a draft without changing it:

```powershell
python -m agri_extract gold-check --artifact artifacts/documents/example_P4.json --gold artifacts/gold/example.json
```

`NOT_READY` identifies missing review steps or source-reference problems. Real-run commands accept `--gold` only after this check returns `READY`; without gold, extraction still runs but gold-based metrics remain unavailable.

Compare each quantity, interval, unit, table note and citation against the original page. Record discrepancies without silently changing the source or treating a parsing discrepancy as a recommendation.
