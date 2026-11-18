# Agricultural recommendation extraction

This project extracts recommendations from agricultural PDF manuals, validates them as structured rules, and evaluates question-answering decisions against a separately supplied reference set. Review provenance must be documented separately before describing a reference set as human adjudicated.

## Components

- PDF preparation pipelines P0–P4 preserve page text, tables, source positions and numeric audit information.
- Pydantic schemas S1–S5 represent actions, quantities, conditions, exceptions, citations and dependencies.
- Extraction strategies E1–E8 and validation workflows V0–V5 retain raw outputs and derived results separately.
- `S5Query` records the request and explicitly stated farmer facts without assigning a decision label.
- `S5Matcher` applies deterministic Python matching to supplied S5 rules.
- `S5Reasoner` validates a language model's S5-only decision and its cited rule IDs and pages.

The code is general-purpose. PDF manuals, S5 rule files, annotated questions, credentials and generated results are local inputs and are excluded from the public repository. Example configurations contain no credentials. The original PDF files are never modified by the pipeline.

To register PDF manuals, copy `configs/documents.example.json` to `configs/documents.local.json` and set each local file path and metadata. The local configuration is ignored by Git. You can also point `AGRI_DOCUMENT_CONFIG` at another configuration file.

## Install and test

Python 3.13 is used for the current environment:

```powershell
python -m pip install -r requirements.txt
python -m pytest -q
python scripts/smoke_all_variants.py
```

The test suite uses synthetic fixtures and passes without local PDF files or API credentials. The smoke script uses the locally configured documents and their prepared artifacts. Matching and query tests can also be run separately:

```powershell
python -m pytest tests/test_s5_matcher.py tests/test_s5_reasoner.py tests/test_s5_query_pipeline.py -q
```

## S5 question workflow

Prepare a validated S5 RuleSet JSON and a question. The first command creates messages for a caller-managed language model; it does not send them. Save the model's `S5Query` JSON, then run the deterministic matcher:

```powershell
python -m agri_extract.s5_matcher_cli freeze --rules path/to/verified_rules.json
python -m agri_extract.s5_matcher_cli messages --rules path/to/verified_rules.json --question "What is recommended for this field?" --out messages.json
python -m agri_extract.s5_matcher_cli decide --rules path/to/verified_rules.json --query query.json --out decision.json
```

The [S5 query guide](docs/S5_MATCHER_PIPELINE.md) describes the input contract and known representation limits. Independent evaluation requires an annotated question set fixed before model requests.

## Research results

The [V5 three-model report](reports/v5/README.md) includes aggregate metrics, confusion matrices for every model and method, question outcome charts, cost and parameter comparisons. Its figure generator is `scripts/build_v5_dashboard.py`; raw model responses, the PDF, S5 rules and human labels remain local ignored inputs.

## Repository hygiene

`.gitignore` excludes local PDFs, S5 rules, reference annotations, credentials, model responses and generated artifacts. Keep API keys in a private process environment or terminal prompt. No publication or deployment is performed by the project commands.
