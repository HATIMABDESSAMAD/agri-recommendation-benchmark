# S5 question and decision pipeline

The workflow is `question → S5Query extraction → S5 rules → decision`. The extraction model returns a structured question only. It receives the question, the `S5Query` JSON schema and vocabulary derived from the supplied S5 RuleSet; it does not receive human labels or choose a final class.

`agri_extract/s5_query.py` validates the question structure and literal source spans. `agri_extract/s5_matcher.py` computes `ANSWER`, `ASK` or `ABSTAIN` using three-valued condition logic, S5 scope, exceptions and supported dependencies. It returns selected rule IDs, source pages and a per-rule trace. `agri_extract/s5_reasoner.py` defines the alternative S5-only language-model reasoning contract and validates its citations. Both methods operate on any valid S5 RuleSet supplied by the caller.

Unsupported chemistry, ambiguous free-form relations and missing requested fields remain explicit limits. A source citation confirms the page exists in S5; a human reviewer must still assess whether the answer text is agronomically supported.

## Local commands

```powershell
python -m agri_extract.s5_matcher_cli freeze --rules path/to/verified_rules.json
python -m agri_extract.s5_matcher_cli messages --rules path/to/verified_rules.json --question "What is recommended?" --out messages.json
python -m agri_extract.s5_matcher_cli decide --rules path/to/verified_rules.json --query query.json --out decision.json
```

The `messages` command prepares input without calling a provider. The `decide` command consumes an existing `S5Query`. The freeze records hashes of the source code, prompt, schema and selected rule file. Do not use labels during request generation or decision making; load them only in a separate evaluator after all predictions are saved.
