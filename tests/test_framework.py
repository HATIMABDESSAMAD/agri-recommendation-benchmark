import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from agri_extract.alignment import align, compare
from agri_extract.documents import _blocks, _elements, convert
from agri_extract.evaluation import AnnotationDisagreement, GoldAgriculturalRuleSet, check_gold, evaluate, gold_template
from agri_extract.experiments import compare_campaigns, execute, matrix, plan, repeatability, report, simulate_matrix
from agri_extract.extract import parse_json, run_strategy
from agri_extract.providers import LiveProvider, MockProvider
from agri_extract.schema import Logic, Predicate, S3Rule, SourceRef, parse_ruleset, to_canonical
from agri_extract.synthetic import artifact, responses, rule
from agri_extract.transport import flatten, restore, restore_ruleset, transport_schema
from agri_extract.validation import validate


@pytest.mark.parametrize("schema", ["S1", "S2", "S3", "S4", "S5"])
def test_all_schemas_and_canonical(schema):
    rs = parse_ruleset(schema, {"document_id": "synthetic", "rules": [rule(schema)]})
    canonical = to_canonical(rs)
    assert canonical.rules[0].quantity.value == 50
    assert canonical.source_schema == schema


def test_ast_operators_parentheses_and_invalid_nodes():
    a = Predicate(variable="soil", operator="EQ", value="moist")
    b = Predicate(variable="season", operator="EQ", value="wet")
    c = Predicate(variable="soil", operator="EQ", value="flooded")
    node = Logic(kind="ALL", children=[a, Logic(kind="ANY", children=[b, c])])
    assert node.children[1].kind == "ANY"
    with pytest.raises(ValidationError):
        Logic(kind="NOT", children=[a, b])
    with pytest.raises(ValidationError):
        Predicate(variable="rainfall", operator="BETWEEN", lower=10, upper=20)
    with pytest.raises(ValidationError):
        Logic(kind="ANY", children=[])
    deep = rule("S3")
    deep["conditions"] = {"kind": "NOT", "children": [{"kind": "NOT", "children": [
        {"kind": "PREDICATE", "variable": "soil", "operator": "EQ", "value": "moist"}]}]}
    with pytest.raises(ValidationError):
        parse_ruleset("S3", {"document_id": "synthetic", "rules": [deep]}, max_ast_depth=2)
    parse_ruleset("S3", {"document_id": "synthetic", "rules": [deep]}, max_ast_depth=3)


def test_dependencies_and_cross_document_integrity():
    a, b = rule("S5", "a"), rule("S5", "b")
    a["dependencies"] = [{"kind": "PRECEDES", "target_rule_id": "b"}]
    assert len(parse_ruleset("S5", {"document_id": "synthetic", "rules": [a, b]}).rules) == 2
    b["dependencies"] = [{"kind": "PRECEDES", "target_rule_id": "a"}]
    with pytest.raises(ValidationError):
        parse_ruleset("S5", {"document_id": "synthetic", "rules": [a, b]})
    b["document_id"] = "other"
    with pytest.raises(ValidationError):
        parse_ruleset("S5", {"document_id": "synthetic", "rules": [a, b]})


def test_absent_quantity_is_not_zero():
    r = rule("S1")
    r["dose"] = None
    assert to_canonical(parse_ruleset("S1", {"document_id": "synthetic", "rules": [r]})).rules[0].quantity.value is None


def test_compound_explicit_quantity_keeps_unparsed_source_text():
    from agri_extract.schema import Quantity, Presence
    compound = Quantity(status=Presence.EXPLICIT,
        original="92 kg N + 40 kg P + 30 kg S per hectare")
    assert compound.value is None and compound.lower is None
    qualitative = Quantity(status=Presence.EXPLICIT, original="the remaining half")
    assert qualitative.value is None and qualitative.original == "the remaining half"
    item = rule("S3")
    item["quantity"] = {"status": "EXPLICIT", "original": "the remaining half"}
    canonical = to_canonical(parse_ruleset("S3", {"document_id": "synthetic", "rules": [item]}))
    assert "unparsed_explicit_quantity" in canonical.rules[0].representation_limits
    with pytest.raises(ValueError, match="original source text"):
        Quantity(status=Presence.EXPLICIT, original=" ")


def test_fenced_json_can_have_trailing_explanation_but_not_second_payload():
    response = '```json\n{"document_id":"synthetic","rules":[]}\n```\nNo rules found.'
    assert parse_json(response)["rules"] == []
    with pytest.raises(json.JSONDecodeError):
        parse_json('```json\n{"rules":[]}\n```\n{"rules":[1]}')


@pytest.mark.parametrize("strategy,count", [("E1",1),("E2",2),("E3",2),("E4",3),
                                         ("E5",2),("E6",2),("E7",3),("E8",2)])
def test_strategy_calls_and_raw_retention(strategy, count):
    provider = MockProvider(responses(strategy, "S1"))
    providers = [provider, provider] if strategy in {"E3", "E4", "E5", "E6"} else [provider]
    output = run_strategy(strategy, artifact(), providers, "S1")
    assert len(output["trace"]) == count
    assert all("raw_response" in call for call in output["trace"])
    assert len(output["outputs"]) == (2 if strategy in {"E3", "E4", "E8"} else 1)


def test_e2_candidate_and_e7_atom_preserved():
    for strategy in ("E2", "E7"):
        output = run_strategy(strategy, artifact(), [MockProvider(responses(strategy, "S3"))], "S3")
        assert output["candidates"]["candidates"][0]["candidate_id"] == "c1"
        if strategy == "E7":
            assert output["atoms"]["atoms"][0]["atom_id"] == "a1"


def test_prompt_version_is_configurable_and_traced(tmp_path, monkeypatch):
    import agri_extract.prepare as sources
    folder = tmp_path / "prompts" / "v2"
    folder.mkdir(parents=True)
    (folder / "joint_extraction.txt").write_text("SYNTHETIC V2 PROMPT", encoding="utf-8")
    monkeypatch.setattr(sources, "ROOT", tmp_path)
    output = run_strategy("E1", artifact(), [MockProvider(responses("E1", "S1"))],
                          "S1", prompt_version="v2")
    assert output["trace"][0]["prompt_version"] == "v2"
    assert output["trace"][0]["system_prompt"] == "SYNTHETIC V2 PROMPT"


def test_validation_levels_and_missing_evidence():
    art = artifact()
    raw = {"document_id": "synthetic", "rules": [rule("S4")]}
    assert validate(raw, "S4", art, "V0")["status"] == "RAW_ONLY"
    assert validate(raw, "S4", art, "V1")["status"] == "SCHEMA_VALID"
    assert "V2" in validate(raw, "S4", art, "V2")["states"]
    v3 = validate(raw, "S4", art, "V3")
    assert v3["states"]["V3"]["checks"][0]["status"] == "NOT_CHECKED"
    assert validate(raw, "S4", art, "V4")["states"]["V4"]["candidate_passages"]
    assert validate(raw, "S4", art, "V5")["states"]["V5"]["status"] == "NOT_CHECKED_NO_VERIFIER"
    raw["rules"][0]["evidence"][0]["source_text"] = "not in document"
    assert validate(raw, "S4", art, "V3")["states"]["V3"]["checks"][0]["status"] == "INSUFFICIENT_EVIDENCE"
    dimensional = {"document_id": "synthetic", "rules": [rule("S2")]}
    dimensional["rules"][0]["quantity"].update(value=10, unit="g/m2", original="10 g/m2")
    converted = validate(dimensional, "S2", art, "V2")["states"]["V2"]["changes"]
    assert any(x.get("normalized", {}).get("value") == 100 for x in converted if isinstance(x.get("normalized"), dict))


def test_v0_is_measured_without_publishing_validated_output(tmp_path):
    config = {"strategy": "E1", "schema": "S3", "validation": "V0", "parser": "P0"}
    path = execute(config, artifact(), [MockProvider(responses("E1", "S3"))],
                   "v0", output_root=tmp_path)
    manifest = json.loads((path / "manifest.json").read_text())
    assert manifest["status"] == "SIMULATED"
    assert json.loads((path / "validation_1.json").read_text())["raw_schema_valid"] is True
    assert (path / "raw.json").exists()
    assert not (path / "validated.json").exists()
    assert not (path / "final.json").exists()


def test_v2_final_is_separate_from_v1_and_raw(tmp_path):
    item = rule("S3")
    item["product"] = "UREA"
    item["nutrient"] = "Nitrogen"
    config = {"strategy": "E1", "schema": "S3", "validation": "V2", "parser": "P0"}
    path = execute(config, artifact(), [MockProvider([json.dumps({"document_id": "synthetic", "rules": [item]})])],
                   "v2", output_root=tmp_path)
    assert json.loads((path / "raw.json").read_text())["rules"][0]["product"] == "UREA"
    assert json.loads((path / "validated.json").read_text())["rules"][0]["product"] == "UREA"
    assert json.loads((path / "final.json").read_text())["rules"][0]["product"] == "urea"
    assert json.loads((path / "final.json").read_text())["rules"][0]["nutrient"] == "N"


def test_v5_repair_is_separate_and_evaluated_against_gold(tmp_path):
    item = rule("S4")
    item["product"] = "unknown"
    item["evidence"][0]["source_text"] = "missing quote"
    raw = json.dumps({"document_id": "synthetic", "rules": [item]})
    semantic = json.dumps({"checks": [{"rule_id": "synthetic-r1", "field": "product",
                                    "status": "CONTRADICTED", "reason": "wrong product"}]})
    repair = json.dumps({"repairs": [{"rule_id": "synthetic-r1", "field": "product",
                        "old": "unknown", "new": "urea", "source_text": "urea"}]})
    canonical = to_canonical(parse_ruleset("S4", {"document_id": "synthetic", "rules": [rule("S4")]}))
    gold = GoldAgriculturalRuleSet(document_id="synthetic", source_hash="synthetic-fixture",
        page_count=1, split="development", status="ADJUDICATED", annotators=["fixture"],
        single_annotator_limitation="Synthetic fixture", rules=canonical.rules, reviewed_pages=[1])
    config = {"strategy": "E1", "schema": "S4", "validation": "V5", "parser": "P0"}
    path = execute(config, artifact(), [MockProvider([raw])], "repair", output_root=tmp_path,
                   verifier=MockProvider([semantic, repair]), gold=gold)
    assert json.loads((path / "raw.json").read_text())["rules"][0]["product"] == "unknown"
    assert json.loads((path / "validated.json").read_text())["rules"][0]["product"] == "unknown"
    assert json.loads((path / "repaired.json").read_text())["rules"][0]["product"] == "urea"
    assert json.loads((path / "final.json").read_text())["rules"][0]["product"] == "urea"
    assert json.loads((path / "evaluation_raw.json").read_text())["metrics"]["recommendation"]["f1"] is None
    assert json.loads((path / "evaluation.json").read_text())["metrics"]["recommendation"]["f1"] == 1.0
    assert (path / "evidence_checks.json").exists() and (path / "coverage.json").exists()
    assert (path / "repairs.json").exists()


def test_v5_targeted_repair_retains_before_and_after():
    raw = {"document_id": "synthetic", "rules": [rule("S4")]}
    raw["rules"][0]["product"] = "unknown"
    raw["rules"][0]["evidence"][0]["source_text"] = "missing quote"
    semantic = json.dumps({"checks": [{"rule_id": "synthetic-r1", "field": "product",
                                    "status": "CONTRADICTED", "reason": "wrong product"}]})
    repair = json.dumps({"repairs": [{"rule_id": "synthetic-r1", "field": "product",
                      "old": "unknown", "new": "urea", "source_text": "urea"}]})
    verifier = MockProvider([semantic, repair])
    result = validate(raw, "S4", artifact(), "V5", verifier=verifier, max_output_tokens=512)
    assert result["states"]["V1"]["rules"][0]["product"] == "unknown"
    assert result["states"]["V5"]["ruleset_after"]["rules"][0]["product"] == "urea"
    assert len(result["states"]["V5"]["repairs_applied"]) == 1
    assert all(call["max_output_tokens"] == 512 for call in verifier.calls)


def test_verifier_failure_keeps_extraction_and_validation(tmp_path):
    config = {"strategy": "E1", "schema": "S4", "validation": "V5", "parser": "P0"}
    path = execute(config, artifact(), [MockProvider(responses("E1", "S4"))],
                   "failed_verifier", output_root=tmp_path,
                   verifier=MockProvider(["{invalid"]))
    assert json.loads((path / "manifest.json").read_text())["status"] == "VERIFICATION_FAILED"
    assert (path / "raw.json").exists() and (path / "validated.json").exists()
    assert not (path / "repaired.json").exists()
    assert json.loads((path / "validation_1.json").read_text())["calls"][0]["raw_response"] == "{invalid"


def test_provenance_quote_must_match_cited_element():
    art = artifact()
    art.pages[0].elements.append(type(art.pages[0].elements[0])(
        element_id="synthetic:p001:e0002", page_number=1, kind="paragraph",
        original="unrelated", cleaned="unrelated"))
    raw = {"document_id": "synthetic", "rules": [rule("S4")]}
    raw["rules"][0]["evidence"][0]["element_id"] = "synthetic:p001:e0002"
    checks = validate(raw, "S4", art, "V3")["states"]["V3"]["checks"]
    assert checks[0]["status"] == "INSUFFICIENT_EVIDENCE"
    assert checks[0]["reason"] == "quote absent from cited element"


def test_coverage_is_an_independent_candidate_signal():
    art = artifact()
    supported = validate({"document_id": "synthetic", "rules": [rule("S4")]},
                         "S4", art, "V4")["states"]["V4"]
    assert supported["candidate_passages"][0]["status"] == "CITED_NOT_SEMANTICALLY_VERIFIED"
    assert not supported["potentially_unmatched_passages"]
    unlinked = validate({"document_id": "synthetic", "rules": []},
                        "S4", art, "V4")["states"]["V4"]
    assert unlinked["candidate_passages"][0]["status"] == "UNLINKED_CANDIDATE"
    assert unlinked["pages_with_candidates_without_cited_rules"] == [1]


def test_targeted_repair_rejects_unsupported_numeric_change():
    raw = {"document_id": "synthetic", "rules": [rule("S4")]}
    raw["rules"][0]["quantity"]["value"] = 75
    raw["rules"][0]["evidence"][0]["source_text"] = "missing quote"
    semantic = json.dumps({"checks": [{"rule_id": "synthetic-r1", "field": "quantity",
                                    "status": "CONTRADICTED", "reason": "dose differs"}]})
    old_quantity = parse_ruleset("S4", raw).model_dump(mode="json")["rules"][0]["quantity"]
    unsupported = json.dumps({"repairs": [{"rule_id": "synthetic-r1", "field": "quantity",
        "old": old_quantity,
        "new": {**old_quantity, "value": 900},
        "source_text": "apply 50 kg/ha urea at planting"}]})
    verifier = MockProvider([semantic, unsupported, unsupported])
    result = validate(raw, "S4", artifact(), "V5", verifier=verifier)
    assert not result["states"]["V5"]["repairs_applied"]
    assert result["states"]["V5"]["ruleset_after"]["rules"][0]["quantity"]["value"] == 75
    assert len(result["states"]["V5"]["attempts"]) == 2


def test_v5_repairs_citation_only_when_it_matches_original_page():
    raw = {"document_id": "synthetic", "rules": [rule("S4")]}
    raw["rules"][0]["evidence"][0]["source_text"] = "missing quote"
    old = parse_ruleset("S4", raw).model_dump(mode="json")["rules"][0]["evidence"]
    new = [SourceRef.model_validate(rule("S4")["evidence"][0]).model_dump(mode="json")]
    repair = json.dumps({"repairs": [{"rule_id": "synthetic-r1", "field": "evidence",
        "old": old, "new": new, "source_text": new[0]["source_text"]}]})
    result = validate(raw, "S4", artifact(), "V5",
                      verifier=MockProvider([json.dumps({"checks": []}), repair]))
    assert result["states"]["V5"]["ruleset_after"]["rules"][0]["evidence"] == new
    assert result["states"]["V1"]["rules"][0]["evidence"] == old


def test_v5_can_repair_semantically_flagged_condition_with_literal_support():
    raw = {"document_id": "synthetic", "rules": [rule("S4")]}
    raw["rules"][0]["conditions"]["value"] = "dry"
    old = parse_ruleset("S4", raw).model_dump(mode="json")["rules"][0]["conditions"]
    new = {**old, "value": "moist"}
    semantic = json.dumps({"checks": [{"rule_id": "synthetic-r1", "field": "conditions",
        "status": "CONTRADICTED", "reason": "source says moist"}]})
    repair = json.dumps({"repairs": [{"rule_id": "synthetic-r1", "field": "conditions",
        "old": old, "new": new, "source_text": "soil is moist"}]})
    result = validate(raw, "S4", artifact(), "V5", verifier=MockProvider([semantic, repair]))
    assert result["states"]["V5"]["ruleset_after"]["rules"][0]["conditions"]["value"] == "moist"
    assert result["states"]["V1"]["rules"][0]["conditions"]["value"] == "dry"


def test_invalid_json_and_schema_error_recorded():
    mock = MockProvider(["{invalid"])
    with pytest.raises(json.JSONDecodeError):
        run_strategy("E1", artifact(), [mock], "S1")
    invalid = {"document_id": "synthetic", "rules": [{"rule_id": "x"}]}
    assert validate(invalid, "S1", artifact(), "V1")["status"] == "SCHEMA_INVALID"


def test_alignment_is_one_to_one_and_reports_conflict():
    a = to_canonical(parse_ruleset("S1", {"document_id": "synthetic", "rules": [rule("S1", "a"), rule("S1", "b")]})).rules
    b = to_canonical(parse_ruleset("S1", {"document_id": "synthetic", "rules": [rule("S1", "c")]})).rules
    b[0].quantity.value = 60
    result = compare(a, b)
    assert len(result["matches"]) == 1
    assert len(result["left_only"]) == 1
    assert "quantity" in result["conflicts"][0]["fields"]


def test_gold_guard_and_synthetic_metrics():
    rs = to_canonical(parse_ruleset("S1", {"document_id": "synthetic", "rules": [rule("S1")]}))
    gold = gold_template("synthetic", "synthetic-fixture", 1)
    assert evaluate(rs, gold, "synthetic-fixture")["status"] == "NOT_EVALUATED"
    approved = GoldAgriculturalRuleSet(document_id="synthetic", source_hash="synthetic-fixture",
        page_count=1, split="development", status="ADJUDICATED", annotators=["test fixture"],
        single_annotator_limitation="Synthetic fixture", rules=rs.rules, reviewed_pages=[1])
    approved.rules[0].evidence = [SourceRef.model_validate(rule("S4")["evidence"][0])]
    result = evaluate(rs, approved, "synthetic-fixture")
    assert result["metrics"]["recommendation"]["f1"] == 1.0
    assert result["metrics"]["cost"] is None
    assert result["by_difficulty"]["table"]["status"] == "NOT_EVALUATED"
    approved.rules[0].difficulty_tags = ["simple", "quantity"]
    assert evaluate(rs, approved, "synthetic-fixture")["by_difficulty"]["quantity"]["recall"] == 1.0
    rs4 = to_canonical(parse_ruleset("S4", {"document_id": "synthetic", "rules": [rule("S4")]}))
    approved4 = GoldAgriculturalRuleSet(document_id="synthetic", source_hash="synthetic-fixture",
        page_count=1, split="development", status="ADJUDICATED", annotators=["test fixture"],
        single_annotator_limitation="Synthetic fixture", rules=rs4.rules, reviewed_pages=[1])
    m4 = evaluate(rs4, approved4, "synthetic-fixture")["metrics"]
    assert m4["evidence_support_rate"] == 1.0
    assert m4["field_grounding_accuracy"] == 1.0
    assert m4["operator_accuracy"] == 1.0


def test_gold_quality_gate_requires_source_and_resolved_disagreements():
    canonical = to_canonical(parse_ruleset("S4", {"document_id": "synthetic", "rules": [rule("S4")]}))
    gold = GoldAgriculturalRuleSet(document_id="synthetic", source_hash="synthetic-fixture",
        page_count=1, split="test", status="ADJUDICATED", annotators=["annotator A"],
        single_annotator_limitation="Only one annotator was available",
        rules=canonical.rules, reviewed_pages=[1],
        annotation_disagreements=[AnnotationDisagreement(field="quantity", positions={"A": "50", "B": "75"})])
    assert check_gold(gold, artifact())["status"] == "NOT_READY"
    assert evaluate(canonical, gold, "synthetic-fixture")["status"] == "NOT_EVALUATED"
    gold.annotation_disagreements[0].status = "RESOLVED"
    gold.annotation_disagreements[0].resolution = "Original PDF says 50 kg/ha"
    assert check_gold(gold, artifact())["status"] == "READY"
    gold.rules[0].evidence[0].source_text = "not in source"
    assert check_gold(gold, artifact())["status"] == "NOT_READY"


def test_execute_rejects_unreviewed_gold_before_creating_run(tmp_path):
    source = artifact()
    unreviewed = gold_template(source.document_id, source.file_hash, len(source.pages))
    config = {"strategy": "E1", "schema": "S3", "validation": "V1", "parser": "P0"}
    with pytest.raises(ValueError, match="gold standard is not ready"):
        execute(config, source, [MockProvider(responses("E1", "S3"))],
                "unreviewed", output_root=tmp_path, gold=unreviewed)
    assert not (tmp_path / "unreviewed").exists()


def test_matrix_simulation_and_resume(tmp_path):
    assert len(matrix()) == 18
    assert all(r["status"] == "BLOCKED_MISSING_CONFIGURATION" for r in plan())
    paths = simulate_matrix("demo", output_root=tmp_path)
    assert len(paths) == 18
    assert all(json.loads((p/"manifest.json").read_text())["status"] == "SIMULATED" for p in paths)
    assert paths == simulate_matrix("demo", output_root=tmp_path, resume=True)
    out_json, out_csv = report("demo", output_root=tmp_path)
    assert len(json.loads(out_json.read_text())) == 18
    assert out_csv.exists()
    assert (tmp_path / "demo" / "repeatability.json").exists()
    assert all(row["f1_difference"] is None for row in compare_campaigns("demo", "demo", output_root=tmp_path))


def test_configured_parser_must_match_artifact(tmp_path):
    config = {"strategy": "E1", "schema": "S3", "validation": "V1", "parser": "P4"}
    with pytest.raises(ValueError, match="parser"):
        execute(config, artifact(), [MockProvider(responses("E1", "S3"))],
                "mismatch", output_root=tmp_path)


def test_repeatability_interval_requires_multiple_real_scores():
    base = {"document_id": "synthetic", "model": "mock", "strategy": "E1",
            "schema": "S4", "parser": "P0", "validation": "V1"}
    single = repeatability([{**base, "recommendation_f1": None}])
    assert single[0]["bootstrap_95_ci_recommendation_f1"] is None
    repeated = repeatability([{**base, "recommendation_f1": 0.5},
                              {**base, "recommendation_f1": 1.0}], bootstrap_samples=100)
    assert repeated[0]["mean_recommendation_f1"] == 0.75
    assert repeated[0]["bootstrap_95_ci_recommendation_f1"] == [0.5, 1.0]


def test_stage_metrics_are_saved_only_with_adjudicated_gold(tmp_path):
    canonical = to_canonical(parse_ruleset("S4", {"document_id": "synthetic", "rules": [rule("S4")]}))
    gold = GoldAgriculturalRuleSet(document_id="synthetic", source_hash="synthetic-fixture",
        page_count=1, split="development", status="ADJUDICATED", annotators=["fixture"],
        single_annotator_limitation="Synthetic fixture", rules=canonical.rules, reviewed_pages=[1])
    config = {"run": "T01", "strategy": "E1", "schema": "S4", "validation": "V5", "parser": "P0"}
    path = execute(config, artifact(), [MockProvider(responses("E1", "S4"))], "test",
                   output_root=tmp_path, gold=gold)
    stages = json.loads((path / "evaluation_by_stage.json").read_text())
    assert set(stages) == {"V0", "V1", "V2", "V3", "V4", "V5"}
    assert all(x["status"] == "EVALUATED" for x in stages.values())


def test_two_extractor_candidate_sets_are_evaluated_without_claiming_consensus(tmp_path):
    canonical = to_canonical(parse_ruleset("S4", {"document_id": "synthetic", "rules": [rule("S4")]}))
    gold = GoldAgriculturalRuleSet(document_id="synthetic", source_hash="synthetic-fixture",
        page_count=1, split="development", status="ADJUDICATED", annotators=["fixture"],
        single_annotator_limitation="Synthetic fixture", rules=canonical.rules, reviewed_pages=[1])
    config = {"strategy": "E4", "schema": "S4", "validation": "V1", "parser": "P0"}
    mock = MockProvider(responses("E4", "S4"))
    path = execute(config, artifact(), [mock, mock], "dual", output_root=tmp_path, gold=gold)
    multi = json.loads((path / "evaluation_multi.json").read_text())
    assert multi["status"] == "CANDIDATE_SET_COMPARISON"
    assert multi["extractor_a"]["metrics"]["recommendation"]["f1"] == 1.0
    assert multi["extractor_b"]["metrics"]["recommendation"]["f1"] == 1.0
    assert multi["matched_intersection"]["metrics"]["recommendation"]["f1"] == 1.0
    assert (path / "consolidation.json").exists()
    assert not (path / "evaluation.json").exists()


def test_live_provider_requires_explicit_mode():
    with pytest.raises(PermissionError):
        LiveProvider("openai", "some-model")
    with pytest.raises(ValueError):
        LiveProvider("deepinfra", "some-model", allow_live=True)
    incomplete = {"deepinfra": {"model": "unknown", "endpoint": "https://example.invalid/v1"}}
    assert plan(incomplete)[2]["status"] == "BLOCKED_MISSING_CONFIGURATION"


def test_plan_separates_model_configuration_from_missing_credential(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    configured = {"anthropic": {"model": "offline-model", "context_window_tokens": 100000}}
    assert plan(configured)[0]["status"] == "BLOCKED_MISSING_CREDENTIAL"


def test_p0_pdf_and_page_provenance(tmp_path):
    pdf = tmp_path / "synthetic.pdf"
    c = canvas.Canvas(str(pdf), pagesize=letter)
    c.drawString(70, 720, "Apply 50 kg/ha urea at planting; apply 120 kg/ha NPS and 30 kg/ha potassium under suitable soil moisture.")
    c.drawString(70, 690, "Note 1: do not apply when flooded.")
    c.save()
    art = convert("synthetic", "P0", path=pdf)
    assert art.pages[0].number == 1
    assert "50 kg/ha" in art.pages[0].raw_text
    assert art.pages[0].elements[0].element_id.startswith("synthetic:p001")


def test_markdown_table_keeps_rows_columns_and_caption_separate():
    text = "Table 1: fertilizer doses\n| Product | Dose |\n|---|---|\n| Urea | 50 kg/ha |"
    elements, _ = _elements("synthetic", 1, text, True)
    assert [e.kind for e in elements] == ["note", "table"]
    assert elements[1].cells[1] == ["Urea", "50 kg/ha"]


@pytest.mark.parametrize("pipeline,table", [("P1",False),("P2",True),("P3",True),("P4",True)])
def test_docling_variants_configured_without_model_download(tmp_path, monkeypatch, pipeline, table):
    pdf = tmp_path / "synthetic.pdf"
    c = canvas.Canvas(str(pdf), pagesize=letter)
    c.drawString(70, 720, "Apply 50 kg/ha urea at planting; apply 120 kg/ha NPS and 30 kg/ha potassium under suitable soil moisture.")
    c.save()
    import docling.document_converter as dc
    captured = {}
    class FakeConverter:
        def __init__(self, format_options):
            options = next(iter(format_options.values())).pipeline_options
            captured.update(table=options.do_table_structure, ocr=options.do_ocr)
        def convert(self, path):
            doc = SimpleNamespace(export_to_markdown=lambda page_no: "| Product | Dose |\n|---|---|\n| Urea | 50 kg/ha |")
            return SimpleNamespace(document=doc)
    monkeypatch.setattr(dc, "DocumentConverter", FakeConverter)
    art = convert("synthetic", pipeline, path=pdf)
    assert captured == {"table": table, "ocr": False}
    assert art.pages[0].raw_text
    assert art.pages[0].structured_text
    assert art.pages[0].warnings  # Docling mock dropped native numeric/text; fallback is explicit


def test_conditional_ocr_only_sparse_page(tmp_path, monkeypatch):
    pdf = tmp_path / "two_pages.pdf"
    c = canvas.Canvas(str(pdf), pagesize=letter)
    c.drawString(70, 720, "This is a long native page with enough text to avoid OCR. Apply 50 kg/ha urea at planting when soil is moist.")
    c.showPage()
    c.drawString(70, 720, "Scan")
    c.save()
    import docling.document_converter as dc
    calls = []
    class FakeConverter:
        def __init__(self, format_options):
            self.ocr = next(iter(format_options.values())).pipeline_options.do_ocr
        def convert(self, path):
            calls.append((self.ocr, str(path)))
            return SimpleNamespace(document=SimpleNamespace(
                export_to_markdown=lambda page_no: "OCR recovered text" if self.ocr else "Normal parsed text"))
    monkeypatch.setattr(dc, "DocumentConverter", FakeConverter)
    art = convert("synthetic", "P3", path=pdf)
    assert len(calls) == 2
    assert calls[0][0] is False and calls[1][0] is True
    assert not art.pages[0].ocr_used and art.pages[1].ocr_used


def test_two_atomic_actions_and_split_application():
    first, second = rule("S1", "first"), rule("S1", "second")
    first["timing"], second["timing"] = "at planting", "at flowering"
    first["dose"], second["dose"] = 50, 25
    rs = parse_ruleset("S1", {"document_id": "synthetic", "rules": [first, second]})
    assert [(r.dose, r.timing) for r in rs.rules] == [(50, "at planting"), (25, "at flowering")]
    with pytest.raises(ValidationError):
        parse_ruleset("S1", {"document_id": "synthetic", "rules": [first, first]})


def test_interval_ambiguity_and_inherited_context():
    r = rule("S3")
    r["quantity"] = {"lower": 50, "upper": 75, "unit": "kg/ha", "basis": "product",
                     "original": "50-75 kg/ha", "status": "AMBIGUOUS"}
    r["context"]["region"] = "inherited from section title"
    parsed = parse_ruleset("S3", {"document_id": "synthetic", "rules": [r]})
    assert parsed.rules[0].quantity.lower == 50
    assert parsed.rules[0].quantity.value is None
    assert parsed.rules[0].context.region == "inherited from section title"


def test_table_cell_and_note_evidence_preserved():
    r = rule("S4")
    ref = r["evidence"][0]
    ref.update(table_id="table-1", row_id="row-2", column_id="dose")
    note = dict(ref, element_id="note-1", source_text="Do not apply when flooded")
    r["field_evidence"]["exceptions"] = [note]
    parsed = parse_ruleset("S4", {"document_id": "synthetic", "rules": [r]})
    assert parsed.rules[0].evidence[0].column_id == "dose"
    assert parsed.rules[0].field_evidence["exceptions"][0].element_id == "note-1"


def test_flat_ast_transport_roundtrip_and_cycle_rejection():
    original = Logic(kind="ALL", children=[Predicate(variable="soil", operator="EQ", value="moist"),
        Logic(kind="NOT", children=[Predicate(variable="field", operator="EQ", value="flooded")])])
    flat = flatten(original)
    assert restore(flat) == original
    bad = {"root": "n1", "nodes": [{"id": "n1", "kind": "NOT", "children": ["n1"]}]}
    with pytest.raises(ValueError):
        restore(bad)
    from agri_extract.extract import _ruleset_schema
    s = transport_schema(_ruleset_schema("S3"))
    assert "conditions_transport" in s["properties"]["rules"]["items"]["properties"]


def test_native_flat_transport_is_restored_before_validation():
    item = rule("S3")
    node = Predicate.model_validate(item.pop("conditions"))
    exceptions = item.pop("exceptions")
    item["conditions_transport"] = flatten(node)
    item["exceptions_transport"] = [flatten(Predicate.model_validate(e)) for e in exceptions]
    mock = MockProvider([json.dumps({"document_id": "synthetic", "rules": [item]})])
    output = run_strategy("E1", artifact(), [mock], "S3", protocol="native")
    assert output["outputs"][0]["rules"][0]["conditions"]["kind"] == "PREDICATE"
    assert parse_ruleset("S3", output["outputs"][0]).rules[0].conditions.variable == "soil_moisture"
