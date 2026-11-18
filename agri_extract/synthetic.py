"""Clearly synthetic fixture for no-cost workflow tests; never a real result."""
from __future__ import annotations

import json
from .documents import DocumentArtifact, Element, Page


TEXT = "Synthetic document. For test crop, apply 50 kg/ha urea at planting if soil is moist. Do not apply urea if the field is flooded."


def artifact(parser: str = "P0") -> DocumentArtifact:
    element = Element(element_id="synthetic:p001:e0001", page_number=1, kind="paragraph",
                      original=TEXT, cleaned=TEXT)
    return DocumentArtifact(document_id="synthetic", file_name="SYNTHETIC_NOT_REAL.pdf",
        file_hash="synthetic-fixture", parser=parser, version="1",
        pages=[Page(number=1, raw_text=TEXT, structured_text=TEXT, prepared_text=TEXT,
                    ocr_candidate=False, ocr_used=False, elements=[element])],
        conversion={"tool": "synthetic"}, warnings=["SYNTHETIC FIXTURE; not agricultural evidence"])


def rule(schema: str, rule_id: str = "synthetic-r1") -> dict:
    base = {"rule_id": rule_id, "document_id": "synthetic", "schema_version": schema,
            "crop": "test crop", "modality": "SHOULD", "product": "urea", "nutrient": None,
            "notes": "Synthetic fixture"}
    if schema == "S1":
        return {**base, "action": "apply urea", "dose": 50, "unit": "kg/ha", "basis": "product",
                "timing": "at planting", "method": None, "context": "test crop",
                "conditions": "soil is moist", "exceptions": "field is flooded"}
    result = {**base, "action": {"verb": "apply", "target": "urea", "method": None},
              "quantity": {"value": 50, "unit": "kg/ha", "basis": "product",
                           "original": "50 kg/ha", "status": "EXPLICIT"},
              "context": {"crop": "test crop"}, "timing": "at planting"}
    moist = {"kind": "PREDICATE", "variable": "soil_moisture", "operator": "EQ", "value": "moist"}
    flooded = {"kind": "PREDICATE", "variable": "field_state", "operator": "EQ", "value": "flooded"}
    if schema == "S2":
        result.update({"conditions": [moist], "exceptions": ["field is flooded"]})
    else:
        result.update({"conditions": moist, "exceptions": [flooded]})
    if schema in {"S4", "S5"}:
        ref = {"document_id": "synthetic", "page_number": 1, "element_id": "synthetic:p001:e0001",
               "source_text": "apply 50 kg/ha urea at planting if soil is moist"}
        result.update({"evidence": [ref], "field_evidence": {"quantity": [ref]}})
    if schema == "S5":
        result["dependencies"] = []
    return result


def responses(strategy: str, schema: str) -> list[str]:
    ruleset = json.dumps({"document_id": "synthetic", "rules": [rule(schema)]})
    detected = json.dumps({"candidates": [{"candidate_id": "c1", "page_number": 1,
        "action": "apply urea", "source_text": "apply 50 kg/ha urea at planting if soil is moist"}]})
    atoms = json.dumps({"atoms": [{"candidate_id": "c1", "atom_id": "a1", "action": "apply urea",
        "source_text": "apply 50 kg/ha urea at planting if soil is moist"}]})
    verification = json.dumps({"agreements": [], "disagreements": [],
                               "supported_rule_ids": ["synthetic-r1"], "review_required": []})
    return {"E1": [ruleset], "E2": [detected, ruleset], "E3": [ruleset, ruleset],
            "E4": [ruleset, ruleset, verification], "E5": [detected, ruleset],
            "E6": [detected, ruleset], "E7": [detected, atoms, ruleset],
            "E8": [ruleset, ruleset]}[strategy]
