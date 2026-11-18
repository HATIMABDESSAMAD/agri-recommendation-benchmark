"""Synthetic S5 examples shared by query tests."""


def pred(variable, operator, value=None, **kwargs):
    return {"kind": "PREDICATE", "variable": variable, "operator": operator,
            "value": value, **kwargs}


def rule(rid="sample", crop="orchard_pear", target="material alpha", **changes):
    item = {"rule_id": rid, "document_id": "synthetic_manual", "schema_version": "S5",
            "crop": crop, "modality": "SHOULD", "product": target, "nutrient": None,
            "notes": None, "action": {"verb": "apply", "target": target, "method": None},
            "quantity": {"value": 5, "lower": None, "upper": None, "unit": "kg/ha",
                         "basis": "product mass", "original": "5 kg/ha", "status": "EXPLICIT"},
            "context": None, "timing": None, "conditions": None, "exceptions": [],
            "evidence": [{"document_id": "synthetic_manual", "page_number": 2,
                          "source_text": "Apply material alpha at 5 kg/ha."}],
            "field_evidence": {}, "dependencies": []}
    item.update(changes)
    return item


def rules(*items):
    return {"document_id": "synthetic_manual", "schema_version": "S5", "rules": list(items)}
