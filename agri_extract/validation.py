"""Cumulative V0–V5; every state is returned without overwriting raw output."""
from __future__ import annotations

import json
import re
from copy import deepcopy

import pint
import jsonschema
from pydantic import ValidationError

from .documents import DocumentArtifact
from .prepare import DOCUMENT_CONFIG
from .providers import Provider
from .schema import RULES, RuleSet, SourceRef, parse_ruleset, to_canonical


VOCAB_VERSION = "v1"
CROPS = {alias.casefold(): entry.get("crop", document_id)
         for document_id, entry in DOCUMENT_CONFIG.items()
         for alias in [entry.get("crop", document_id), *entry.get("crop_aliases", [])]}
PRODUCTS = {"urea": "urea"}
NUTRIENTS = {"nitrogen": "N", "phosphorus": "P", "potassium": "K", "sulfur": "S", "sulphur": "S"}
UNITS = {"kg/ha": "kilogram / hectare", "g/kg": "gram / kilogram", "t/ha": "tonne / hectare"}
DIMENSIONAL_UNITS = {"kg/ha": "kilogram/hectare", "g/m2": "gram/meter**2", "t/ha": "tonne/hectare"}
ureg = pint.UnitRegistry()


def _normalize(ruleset: RuleSet) -> tuple[dict, list[dict]]:
    result = deepcopy(ruleset.model_dump(mode="json"))
    changes = []
    for rule in result["rules"]:
        for field, dictionary in (("crop", CROPS), ("product", PRODUCTS), ("nutrient", NUTRIENTS)):
            value = rule.get(field)
            if isinstance(value, str) and value.lower() in dictionary:
                new = dictionary[value.lower()]
                if new != value:
                    changes.append({"rule_id": rule["rule_id"], "field": field, "original": value, "normalized": new})
                    rule[field] = new
        q = rule.get("quantity")
        if q and q.get("unit") in UNITS:
            # Parse dimension only. No composition-dependent fertilizer conversion.
            try:
                ureg.parse_expression(UNITS[q["unit"]])
                changes.append({"rule_id": rule["rule_id"], "field": "quantity.unit",
                                "original": q["unit"], "normalized": UNITS[q["unit"]],
                                "applied": False, "reason": "dimensional spelling only; original preserved"})
            except (pint.errors.UndefinedUnitError, ValueError):
                pass
        if q and q.get("unit") in DIMENSIONAL_UNITS and any(q.get(k) is not None for k in ("value", "lower", "upper")):
            converted = {}
            for key in ("value", "lower", "upper"):
                if q.get(key) is not None:
                    converted[key] = (q[key] * ureg(DIMENSIONAL_UNITS[q["unit"]])).to("kilogram/hectare").magnitude
            changes.append({"rule_id": rule["rule_id"], "field": "quantity",
                "original": {k: q.get(k) for k in ("value", "lower", "upper", "unit", "basis")},
                "normalized": {**converted, "unit": "kg/ha", "basis": q.get("basis")},
                "applied": False, "reason": "Pint dimensional conversion; source quantity preserved"})
    return result, changes


def _evidence(rule: dict, artifact: DocumentArtifact) -> list[dict]:
    refs = [("evidence", ref) for ref in rule.get("evidence", [])]
    for field, values in rule.get("field_evidence", {}).items():
        refs.extend((field, ref) for ref in values)
    checks = []
    compact = lambda s: re.sub(r"\s+", " ", s).strip().casefold()
    for field, ref in refs:
        page = ref.get("page_number")
        detail = {"rule_id": rule["rule_id"], "field": field,
                  "page_number": page, "element_id": ref.get("element_id"),
                  "source_text": ref.get("source_text"), "origin": "provenance"}
        if ref.get("document_id") != artifact.document_id or not isinstance(page, int) or page < 1 or page > len(artifact.pages):
            checks.append({**detail, "status": "CONTRADICTED", "reason": "invalid document or page"})
            continue
        item = artifact.pages[page - 1]
        quote = ref.get("source_text", "").strip()
        if not quote or compact(quote) not in compact(item.raw_text + "\n" + item.structured_text):
            status, reason = "INSUFFICIENT_EVIDENCE", "quote absent from cited page"
        else:
            status, reason = "NOT_CHECKED", "quote located; semantic support not yet verified"
        element_id = ref.get("element_id")
        if element_id:
            element = next((e for e in item.elements if e.element_id == element_id), None)
            if element is None:
                status, reason = "INSUFFICIENT_EVIDENCE", "element ID absent from cited page"
            elif compact(quote) not in compact(element.original + "\n" + element.cleaned):
                status, reason = "INSUFFICIENT_EVIDENCE", "quote absent from cited element"
            elif ref.get("table_id") and ref["table_id"] != element.table_id:
                status, reason = "INSUFFICIENT_EVIDENCE", "table ID does not match cited element"
        checks.append({**detail, "status": status, "reason": reason})
    if not refs:
        checks.append({"rule_id": rule["rule_id"], "field": "evidence", "status": "NOT_CHECKED",
                       "reason": "schema has no evidence references"})
    return checks


def _coverage(artifact: DocumentArtifact, ruleset: RuleSet) -> dict:
    # Deliberately broad independent signals; never treated as gold labels.
    pattern = re.compile(r"\b(fertili[sz]er|urea|\bDAP\b|\bNPS\b|manure|compost|nitrogen|phosphorus|potassium)\b", re.I)
    candidates = []
    cited = {r.rule_id: list(getattr(r, "evidence", [])) for r in ruleset.rules}
    for page in artifact.pages:
        for element in page.elements:
            if pattern.search(element.cleaned):
                linked = [rule_id for rule_id, refs in cited.items() if any(
                    ref.page_number == page.number and
                    ((ref.element_id and ref.element_id == element.element_id) or
                     (not ref.element_id and " ".join(ref.source_text.split()).casefold() in
                      " ".join(element.original.split()).casefold())) for ref in refs)]
                candidates.append({"element_id": element.element_id, "page_number": page.number,
                                   "kind": element.kind, "table_id": element.table_id,
                                   "text": element.cleaned[:500],
                                   "text_truncated": len(element.cleaned) > 500,
                                   "cited_rule_ids": linked,
                                   "status": "CITED_NOT_SEMANTICALLY_VERIFIED" if linked else "UNLINKED_CANDIDATE"})
    covered_pages = {c["page_number"] for c in candidates if c["cited_rule_ids"]}
    canonical = to_canonical(ruleset)
    keys = [(r.action.casefold(), (r.product or "").casefold(),
             r.quantity.value if r.quantity else None, r.timing) for r in canonical.rules]
    return {"candidate_passages": candidates,
            "pages_with_candidates_without_cited_rules": sorted({c["page_number"] for c in candidates} - covered_pages),
            "potentially_unmatched_passages": [c for c in candidates if not c["cited_rule_ids"]],
            "possible_duplicate_rule_count": len(keys) - len(set(keys)),
            "interpretation": "lexical candidate signals only; missing citations are not proof of omission or a gold label"}


def _verifier_call(verifier: Provider, stage: str, payload: dict, report: dict,
                   prompt_version: str, max_output_tokens: int) -> dict:
    from .extract import prompt, parse_json
    response = verifier.complete(prompt(stage, prompt_version),
                                 json.dumps(payload, ensure_ascii=False),
                                 max_output_tokens=max_output_tokens)
    report["calls"].append({"provider": response.provider, "model": response.model,
        "input_tokens": response.input_tokens, "output_tokens": response.output_tokens,
        "latency_seconds": response.latency_seconds, "stage": stage,
        "raw_response": response.text, "stop_reason": response.stop_reason,
        "max_output_tokens": max_output_tokens})
    if response.stop_reason not in {"end_turn", "stop", "stop_sequence"}:
        raise RuntimeError(f"INCOMPLETE_VERIFIER_OUTPUT: stop_reason={response.stop_reason}")
    parsed = parse_json(response.text)
    if not isinstance(parsed, dict):
        raise ValueError("verifier response must be a JSON object")
    report["calls"][-1]["parsed_response"] = parsed
    return parsed


def validate(raw: dict, schema: str, artifact: DocumentArtifact, level: str,
             verifier: Provider | None = None, max_repairs: int = 2,
             prompt_version: str = "v1", max_output_tokens: int = 8192) -> dict:
    if level not in {f"V{i}" for i in range(6)}:
        raise ValueError(level)
    if max_repairs < 0 or max_output_tokens <= 0:
        raise ValueError("max_repairs must be nonnegative and max_output_tokens positive")
    report = {"level": level, "states": {"V0": deepcopy(raw)}, "errors": [],
              "calls": [], "status": "RAW_ONLY"}
    if level == "V0":
        try:
            parse_ruleset(schema, raw)
            report["raw_schema_valid"] = True
        except (ValidationError, KeyError, TypeError, ValueError):
            report["raw_schema_valid"] = False
        return report
    try:
        for item in raw["rules"]:
            jsonschema.validate(item, RULES[schema].model_json_schema())
        ruleset = parse_ruleset(schema, raw)
    except (ValidationError, jsonschema.ValidationError, KeyError, TypeError, ValueError) as exc:
        report["errors"].append(str(exc))
        report["status"] = "SCHEMA_INVALID"
        return report
    report["states"]["V1"] = ruleset.model_dump(mode="json")
    report["status"] = "SCHEMA_VALID"
    if level == "V1":
        return report
    normalized, changes = _normalize(ruleset)
    report["states"]["V2"] = {"ruleset": normalized, "changes": changes,
                                "vocabulary_version": VOCAB_VERSION}
    if level == "V2":
        return report
    checks = [check for rule in report["states"]["V1"]["rules"] for check in _evidence(rule, artifact)]
    semantic = None
    if verifier is not None:
        try:
            parsed = _verifier_call(verifier, "semantic_verification",
                {"document": artifact.llm_text(), "ruleset": report["states"]["V1"],
                 "provenance_checks": checks}, report, prompt_version, max_output_tokens)
            semantic = {"parsed": parsed, "provider": verifier.name, "model": verifier.model}
        except (RuntimeError, ValueError, json.JSONDecodeError) as exc:
            report["status"] = "VERIFICATION_FAILED"
            report["errors"].append(f"{type(exc).__name__}: {exc}")
    report["states"]["V3"] = {"checks": checks, "semantic": semantic,
        "semantic_status": "NOT_CHECKED" if verifier is None else
            "VERIFICATION_FAILED" if report["status"] == "VERIFICATION_FAILED" else
            "VERIFIER_RETURNED_UNADJUDICATED"}
    if report["status"] == "VERIFICATION_FAILED":
        return report
    if level == "V3":
        return report
    coverage = _coverage(artifact, ruleset)
    report["states"]["V4"] = coverage
    if level == "V4":
        return report
    issues = [c for c in checks if c["status"] in {"CONTRADICTED", "INSUFFICIENT_EVIDENCE"}]
    if semantic and isinstance(semantic.get("parsed"), dict):
        issues.extend({**c, "origin": "semantic"} for c in semantic["parsed"].get("checks", [])
                      if c.get("status") in {"CONTRADICTED", "INSUFFICIENT_EVIDENCE", "AMBIGUOUS"})
    if verifier is None:
        report["states"]["V5"] = {"status": "NOT_CHECKED_NO_VERIFIER", "issues": issues,
                                    "repairs": [], "attempts": 0}
        return report
    attempts = []
    current = deepcopy(report["states"]["V2"]["ruleset"])
    applied = []
    for _ in range(max_repairs):
        if not issues:
            break
        try:
            proposals = _verifier_call(verifier, "targeted_repair",
                {"issues": issues, "ruleset": current,
                 "document": artifact.llm_text()}, report, prompt_version, max_output_tokens)
        except (RuntimeError, ValueError, json.JSONDecodeError) as exc:
            attempts.append({"error": f"{type(exc).__name__}: {exc}"})
            report["status"] = "VERIFICATION_FAILED"
            break
        accepted = []
        for proposal in proposals.get("repairs", []):
            target = next((r for r in current["rules"] if r["rule_id"] == proposal.get("rule_id")), None)
            field = proposal.get("field")
            quote = proposal.get("source_text", "")
            if target is None or field not in {"product", "nutrient", "timing", "modality",
                                                "quantity", "conditions", "exceptions", "evidence"}:
                continue
            if not any(i.get("rule_id") == target["rule_id"] and i.get("field") == field and
                       (i.get("origin") == "semantic" or field == "evidence") for i in issues):
                continue
            if target.get(field) != proposal.get("old"):
                continue
            new = proposal.get("new")
            if field == "evidence" and isinstance(new, list):
                cited_pages = {ref.get("page_number") for ref in new}
            else:
                cited_pages = {ref.get("page_number") for ref in
                    [*target.get("evidence", []), *target.get("field_evidence", {}).get(field, [])]}
            if not any(quote and quote.casefold() in p.raw_text.casefold()
                       for p in artifact.pages if not cited_pages or p.number in cited_pages):
                continue
            if isinstance(new, (str, int, float)) and str(new).casefold() not in quote.casefold():
                continue
            if isinstance(new, dict):
                numeric_values = [str(v) for key, v in new.items()
                                  if key in {"value", "lower", "upper"} and v is not None]
                if field == "quantity" and (not numeric_values or
                                            not all(value in quote for value in numeric_values)):
                    continue
            if field in {"conditions", "exceptions"}:
                def literals(node):
                    if isinstance(node, dict):
                        return [str(node[k]) for k in ("value", "lower", "upper")
                                if node.get(k) is not None] + [v for child in node.get("children", [])
                                for v in literals(child)]
                    if isinstance(node, list):
                        return [v for child in node for v in literals(child)]
                    return []
                values = literals(new)
                if not values or not all(value.casefold() in quote.casefold() for value in values):
                    continue
            if field == "evidence":
                if not isinstance(new, list) or not new:
                    continue
                if not any(quote.casefold() in str(ref.get("source_text", "")).casefold()
                           for ref in new):
                    continue
                evidence_checks = _evidence({"rule_id": target["rule_id"], "evidence": new}, artifact)
                if any(check["status"] != "NOT_CHECKED" for check in evidence_checks):
                    continue
            trial = deepcopy(current)
            next(t for t in trial["rules"] if t["rule_id"] == target["rule_id"])[field] = new
            try:
                parse_ruleset(schema, trial)
            except (ValidationError, ValueError, KeyError, TypeError):
                continue
            current = trial
            accepted.append(proposal)
        applied.extend(accepted)
        attempts.append({"proposals": proposals, "accepted": accepted})
        if accepted:
            break
    report["states"]["V5"] = {"status": "VERIFICATION_FAILED" if report["status"] == "VERIFICATION_FAILED" else
                                "REVIEW_REQUIRED" if issues else "NO_FLAGGED_ISSUES",
                                "issues": issues, "attempts": attempts,
                                "repairs_applied": applied, "ruleset_after": current}
    return report
