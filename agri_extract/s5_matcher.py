"""S5Matcher: conservative S5 matching for validated S5Query input.

No source document, benchmark question, model output, or gold label is imported.
Unknown semantics are reported instead of being converted into agronomic policy.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import re
from typing import Any

from .schema import parse_ruleset

VERSION = "S5Matcher-1.0.0"
FIELDS = frozenset({"action", "modality", "product", "nutrient", "quantity", "unit",
                    "basis", "timing", "method", "evidence", "conversion"})
_NUMERIC = {"LT", "LE", "GT", "GE", "BETWEEN"}


class Truth(str, Enum):
    TRUE = "TRUE"
    FALSE = "FALSE"
    UNKNOWN = "UNKNOWN"


def _norm(value: Any) -> str:
    """Lexical normalization only: no unproven synonyms or domain substitutions."""
    return " ".join(re.findall(r"[\w]+", str(value).casefold(), flags=re.UNICODE))


def _same_or_phrase(query: str, source: str) -> bool:
    q, s = _norm(query), _norm(source)
    return bool(q and s and (q == s or f" {q} " in f" {s} "))


def _fact(facts: dict, name: str) -> tuple[bool, Any]:
    record = facts.get(name)
    if isinstance(record, dict) and set(record) <= {"value", "status", "source_text"}:
        if record.get("status", "EXPLICIT") != "EXPLICIT":
            return False, None
        return (record.get("value") is not None), record.get("value")
    return (record is not None), record


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _area(facts: dict) -> float | None:
    present, value = _fact(facts, "area_ha")
    number = _number(value) if present else None
    return number if number is not None and number > 0 else None


def evaluate_ast(node: dict | None, facts: dict) -> tuple[Truth, set[str]]:
    """Strong Kleene logic; missing and ambiguous facts both remain UNKNOWN."""
    if node is None:
        return Truth.TRUE, set()
    kind = node["kind"]
    if kind != "PREDICATE":
        children = [evaluate_ast(c, facts) for c in node["children"]]
        values = [v for v, _ in children]
        missing = set().union(*(m for _, m in children))
        if kind == "NOT":
            return ({Truth.TRUE: Truth.FALSE, Truth.FALSE: Truth.TRUE,
                     Truth.UNKNOWN: Truth.UNKNOWN}[values[0]], missing)
        if kind == "ALL":
            return (Truth.FALSE if Truth.FALSE in values else
                    Truth.UNKNOWN if Truth.UNKNOWN in values else Truth.TRUE, missing)
        if kind == "ANY":
            return (Truth.TRUE if Truth.TRUE in values else
                    Truth.UNKNOWN if Truth.UNKNOWN in values else Truth.FALSE, missing)
        raise ValueError(f"Unknown AST kind: {kind}")
    variable, operator = node["variable"], node["operator"]
    present, actual = _fact(facts, variable)
    if not present:
        return Truth.UNKNOWN, {variable}
    if operator == "EXISTS":
        return Truth.TRUE, set()
    expected = node.get("value")
    if operator in {"EQ", "NE", "IN"}:
        if operator == "IN":
            result = any(_norm(actual) == _norm(v) for v in expected or [])
        else:
            result = (_norm(actual) == _norm(expected) if isinstance(actual, str)
                      and isinstance(expected, str) else actual == expected)
            if operator == "NE":
                result = not result
        return (Truth.TRUE if result else Truth.FALSE), set()
    if operator in _NUMERIC:
        number = _number(actual)
        if number is None:
            return Truth.UNKNOWN, {variable}
        if operator == "BETWEEN":
            lo, hi = float(node["lower"]), float(node["upper"])
            result = ((number >= lo if node["lower_inclusive"] else number > lo) and
                      (number <= hi if node["upper_inclusive"] else number < hi))
        else:
            target = _number(expected)
            if target is None:
                return Truth.UNKNOWN, {variable}
            result = {"LT": number < target, "LE": number <= target,
                      "GT": number > target, "GE": number >= target}[operator]
        return (Truth.TRUE if result else Truth.FALSE), set()
    raise ValueError(f"Unknown predicate operator: {operator}")


def _and(left: Truth, right: Truth) -> Truth:
    if Truth.FALSE in (left, right):
        return Truth.FALSE
    if Truth.UNKNOWN in (left, right):
        return Truth.UNKNOWN
    return Truth.TRUE


def _scope(rule: dict, facts: dict) -> tuple[Truth, set[str]]:
    result, missing = Truth.TRUE, set()
    context = rule.get("context") or {}
    for key, expected in (("crop", rule.get("crop")),
                          *((k, context.get(k)) for k in ("crop", "region", "soil", "stage"))):
        if expected is None:
            continue
        present, actual = _fact(facts, key)
        if not present:
            result = _and(result, Truth.UNKNOWN)
            missing.add(key)
        elif not _same_or_phrase(str(actual), str(expected)):
            result = Truth.FALSE
    return result, missing


def _terms(rule: dict) -> list[str]:
    action = rule.get("action") or {}
    return [str(v) for v in (rule.get("product"), rule.get("nutrient"),
            action.get("target")) if v]


def _opaque_context(context: dict) -> list[str]:
    """Structured free-form payloads may encode untyped relations or branches."""
    opaque = []
    for key, value in (context.get("other") or {}).items():
        if isinstance(value, (dict, list)):
            opaque.append(key)
        elif isinstance(value, str) and value.lstrip().startswith(("{", "[")):
            try:
                decoded = json.loads(value)
            except json.JSONDecodeError:
                continue
            if isinstance(decoded, (dict, list)):
                opaque.append(key)
    return sorted(opaque)


def _related(rule: dict, request: dict) -> bool:
    subject = request["subject"]
    if not any(_same_or_phrase(subject, term) for term in _terms(rule)):
        return False
    for field in ("product", "nutrient"):
        asked = request.get(field)
        if asked and not _same_or_phrase(asked, rule.get(field) or ""):
            return False
    if request.get("action") and not _same_or_phrase(request["action"], rule["action"]["verb"]):
        return False
    return True


def _signature(rule: dict, requested_fields: list[str]) -> str:
    fields = set(requested_fields) | {"action", "modality", "product", "nutrient"}
    action = rule["action"]
    payload: dict[str, Any] = {"action": action, "modality": rule["modality"],
                               "product": rule.get("product"), "nutrient": rule.get("nutrient")}
    for field in ("quantity", "timing"):
        if field in fields:
            payload[field] = rule.get(field)
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def _source_refs(rule: dict, requested_fields: list[str]) -> list[dict]:
    refs = list(rule.get("evidence") or [])
    for field in requested_fields:
        refs.extend((rule.get("field_evidence") or {}).get(field) or [])
    return list({json.dumps(r, sort_keys=True): r for r in refs}.values())


def _render(rule: dict, request: dict, area_ha: float | None) -> tuple[str, list[str]]:
    action = rule["action"]
    parts = [f"{rule['modality']}: {action['verb']} {action.get('target') or ''}".strip()]
    if rule.get("product"):
        parts.append(f"product: {rule['product']}")
    if rule.get("nutrient"):
        parts.append(f"nutrient: {rule['nutrient']}")
    quantity = rule.get("quantity") or {}
    warnings: list[str] = []
    if quantity.get("value") is not None:
        parts.append(f"quantity: {quantity['value']} {quantity.get('unit') or ''}".strip())
    elif quantity.get("lower") is not None and quantity.get("upper") is not None:
        parts.append(f"quantity: {quantity['lower']}-{quantity['upper']} {quantity.get('unit') or ''}".strip())
    elif quantity.get("original"):
        parts.append(f"source quantity: {quantity['original']}")
    if quantity.get("basis"):
        parts.append(f"basis: {quantity['basis']}")
    if action.get("method"):
        parts.append(f"method: {action['method']}")
    if rule.get("timing"):
        parts.append(f"timing: {rule['timing']}")
    if area_ha is not None and "total_quantity" in request.get("fields", []):
        unit = quantity.get("unit")
        # Only identical area denominators are cancelled. No chemistry or split inference.
        match = re.fullmatch(r"(kg|g|t)/ha", str(unit or "").strip())
        if match and quantity.get("basis") and (quantity.get("value") is not None or
                                                 quantity.get("lower") is not None and quantity.get("upper") is not None):
            bounds = ([quantity["value"]] if quantity.get("value") is not None else
                      [quantity["lower"], quantity["upper"]])
            amount = "-".join(f"{float(n) * area_ha:g}" for n in bounds)
            parts.append(f"total for {area_ha:g} ha: {amount} {match.group(1)}; basis: {quantity['basis']}")
        else:
            warnings.append("UNSUPPORTED_SEMANTICS: area conversion requires numeric per-ha quantity and explicit basis")
    return "; ".join(parts), warnings


@dataclass
class _Candidate:
    rule: dict
    state: Truth
    missing: set[str]
    unsupported: list[str]
    trace: dict
    required_ids: list[str]


def _validate_request(request: dict) -> None:
    if not isinstance(request, dict) or not isinstance(request.get("requests"), list) or not request["requests"]:
        raise ValueError("S5Matcher input requires nonempty requests list")
    if not isinstance(request.get("facts"), dict):
        raise ValueError("S5Matcher input requires facts object")
    for item in request["requests"]:
        if not isinstance(item, dict) or not isinstance(item.get("subject"), str) or not item["subject"].strip():
            raise ValueError("Each request needs a nonempty subject")
        if not isinstance(item.get("fields"), list) or not all(
                isinstance(v, str) and v in FIELDS | {"total_quantity"} for v in item["fields"]):
            raise ValueError("Each request needs a list of supported fields")
    for name, value in request["facts"].items():
        if not isinstance(name, str) or not name:
            raise ValueError("Fact variable names must be nonempty strings")
        if isinstance(value, dict) and "status" in value and value["status"] not in {"EXPLICIT", "UNKNOWN", "AMBIGUOUS"}:
            raise ValueError("Unknown fact status")


def decide(request: dict, ruleset: dict) -> dict:
    """Return one traceable ANSWER, ASK, or ABSTAIN for the full request.

    ``status=UNSUPPORTED_SEMANTICS`` marks an abstention caused by S5's lack of
    machine-executable meaning, rather than an ordinary unsupported topic.
    """
    _validate_request(request)
    # Pydantic validation is mandatory; original dict is not modified.
    parse_ruleset("S5", ruleset)
    by_id = {r["rule_id"]: r for r in ruleset["rules"]}
    facts = request["facts"]
    traces: list[dict] = []
    chosen: list[dict] = []
    pending: set[str] = set()
    unsupported: list[str] = []
    for asked in request["requests"]:
        candidates: list[_Candidate] = []
        for rule in sorted(ruleset["rules"], key=lambda r: r["rule_id"]):
            if not _related(rule, asked):
                continue
            scope, scope_missing = _scope(rule, facts)
            condition, cond_missing = evaluate_ast(rule.get("conditions"), facts)
            state = _and(scope, condition)
            missing = scope_missing | cond_missing
            exception_states = []
            for node in rule.get("exceptions") or []:
                value, names = evaluate_ast(node, facts)
                exception_states.append(value.value)
                if value == Truth.TRUE:
                    state = Truth.FALSE
                elif value == Truth.UNKNOWN and state != Truth.FALSE:
                    state = Truth.UNKNOWN
                    missing |= names
            issues: list[str] = []
            required_ids: list[str] = []
            context = rule.get("context") or {}
            # context.other is descriptive unless it encodes a relation that
            # could change this prescription. It is never turned into a predicate.
            descriptive = sorted((context.get("other") or {}).keys())
            issues.extend(f"context.other.{key} has untyped structured semantics"
                          for key in _opaque_context(context))
            for dep in rule.get("dependencies") or []:
                target = by_id[dep["target_rule_id"]]
                kind = dep["kind"]
                if kind == "CONTEXT_FROM":
                    inherited_scope, inherited_missing = _scope(target, facts)
                    inherited_condition, inherited_condition_missing = evaluate_ast(target.get("conditions"), facts)
                    state = _and(state, _and(inherited_scope, inherited_condition))
                    missing |= inherited_missing | inherited_condition_missing
                    issues.extend(f"CONTEXT_FROM {target['rule_id']} context.other.{key} is untyped"
                                  for key in _opaque_context(target.get("context") or {}))
                elif kind == "REQUIRES":
                    required_ids.append(target["rule_id"])
                    required_scope, required_missing = _scope(target, facts)
                    required_condition, required_cond_missing = evaluate_ast(target.get("conditions"), facts)
                    state = _and(state, _and(required_scope, required_condition))
                    missing |= required_missing | required_cond_missing
                    issues.extend(f"REQUIRES {target['rule_id']} context.other.{key} is untyped"
                                  for key in _opaque_context(target.get("context") or {}))
                    for required_exception in target.get("exceptions") or []:
                        exclusion, exclusion_missing = evaluate_ast(required_exception, facts)
                        if exclusion == Truth.TRUE:
                            state = Truth.FALSE
                        elif exclusion == Truth.UNKNOWN and state != Truth.FALSE:
                            state = Truth.UNKNOWN
                            missing |= exclusion_missing
                    if target.get("dependencies"):
                        issues.append(f"REQUIRES {target['rule_id']} has transitive dependencies")
                elif kind in {"EXCEPTION_TO", "SAME_APPLICATION", "PRECEDES", "CROSS_REFERENCE"}:
                    # Direction, priority, timing and linkage are not fully specified by
                    # these labels. Preserve the edge, never invent its agronomic effect.
                    issues.append(f"{kind} relation to {target['rule_id']} lacks executable S5 semantics")
                else:
                    raise ValueError(f"Unexpected dependency: {kind}")
            for field in asked["fields"]:
                if field == "conversion":
                    issues.append("S5 has no executable product-to-nutrient conversion relation")
                elif field == "quantity":
                    q = rule.get("quantity") or {}
                    if q.get("value") is None and (q.get("lower") is None or q.get("upper") is None):
                        issues.append("requested numeric quantity absent from S5")
                elif field in {"unit", "basis"} and not (rule.get("quantity") or {}).get(field):
                    issues.append(f"requested {field} absent from S5")
                elif field == "timing" and rule.get("timing") is None:
                    issues.append("requested timing absent from S5")
                elif field == "method" and not rule["action"].get("method"):
                    issues.append("requested method absent from S5")
                elif field == "total_quantity" and _area(facts) is None:
                    missing.add("area_ha")
                    state = _and(state, Truth.UNKNOWN)
            trace = {"rule_id": rule["rule_id"], "request_subject": asked["subject"],
                     "scope": scope.value, "condition": condition.value,
                     "exceptions": exception_states, "state": state.value,
                     "missing": sorted(missing), "unsupported": sorted(set(issues)),
                     "dependencies": rule.get("dependencies") or [],
                     "required_rule_ids": required_ids,
                     "descriptive_context_keys": descriptive,
                     "evidence": _source_refs(rule, asked["fields"])}
            traces.append(trace)
            candidates.append(_Candidate(rule, state, missing, issues, trace, required_ids))
        if not candidates:
            return _result("ABSTAIN", "NO_S5_RULE", "No S5 rule matches a requested subject.",
                           [], traces, [], [])
        true = [c for c in candidates if c.state == Truth.TRUE]
        possible = [c for c in candidates if c.state == Truth.UNKNOWN]
        if any(c.unsupported for c in true + possible):
            unsupported.extend(f"{c.rule['rule_id']}: {issue}" for c in true + possible for issue in c.unsupported)
            continue
        if not true and not possible:
            return _result("ABSTAIN", "NO_APPLICABLE_RULE", "All matching rules are explicitly incompatible.",
                           [], traces, [], [])
        signatures = {_signature(c.rule, asked["fields"]) for c in true + possible}
        if possible and (not true or len(signatures) > 1):
            pending |= set().union(*(c.missing for c in possible))
            continue
        if len(signatures) > 1:
            unsupported.append("Multiple applicable S5 prescriptions conflict or lack explicit priority")
            continue
        # Equivalent prescriptions need only one representative; provenance retains all.
        for candidate in true:
            chosen.append(candidate.rule)
            chosen.extend(by_id[rid] for rid in candidate.required_ids)
    if unsupported:
        return _result("ABSTAIN", "UNSUPPORTED_SEMANTICS", "S5 cannot resolve the requested relation safely.",
                       [], traces, [], sorted(set(unsupported)))
    if pending:
        return _result("ASK", "MISSING_DECISIVE_FACT", "Please clarify: " + ", ".join(sorted(pending)) + ".",
                       [], traces, sorted(pending), [])
    if not chosen:
        return _result("ABSTAIN", "NO_APPLICABLE_RULE", "No applicable S5 prescription was established.",
                       [], traces, [], [])
    seen: dict[str, dict] = {r["rule_id"]: r for r in chosen}
    lines: list[str] = []
    render_issues: list[str] = []
    for rule in seen.values():
        asked = next((a for a in request["requests"] if _related(rule, a)),
                     {"subject": rule["action"].get("target") or "", "fields": []})
        line, warnings = _render(rule, asked, _area(facts))
        lines.append(line)
        render_issues.extend(warnings)
    if render_issues:
        return _result("ABSTAIN", "UNSUPPORTED_SEMANTICS", "Requested calculation is not justified by S5.",
                       [], traces, [], render_issues)
    refs = [ref for r in seen.values() for ref in _source_refs(r, list(FIELDS))]
    return _result("ANSWER", "OK", "\n".join(lines), list(seen), traces, [], [], refs)


def _result(label: str, status: str, response: str, rule_ids: list[str], traces: list[dict],
            missing: list[str], unsupported: list[str], evidence: list[dict] | None = None) -> dict:
    return {"engine": VERSION, "label": label, "status": status, "response": response,
            "rule_ids": rule_ids, "missing_decisive_facts": missing,
            "unsupported_semantics": unsupported, "evidence": evidence or [], "trace": traces}
