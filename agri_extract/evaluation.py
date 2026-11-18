"""Gold-guarded metrics. Unavailable metrics are null, never invented zeros."""
from __future__ import annotations

import json
from typing import Literal

from pydantic import Field
from .alignment import align
from .documents import DocumentArtifact
from .schema import CanonicalEvaluationRuleSet, CanonicalRule, Logic, Predicate, Strict


class AnnotationDisagreement(Strict):
    rule_id: str | None = None
    field: str
    positions: dict[str, str] = Field(default_factory=dict)
    status: Literal["OPEN", "RESOLVED"] = "OPEN"
    resolution: str | None = None


class GoldAgriculturalRuleSet(Strict):
    document_id: str
    source_hash: str
    page_count: int = Field(ge=1)
    split: Literal["development", "validation", "test", "unassigned"]
    status: Literal["UNREVIEWED", "DRAFT", "ADJUDICATED"] = "UNREVIEWED"
    annotators: list[str] = Field(default_factory=list)
    adjudication_notes: list[str] = Field(default_factory=list)
    annotation_disagreements: list[AnnotationDisagreement] = Field(default_factory=list)
    single_annotator_limitation: str | None = None
    rules: list[CanonicalRule] = Field(default_factory=list)
    reviewed_pages: list[int] = Field(default_factory=list)


DIFFICULTIES = ["simple", "multiple_actions", "multiple_conditions", "and_or",
    "negation", "exceptions", "quantity", "threshold", "timing", "table", "note",
    "inherited_context", "cross_section_dependency", "ambiguous"]


def gold_template(document_id: str, source_hash: str, pages: int, split: str = "unassigned") -> GoldAgriculturalRuleSet:
    return GoldAgriculturalRuleSet(document_id=document_id, source_hash=source_hash,
                                   page_count=pages, split=split, reviewed_pages=[])


def check_gold(gold: GoldAgriculturalRuleSet, artifact: DocumentArtifact | None = None,
               source_hash: str | None = None) -> dict:
    errors = []
    expected_hash = artifact.file_hash if artifact else source_hash
    if artifact and (gold.document_id != artifact.document_id or gold.page_count != len(artifact.pages)):
        errors.append("document identity or page count differs from artifact")
    if expected_hash is not None and gold.source_hash != expected_hash:
        errors.append("source PDF hash differs from gold")
    expected_pages = set(range(1, gold.page_count + 1))
    if set(gold.reviewed_pages) != expected_pages or len(gold.reviewed_pages) != gold.page_count:
        errors.append("every PDF page must be reviewed exactly once")
    if not gold.annotators:
        errors.append("at least one named human annotator is required")
    if len(gold.annotators) == 1 and not gold.single_annotator_limitation:
        errors.append("single-annotator limitation must be documented")
    if any(d.status != "RESOLVED" or not d.resolution for d in gold.annotation_disagreements):
        errors.append("annotation disagreements remain unresolved")
    ids = [r.rule_id for r in gold.rules]
    if len(ids) != len(set(ids)):
        errors.append("duplicate gold rule IDs")
    for rule in gold.rules:
        if rule.document_id != gold.document_id:
            errors.append(f"{rule.rule_id}: wrong document ID")
        if not rule.evidence:
            errors.append(f"{rule.rule_id}: source evidence is required")
        for ref in [*rule.evidence, *(r for refs in rule.field_evidence.values() for r in refs)]:
            if ref.document_id != gold.document_id or ref.page_number not in expected_pages:
                errors.append(f"{rule.rule_id}: invalid source reference")
            elif artifact:
                page = artifact.pages[ref.page_number - 1]
                compact = lambda text: " ".join(text.split()).casefold()
                if compact(ref.source_text) not in compact(page.raw_text):
                    errors.append(f"{rule.rule_id}: quote absent from original PDF page {ref.page_number}")
    return {"status": "READY" if gold.status == "ADJUDICATED" and not errors else "NOT_READY",
            "gold_status": gold.status, "errors": errors, "rule_count": len(gold.rules),
            "reviewed_page_count": len(gold.reviewed_pages)}


def _prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall else None
    return {"precision": precision, "recall": recall, "f1": f1}


def _condition_key(r: CanonicalRule) -> str | None:
    if r.condition_ast:
        return r.condition_ast.model_dump_json()
    return r.condition_text.casefold().strip() if r.condition_text else None


def _exception_key(r: CanonicalRule) -> str | None:
    if r.exception_ast:
        return json.dumps([x.model_dump(mode="json") for x in r.exception_ast], sort_keys=True)
    return "|".join(x.casefold().strip() for x in r.exception_text) if r.exception_text else None


def _operators(node: Predicate | Logic | None) -> dict[str, str]:
    if node is None:
        return {}
    if isinstance(node, Predicate):
        return {node.variable: node.operator}
    merged = {}
    for child in node.children:
        merged.update(_operators(child))
    return merged


def _same_ref(a, b) -> bool:
    x, y = a.source_text.casefold().strip(), b.source_text.casefold().strip()
    return a.page_number == b.page_number and bool(x and y) and (x in y or y in x)


def evaluate(pred: CanonicalEvaluationRuleSet, gold: GoldAgriculturalRuleSet,
             source_hash: str, threshold: float = 0.55) -> dict:
    gate = check_gold(gold, source_hash=source_hash)
    if gate["status"] != "READY" or gold.document_id != pred.document_id:
        return {"status": "NOT_EVALUATED", "reason": "gold is not adjudicated or failed quality checks",
                "gold_check": gate,
                "metrics": None}
    matched = align(pred.rules, gold.rules, threshold)
    pmap = {r.rule_id: r for r in pred.rules}
    gmap = {r.rule_id: r for r in gold.rules}
    rec = _prf(len(matched["matches"]), len(matched["left_only"]), len(matched["right_only"]))
    condition_tp = condition_fp = condition_fn = 0
    exception_tp = exception_fp = exception_fn = 0
    numeric_correct = numeric_total = unit_correct = unit_total = association_correct = association_total = 0
    logical_correct = logical_total = 0
    operator_correct = operator_total = 0
    modality_correct = timing_correct = 0
    evidence_correct = evidence_total = grounding_correct = grounding_total = 0
    for pair in matched["matches"]:
        a, b = pmap[pair["left"]], gmap[pair["right"]]
        for key_fn, counts in ((_condition_key, "condition"), (_exception_key, "exception")):
            x, y = key_fn(a), key_fn(b)
            if counts == "condition":
                condition_tp += int(x is not None and x == y)
                condition_fp += int(x is not None and x != y)
                condition_fn += int(y is not None and x != y)
            else:
                exception_tp += int(x is not None and x == y)
                exception_fp += int(x is not None and x != y)
                exception_fn += int(y is not None and x != y)
        if b.quantity and (b.quantity.value is not None or b.quantity.lower is not None):
            numeric_total += 1
            numeric_correct += int(bool(a.quantity) and (a.quantity.value, a.quantity.lower, a.quantity.upper) ==
                                   (b.quantity.value, b.quantity.lower, b.quantity.upper))
            unit_total += 1
            unit_correct += int(bool(a.quantity) and a.quantity.unit == b.quantity.unit)
            association_total += 1
            association_correct += int(a.product == b.product and bool(a.quantity) and a.quantity.value == b.quantity.value)
        if b.condition_ast:
            logical_total += 1
            logical_correct += int(a.condition_ast == b.condition_ast)
            for variable, operator in _operators(b.condition_ast).items():
                operator_total += 1
                operator_correct += int(_operators(a.condition_ast).get(variable) == operator)
        modality_correct += int(a.modality == b.modality)
        timing_correct += int(a.timing == b.timing)
        if b.evidence:
            evidence_total += 1
            evidence_correct += int(any(_same_ref(x, y) for x in a.evidence for y in b.evidence))
        for field, refs in b.field_evidence.items():
            if refs:
                grounding_total += 1
                grounding_correct += int(any(_same_ref(x, y) for x in a.field_evidence.get(field, []) for y in refs))
    semantic_keys = [(r.action.casefold(), (r.product or "").casefold(), r.timing,
                      r.quantity.value if r.quantity else None) for r in pred.rules]
    metrics = {"recommendation": rec,
               "condition": _prf(condition_tp, condition_fp, condition_fn),
               "exception": _prf(exception_tp, exception_fp, exception_fn),
               "omission_rate": len(matched["right_only"]) / len(gold.rules) if gold.rules else None,
               "hallucinated_rule_rate": len(matched["left_only"]) / len(pred.rules) if pred.rules else None,
               "numeric_accuracy": numeric_correct / numeric_total if numeric_total else None,
               "unit_accuracy": unit_correct / unit_total if unit_total else None,
               "product_dose_association_accuracy": association_correct / association_total if association_total else None,
               "logical_exact_match": logical_correct / logical_total if logical_total else None,
               "operator_accuracy": operator_correct / operator_total if operator_total else None,
               "modality_accuracy": modality_correct / len(matched["matches"]) if matched["matches"] else None,
               "timing_accuracy": timing_correct / len(matched["matches"]) if matched["matches"] else None,
               "evidence_support_rate": evidence_correct / evidence_total if evidence_total else None,
               "field_grounding_accuracy": grounding_correct / grounding_total if grounding_total else None,
               "duplicate_rate": (len(semantic_keys) - len(set(semantic_keys))) / len(semantic_keys) if semantic_keys else None,
               "schema_validity_rate": 1.0,
               "cost": None, "latency": None}
    matched_gold_ids = {m["right"] for m in matched["matches"]}
    by_difficulty = {}
    for tag in DIFFICULTIES:
        tagged = [r for r in gold.rules if tag in r.difficulty_tags]
        by_difficulty[tag] = ({"status": "NOT_EVALUATED", "count": 0, "recall": None}
            if not tagged else {"status": "EVALUATED", "count": len(tagged),
                "recall": sum(r.rule_id in matched_gold_ids for r in tagged) / len(tagged)})
    return {"status": "EVALUATED", "metrics": metrics, "matching": matched,
            "by_difficulty": by_difficulty, "review_required": matched["borderline"]}
