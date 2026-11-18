"""Deterministic one-to-one rule alignment; matching thresholds are configuration."""
from __future__ import annotations

import re
from .schema import CanonicalRule


def tokens(value: str | None) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (value or "").casefold()))


def similarity(a: CanonicalRule, b: CanonicalRule) -> float:
    if a.document_id != b.document_id:
        return 0.0
    if a.product and b.product and a.product.casefold() != b.product.casefold():
        return 0.0
    if a.crop and b.crop and a.crop.casefold() != b.crop.casefold():
        return 0.0
    x, y = tokens(a.action), tokens(b.action)
    action = len(x & y) / len(x | y) if x | y else 0.0
    pages_a = {r.page_number for r in a.evidence}
    pages_b = {r.page_number for r in b.evidence}
    page = 1.0 if pages_a and pages_b and pages_a & pages_b else 0.0
    timing = 1.0 if a.timing and b.timing and tokens(a.timing) == tokens(b.timing) else 0.0
    return 0.75 * action + 0.15 * page + 0.10 * timing


def align(left: list[CanonicalRule], right: list[CanonicalRule], threshold: float = 0.55) -> dict:
    proposals = []
    contested = []
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            score = similarity(a, b)
            if score >= threshold:
                proposals.append((score, i, j))
            elif (a.document_id == b.document_id and a.action.split()[:1] == b.action.split()[:1]
                  and {r.page_number for r in a.evidence} & {r.page_number for r in b.evidence}
                  and a.product != b.product):
                contested.append({"left": a.rule_id, "right": b.rule_id,
                                  "reason": "same action/page but different product; manual review"})
    proposals.sort(key=lambda x: (-x[0], left[x[1]].rule_id, right[x[2]].rule_id))
    used_l, used_r, matches = set(), set(), []
    for score, i, j in proposals:
        if i not in used_l and j not in used_r:
            matches.append({"left": left[i].rule_id, "right": right[j].rule_id, "score": round(score, 4)})
            used_l.add(i); used_r.add(j)
    review = [m for m in matches if m["score"] < threshold + 0.1]
    return {"matches": matches,
            "left_only": [r.rule_id for i, r in enumerate(left) if i not in used_l],
            "right_only": [r.rule_id for j, r in enumerate(right) if j not in used_r],
            "borderline": review, "threshold": threshold,
            "contested": contested,
            "unverified_union": [r.rule_id for r in left + right],
            "intersection": [m["left"] for m in matches]}


def compare(left: list[CanonicalRule], right: list[CanonicalRule], threshold: float = 0.55) -> dict:
    report = align(left, right, threshold)
    lmap, rmap = {r.rule_id: r for r in left}, {r.rule_id: r for r in right}
    conflicts = []
    for match in report["matches"]:
        a, b = lmap[match["left"]], rmap[match["right"]]
        fields = []
        for field in ("modality", "timing", "condition_text", "condition_ast", "exception_text", "exception_ast"):
            if getattr(a, field) != getattr(b, field):
                fields.append(field)
        if a.quantity != b.quantity:
            fields.append("quantity")
        if fields:
            conflicts.append({"left": a.rule_id, "right": b.rule_id, "fields": fields})
    report["conflicts"] = conflicts
    return report
