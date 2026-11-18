"""Pydantic v2 schemas S1–S5 and loss-aware evaluation representation."""
from __future__ import annotations

from enum import Enum
import json
from typing import Annotated, Literal, Union
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, model_validator


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Modality(str, Enum):
    MUST = "MUST"
    SHOULD = "SHOULD"
    MAY = "MAY"
    MUST_NOT = "MUST_NOT"
    UNSPECIFIED = "UNSPECIFIED"


class Presence(str, Enum):
    NOT_SPECIFIED = "NOT_SPECIFIED"
    AMBIGUOUS = "AMBIGUOUS"
    EXPLICIT = "EXPLICIT"
    INHERITED_FROM_CONTEXT = "INHERITED_FROM_CONTEXT"


class SourceRef(Strict):
    document_id: str
    page_number: int = Field(ge=1)
    element_id: str | None = None
    section_id: str | None = None
    table_id: str | None = None
    row_id: str | None = None
    column_id: str | None = None
    source_text: str = Field(min_length=1)


class Quantity(Strict):
    value: int | float | None = None
    lower: int | float | None = None
    upper: int | float | None = None
    unit: str | None = None
    basis: str | None = None
    original: str | None = None
    status: Presence = Presence.NOT_SPECIFIED

    @model_validator(mode="after")
    def check(self):
        if self.lower is not None and self.upper is not None and self.lower > self.upper:
            raise ValueError("lower > upper")
        if self.status == Presence.EXPLICIT and all(
                bound is None for bound in (self.value, self.lower, self.upper)):
            if not self.original or not self.original.strip():
                raise ValueError("explicit quantity without numeric value or original source text")
        return self


class Predicate(Strict):
    kind: Literal["PREDICATE"] = "PREDICATE"
    variable: str = Field(min_length=1)
    operator: Literal["EQ", "NE", "LT", "LE", "GT", "GE", "IN", "BETWEEN", "EXISTS"]
    value: str | int | float | list[str] | None = None
    lower: int | float | None = None
    upper: int | float | None = None
    lower_inclusive: bool | None = None
    upper_inclusive: bool | None = None
    unit: str | None = None

    @model_validator(mode="after")
    def check(self):
        if self.operator == "BETWEEN":
            if None in (self.lower, self.upper, self.lower_inclusive, self.upper_inclusive):
                raise ValueError("BETWEEN needs bounds and inclusivity")
            if self.lower > self.upper:
                raise ValueError("reversed interval")
        elif self.operator == "EXISTS":
            if self.value is not None:
                raise ValueError("EXISTS takes no value")
        elif self.value is None:
            raise ValueError("predicate needs value")
        return self


class Logic(Strict):
    kind: Literal["ALL", "ANY", "NOT"]
    children: list["Condition"]

    @model_validator(mode="after")
    def arity(self):
        if self.kind == "NOT" and len(self.children) != 1:
            raise ValueError("NOT needs one child")
        if self.kind in {"ALL", "ANY"} and len(self.children) < 2:
            raise ValueError("ALL/ANY need two or more children")
        return self


Condition = Annotated[Union[Predicate, Logic], Field(discriminator="kind")]
Logic.model_rebuild()


def check_depth(node: Predicate | Logic | None, maximum: int = 8, depth: int = 1) -> None:
    if node is None:
        return
    if depth > maximum:
        raise ValueError("AST depth exceeded")
    if isinstance(node, Logic):
        for child in node.children:
            check_depth(child, maximum, depth + 1)


class RuleBase(Strict):
    rule_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    schema_version: str
    crop: str | None = None
    modality: Modality = Modality.UNSPECIFIED
    product: str | None = None
    nutrient: str | None = None
    notes: str | None = None


class S1Rule(RuleBase):
    schema_version: Literal["S1"] = "S1"
    action: str = Field(min_length=1)
    dose: int | float | None = None
    unit: str | None = None
    basis: str | None = None
    timing: str | None = None
    method: str | None = None
    context: str | None = None
    conditions: str | None = None
    exceptions: str | None = None


class Action(Strict):
    verb: str = Field(min_length=1)
    target: str | None = None
    method: str | None = None


class Context(Strict):
    crop: str | None = None
    region: str | None = None
    soil: str | None = None
    stage: str | None = None
    other: dict[str, str] = Field(default_factory=dict)


class S2Rule(RuleBase):
    schema_version: Literal["S2"] = "S2"
    action: Action
    quantity: Quantity | None = None
    context: Context | None = None
    timing: str | None = None
    conditions: list[Predicate] = Field(default_factory=list)
    exceptions: list[str] = Field(default_factory=list)


class S3Rule(S2Rule):
    schema_version: Literal["S3"] = "S3"
    conditions: Condition | None = None
    exceptions: list[Condition] = Field(default_factory=list)

    @model_validator(mode="after")
    def depth(self, info: ValidationInfo):
        maximum = (info.context or {}).get("max_ast_depth", 8)
        check_depth(self.conditions, maximum)
        for node in self.exceptions:
            check_depth(node, maximum)
        return self


class S4Rule(S3Rule):
    schema_version: Literal["S4"] = "S4"
    evidence: list[SourceRef] = Field(min_length=1)
    field_evidence: dict[str, list[SourceRef]] = Field(default_factory=dict)


class Dependency(Strict):
    kind: Literal["REQUIRES", "EXCEPTION_TO", "SAME_APPLICATION", "PRECEDES", "CONTEXT_FROM", "CROSS_REFERENCE"]
    target_rule_id: str
    evidence: list[SourceRef] = Field(default_factory=list)


class S5Rule(S4Rule):
    schema_version: Literal["S5"] = "S5"
    dependencies: list[Dependency] = Field(default_factory=list)


RULES = {"S1": S1Rule, "S2": S2Rule, "S3": S3Rule, "S4": S4Rule, "S5": S5Rule}


class RuleSet(Strict):
    document_id: str
    schema_version: Literal["S1", "S2", "S3", "S4", "S5"]
    rules: list[S1Rule | S2Rule | S3Rule | S4Rule | S5Rule]

    @model_validator(mode="after")
    def integrity(self):
        ids = [r.rule_id for r in self.rules]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate rule_id")
        if any(r.document_id != self.document_id or r.schema_version != self.schema_version for r in self.rules):
            raise ValueError("document or schema mismatch")
        if self.schema_version == "S5":
            graph = {r.rule_id: [d.target_rule_id for d in r.dependencies] for r in self.rules}
            if any(t not in graph for targets in graph.values() for t in targets):
                raise ValueError("invalid dependency target")
            def visit(node: str, active: set[str], done: set[str]):
                if node in active:
                    raise ValueError("dependency cycle")
                if node in done:
                    return
                active.add(node)
                for target in graph[node]:
                    visit(target, active, done)
                active.remove(node)
                done.add(node)
            done: set[str] = set()
            for node in graph:
                visit(node, set(), done)
        return self


def parse_ruleset(schema: str, data: dict, max_ast_depth: int = 8) -> RuleSet:
    if schema not in RULES:
        raise ValueError(f"unknown schema {schema}")
    rules = [RULES[schema].model_validate_json(json.dumps(r, ensure_ascii=False),
                                               context={"max_ast_depth": max_ast_depth}) for r in data["rules"]]
    return RuleSet.model_validate({"document_id": data["document_id"], "schema_version": schema, "rules": rules})


class CanonicalRule(Strict):
    rule_id: str
    document_id: str
    crop: str | None = None
    action: str
    product: str | None = None
    nutrient: str | None = None
    quantity: Quantity | None = None
    modality: Modality
    timing: str | None = None
    method: str | None = None
    context: str | None = None
    condition_text: str | None = None
    condition_ast: Condition | None = None
    exception_text: list[str] = Field(default_factory=list)
    exception_ast: list[Condition] = Field(default_factory=list)
    evidence: list[SourceRef] = Field(default_factory=list)
    field_evidence: dict[str, list[SourceRef]] = Field(default_factory=dict)
    dependencies: list[Dependency] = Field(default_factory=list)
    representation_limits: list[str] = Field(default_factory=list)
    difficulty_tags: list[str] = Field(default_factory=list)


class CanonicalEvaluationRuleSet(Strict):
    document_id: str
    source_schema: str
    rules: list[CanonicalRule]
    losses: list[str] = Field(default_factory=list)


def to_canonical(ruleset: RuleSet) -> CanonicalEvaluationRuleSet:
    output = []
    for r in ruleset.rules:
        if isinstance(r, S1Rule):
            q = Quantity(value=r.dose, unit=r.unit, basis=r.basis,
                         status=Presence.EXPLICIT if r.dose is not None else Presence.NOT_SPECIFIED)
            item = CanonicalRule(rule_id=r.rule_id, document_id=r.document_id, crop=r.crop,
                action=r.action, product=r.product, nutrient=r.nutrient, quantity=q,
                modality=r.modality, timing=r.timing, method=r.method, context=r.context,
                condition_text=r.conditions, exception_text=[r.exceptions] if r.exceptions else [],
                representation_limits=["no_nested_action", "no_quantity_interval", "no_ast", "no_field_evidence", "no_dependencies"])
        else:
            ast = isinstance(r, S3Rule)
            item = CanonicalRule(rule_id=r.rule_id, document_id=r.document_id, crop=r.crop,
                action=f"{r.action.verb} {r.action.target or ''}".strip(), product=r.product,
                nutrient=r.nutrient, quantity=r.quantity, modality=r.modality, timing=r.timing,
                method=r.action.method, context=r.context.model_dump_json() if r.context else None,
                condition_ast=r.conditions if ast else None,
                condition_text=None if ast else "; ".join(p.model_dump_json() for p in r.conditions),
                exception_ast=r.exceptions if ast else [], exception_text=[] if ast else r.exceptions,
                evidence=r.evidence if isinstance(r, S4Rule) else [],
                field_evidence=r.field_evidence if isinstance(r, S4Rule) else {},
                dependencies=r.dependencies if isinstance(r, S5Rule) else [],
                representation_limits=[] if isinstance(r, S5Rule) else
                    ["no_dependencies"] if isinstance(r, S4Rule) else
                    ["no_field_evidence", "no_dependencies"] if isinstance(r, S3Rule) else
                    ["no_ast", "no_field_evidence", "no_dependencies"])
            if (item.quantity and item.quantity.status == Presence.EXPLICIT and
                    all(bound is None for bound in
                        (item.quantity.value, item.quantity.lower, item.quantity.upper))):
                item.representation_limits.append("unparsed_explicit_quantity")
        output.append(item)
    return CanonicalEvaluationRuleSet(document_id=ruleset.document_id,
        source_schema=ruleset.schema_version, rules=output,
        losses=sorted({limit for item in output for limit in item.representation_limits}))
