"""Typed, label-blind extraction contract for farmer questions."""
from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .schema import parse_ruleset


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class QueryFact(Strict):
    status: Literal["EXPLICIT", "UNKNOWN", "AMBIGUOUS"]
    value: str | int | float | bool | None = None
    source_text: str | None = None

    @model_validator(mode="after")
    def evidence(self):
        if self.status == "EXPLICIT" and (self.value is None or not self.source_text):
            raise ValueError("Explicit facts require a value and source_text")
        if self.status != "EXPLICIT" and self.value is not None:
            raise ValueError("Unknown and ambiguous facts cannot assert a value")
        return self


class QueryObject(Strict):
    subject: str = Field(min_length=1)
    crop: str | None = None
    product: str | None = None
    nutrient: str | None = None
    action: str | None = None
    fields: list[Literal["action", "modality", "product", "nutrient", "quantity",
                         "unit", "basis", "timing", "method", "evidence",
                         "total_quantity", "conversion"]] = Field(min_length=1)
    source_text: str = Field(min_length=1)


class S5Query(Strict):
    schema_version: Literal["S5Query-1"] = "S5Query-1"
    original_question: str = Field(min_length=1)
    requests: list[QueryObject] = Field(min_length=1)
    facts: dict[str, QueryFact] = Field(default_factory=dict)

    @model_validator(mode="after")
    def grounded(self):
        if any(f.source_text and f.source_text not in self.original_question for f in self.facts.values()):
            raise ValueError("Fact source_text must be a verbatim substring of the question")
        if any(r.source_text not in self.original_question for r in self.requests):
            raise ValueError("Request source_text must be a verbatim substring of the question")
        return self

    def engine_input(self) -> dict:
        facts = {name: fact.model_dump(exclude_none=True) for name, fact in self.facts.items()}
        requests = []
        for item in self.requests:
            requests.append({"subject": item.subject, "fields": item.fields,
                             "crop": item.crop, "product": item.product,
                             "nutrient": item.nutrient, "action": item.action})
            if item.crop and "crop" not in facts:
                facts["crop"] = {"status": "EXPLICIT", "value": item.crop,
                                 "source_text": item.source_text}
        return {"facts": facts, "requests": requests}


def parse_s5_query(content: str, original_question: str) -> S5Query:
    """Parse one complete LLM JSON object; never repair or infer missing facts."""
    value = json.loads(content)
    query = S5Query.model_validate(value)
    if query.original_question != original_question:
        raise ValueError("The model changed the original question")
    return query


def _variables(node: dict | None, output: set[str]) -> None:
    if not node:
        return
    if node["kind"] == "PREDICATE":
        output.add(node["variable"])
    else:
        for child in node["children"]:
            _variables(child, output)


def s5_vocabulary(ruleset: dict) -> dict[str, Any]:
    """Read-only vocabulary; no gold data or decision examples."""
    parse_ruleset("S5", ruleset)
    variables = {"crop", "region", "soil", "stage", "area_ha"}
    subjects = set()
    values: dict[str, set[str]] = {}
    for rule in ruleset["rules"]:
        for term in (rule.get("crop"), rule.get("product"), rule.get("nutrient"),
                     rule["action"].get("target")):
            if term:
                subjects.add(term)
        for node in [rule.get("conditions"), *(rule.get("exceptions") or [])]:
            _variables(node, variables)
            if node and node["kind"] == "PREDICATE" and isinstance(node.get("value"), (str, list)):
                candidates = node["value"] if isinstance(node["value"], list) else [node["value"]]
                values.setdefault(node["variable"], set()).update(str(x) for x in candidates)
        context = rule.get("context") or {}
        for name in ("crop", "region", "soil", "stage"):
            if context.get(name):
                values.setdefault(name, set()).add(context[name])
    return {"variables": sorted(variables), "subjects": sorted(subjects),
            "documented_values": {k: sorted(v) for k, v in sorted(values.items())}}


def extraction_messages(question: str, ruleset: dict, system_prompt: str) -> list[dict]:
    """Messages for a caller-owned LLM adapter; this module makes no network call."""
    return [{"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps({"question": question,
                   "s5_query_schema": S5Query.model_json_schema(),
                   "s5_vocabulary": s5_vocabulary(ruleset)}, ensure_ascii=False)}]
