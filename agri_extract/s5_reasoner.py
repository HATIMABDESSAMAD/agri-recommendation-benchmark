"""S5-only LLM reasoning contract; no gold labels or document-specific policy."""
from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .s5_query import S5Query
from .schema import parse_ruleset


class S5ReasonerDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    label: Literal["ANSWER", "ASK", "ABSTAIN"]
    response: str = Field(min_length=1)
    justification: str = Field(min_length=1)
    rule_ids: list[str] = Field(default_factory=list)
    cited_pages: list[int] = Field(default_factory=list)
    missing_facts: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def decision_shape(self):
        if self.label == "ANSWER" and not self.rule_ids:
            raise ValueError("ANSWER requires at least one supporting S5 rule")
        if self.label == "ASK" and not self.missing_facts:
            raise ValueError("ASK requires a concrete missing fact")
        return self


SYSTEM_PROMPT = """You are an S5 rule reasoner. Use only the supplied S5 rules and explicitly stated farmer facts in S5Query. Do not use general agricultural knowledge, the full PDF, external sources, or any human evaluation labels. Do not invent a crop, product equivalence, condition, dose, conversion, timing, or priority.
Evaluate every relevant S5 rule, its context, AND/OR/NOT conditions, exceptions and dependencies. Missing facts remain unknown. Return ANSWER only when the requested answer is fully supported; ASK only when a specific farmer fact could change the applicable rule or answer; otherwise ABSTAIN. If S5 is silent or its free-form metadata is too ambiguous to execute, abstain and explain. A conditional statement alone does not satisfy a request for a farm-specific decision when the condition is unknown.
Return one JSON object only with exactly: label (ANSWER|ASK|ABSTAIN), response, justification, rule_ids (array of exact S5 rule IDs), cited_pages (array of page numbers from those rules' evidence), missing_facts (array of variable names). For an ANSWER cite at least one rule and its source page. Never cite a page or rule absent from S5."""


def messages(question: str, query: S5Query, ruleset: dict) -> list[dict]:
    parse_ruleset("S5", ruleset)
    if query.original_question != question:
        raise ValueError("Question and frozen S5Query differ")
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps({"question": question,
                 "s5_query": query.model_dump(), "s5_ruleset": ruleset}, ensure_ascii=False)}]


def parse_decision(text: str, ruleset: dict) -> dict:
    """Validate citations and retain unsupported response as an invalid output."""
    parse_ruleset("S5", ruleset)
    decision = S5ReasonerDecision.model_validate_json(text)
    by_id = {rule["rule_id"]: rule for rule in ruleset["rules"]}
    if any(rid not in by_id for rid in decision.rule_ids):
        raise ValueError("Decision cites an unknown S5 rule")
    pages = {ref["page_number"] for rid in decision.rule_ids
             for ref in by_id[rid]["evidence"]}
    if any(page not in pages for page in decision.cited_pages):
        raise ValueError("Decision cites a page absent from cited S5 rules")
    if decision.label == "ANSWER" and not decision.cited_pages:
        raise ValueError("ANSWER requires an S5 evidence page")
    return decision.model_dump()
