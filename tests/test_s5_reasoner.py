"""Synthetic S5 and question fixtures; no benchmark labels."""
import json

import pytest

from agri_extract.s5_reasoner import messages, parse_decision
from agri_extract.s5_query import S5Query
from tests.s5_fixtures import rule, rules


def query():
    return S5Query.model_validate({"schema_version": "S5Query-1",
      "original_question": "How much material alpha?",
      "requests": [{"subject": "material alpha", "crop": None, "product": None,
                    "nutrient": None, "action": None, "fields": ["quantity"],
                    "source_text": "material alpha"}], "facts": {}})


def test_messages_include_all_rules_and_query_without_gold():
    source = rules(rule("a"), rule("b", target="material beta"))
    result = messages(query().original_question, query(), source)
    payload = json.loads(result[1]["content"])
    assert len(payload["s5_ruleset"]["rules"]) == 2
    assert payload["s5_query"]["requests"][0]["subject"] == "material alpha"
    assert "gold_label" not in json.dumps(result)


def test_valid_answer_requires_real_rule_and_page():
    source = rules(rule("a"))
    decision = {"label": "ANSWER", "response": "5 kg/ha", "justification": "S5 rule a",
                "rule_ids": ["a"], "cited_pages": [2], "missing_facts": []}
    assert parse_decision(json.dumps(decision), source)["label"] == "ANSWER"
    decision["cited_pages"] = [99]
    with pytest.raises(ValueError):
        parse_decision(json.dumps(decision), source)
    decision["cited_pages"] = [2]
    decision["rule_ids"] = ["absent"]
    with pytest.raises(ValueError):
        parse_decision(json.dumps(decision), source)


def test_ask_needs_specific_missing_fact_and_query_identity():
    source = rules(rule())
    with pytest.raises(ValueError):
        messages("different", query(), source)
    payload = {"label": "ASK", "response": "What is the soil?", "justification": "unknown",
               "rule_ids": [], "cited_pages": [], "missing_facts": []}
    with pytest.raises(ValueError):
        parse_decision(json.dumps(payload), source)
