"""Independent synthetic questions and rules; no existing benchmark data."""
import json

import pytest

from agri_extract.s5_matcher import decide
from agri_extract.s5_query import extraction_messages, parse_s5_query, s5_vocabulary
from tests.s5_fixtures import rule, rules, pred


def query(question="For my orchard pears, how much material alpha on two hectares?", **overrides):
    payload = {"schema_version": "S5Query-1", "original_question": question,
               "requests": [{"subject": "material alpha", "crop": "orchard_pear",
                             "product": "material alpha", "nutrient": None, "action": None,
                             "fields": ["quantity", "total_quantity"],
                             "source_text": "material alpha"}],
               "facts": {"crop": {"status": "EXPLICIT", "value": "orchard_pear",
                                  "source_text": "orchard pears"},
                         "area_ha": {"status": "EXPLICIT", "value": 2,
                                     "source_text": "two hectares"}}}
    payload.update(overrides)
    return payload


def test_mock_llm_json_to_python_decision():
    question = "For my orchard pears, how much material alpha on two hectares?"
    parsed = parse_s5_query(json.dumps(query(question)), question)
    result = decide(parsed.engine_input(), rules(rule()))
    assert result["label"] == "ANSWER"
    assert "10 kg" in result["response"]
    assert result["evidence"][0]["page_number"] == 2


def test_question_text_cannot_be_changed_or_fabricated():
    with pytest.raises(ValueError):
        parse_s5_query(json.dumps(query(original_question="different")), query()["original_question"])
    bad = query()
    bad["facts"]["area_ha"]["source_text"] = "imagined area"
    with pytest.raises(ValueError):
        parse_s5_query(json.dumps(bad), bad["original_question"])


def test_ambiguous_fact_remains_unknown_to_matching():
    question = "For my orchard pears, how much material alpha on two hectares?"
    payload = query(question)
    payload["facts"]["moisture"] = {"status": "AMBIGUOUS", "value": None,
                                    "source_text": None}
    parsed = parse_s5_query(json.dumps(payload), question)
    result = decide(parsed.engine_input(), rules(rule(conditions=pred("moisture", "EQ", "adequate"))))
    assert result["label"] == "ASK"
    assert result["missing_decisive_facts"] == ["moisture"]


def test_descriptive_context_does_not_block():
    context = {"crop": "orchard_pear", "region": None, "soil": None, "stage": None,
               "other": {"study": "pilot observation"}}
    parsed = parse_s5_query(json.dumps(query()), query()["original_question"])
    assert decide(parsed.engine_input(), rules(rule(context=context)))["label"] == "ANSWER"


def test_untyped_structured_context_is_reported():
    context = {"crop": "orchard_pear", "region": None, "soil": None, "stage": None,
               "other": {"alternatives": '["option A","option B"]'}}
    parsed = parse_s5_query(json.dumps(query()), query()["original_question"])
    assert decide(parsed.engine_input(), rules(rule(context=context)))["status"] == "UNSUPPORTED_SEMANTICS"


def test_prompt_messages_contain_schema_and_s5_vocabulary_only():
    source = rules(rule(conditions=pred("moisture", "EQ", "adequate")))
    vocabulary = s5_vocabulary(source)
    assert "moisture" in vocabulary["variables"]
    messages = extraction_messages(query()["original_question"], source, "Extract JSON only")
    assert len(messages) == 2
    assert "gold_label" not in json.dumps(messages)
    assert "ANSWER" not in messages[0]["content"]
