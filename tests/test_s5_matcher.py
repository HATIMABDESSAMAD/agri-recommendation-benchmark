"""Independent synthetic tests of the retained deterministic matcher."""
import json
import sys

from agri_extract import s5_matcher_cli
from agri_extract.s5_matcher import Truth, decide, evaluate_ast
from tests.s5_fixtures import pred, rule, rules


def request(**facts):
    return {"facts": {"crop": "orchard_pear", **facts},
            "requests": [{"subject": "material alpha", "fields": ["quantity"]}]}


def test_cli_freezes_rule_files_independently(tmp_path, monkeypatch):
    monkeypatch.setattr(s5_matcher_cli, "MANIFEST_ROOT", tmp_path / "manifests")
    for name in ("first", "second"):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(rules(rule(name))), encoding="utf-8")
        monkeypatch.setattr(sys, "argv", ["s5_matcher_cli", "freeze", "--rules", str(path)])
        s5_matcher_cli.main()
        assert s5_matcher_cli.manifest_path(path).exists()
        out = tmp_path / f"{name}_messages.json"
        monkeypatch.setattr(sys, "argv", ["s5_matcher_cli", "messages", "--rules", str(path),
                                              "--question", "How much?", "--out", str(out)])
        s5_matcher_cli.main()
        assert out.exists()


def test_three_valued_logic_and_missing_decisive_fact():
    node = {"kind": "ALL", "children": [pred("rain", "EQ", "wet"),
                                           {"kind": "NOT", "children": [pred("pH", "LT", 5)]}]}
    assert evaluate_ast(node, {"rain": "wet", "pH": 6})[0] == Truth.TRUE
    assert evaluate_ast(node, {"rain": "dry"})[0] == Truth.FALSE
    assert evaluate_ast(node, {"rain": "wet"})[0] == Truth.UNKNOWN
    assert decide(request(), rules(rule(conditions=node)))["label"] == "ASK"


def test_exception_and_alternative_rule():
    excluded = rule("excluded", exceptions=[pred("slope", "GT", 20)])
    alternative = rule("alternative")
    result = decide(request(slope=25), rules(excluded, alternative))
    assert result["label"] == "ANSWER"
    assert result["rule_ids"] == ["alternative"]


def test_conflict_and_rule_order():
    a = rule("a")
    b = rule("b", quantity={**rule()["quantity"], "value": 8})
    first = decide(request(), rules(a, b))
    second = decide(request(), rules(b, a))
    assert first == second
    assert first["status"] == "UNSUPPORTED_SEMANTICS"


def test_missing_nondecisive_condition_does_not_force_ask():
    fixed = rule("fixed")
    conditional = rule("conditional", conditions=pred("rain", "EQ", "wet"))
    assert decide(request(), rules(fixed, conditional))["label"] == "ANSWER"


def test_direct_dependency_and_citation():
    parent = rule("parent", target="material beta",
                  conditions=pred("rain", "EQ", "wet"))
    child = rule("child", dependencies=[{"kind": "REQUIRES", "target_rule_id": "parent",
                                          "evidence": []}])
    assert decide(request(), rules(parent, child))["label"] == "ASK"
    outcome = decide(request(rain="wet"), rules(parent, child))
    assert outcome["label"] == "ANSWER"
    assert outcome["rule_ids"] == ["child", "parent"]
    assert outcome["evidence"][0]["page_number"] == 2
