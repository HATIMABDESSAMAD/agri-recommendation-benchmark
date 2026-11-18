"""Lossless validation adapter for string-only S5 context metadata."""
import json
from copy import deepcopy

from .schema import parse_ruleset


def validated_copy(source: dict) -> tuple[dict, list[dict]]:
    data = deepcopy(source)
    changes = []
    for rule in data["rules"]:
        for key, value in ((rule.get("context") or {}).get("other") or {}).items():
            if not isinstance(value, str):
                rule["context"]["other"][key] = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
                changes.append({"rule_id": rule["rule_id"], "path": f"context.other.{key}",
                                "original_type": type(value).__name__})
    parse_ruleset("S5", data)
    return data, changes
