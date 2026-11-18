"""Non-recursive transport of condition ASTs for constrained JSON providers."""
from __future__ import annotations

from .schema import Condition, Logic, Predicate


def flatten(node: Condition | None) -> dict | None:
    if node is None:
        return None
    nodes = []
    def visit(item):
        identifier = f"n{len(nodes)+1}"
        record = {"id": identifier, "kind": item.kind, "children": []}
        nodes.append(record)
        if isinstance(item, Logic):
            record["children"] = [visit(child) for child in item.children]
        else:
            record.update(item.model_dump(mode="json", exclude={"kind"}, exclude_none=True))
        return identifier
    root = visit(node)
    return {"root": root, "nodes": nodes}


def restore(data: dict | None) -> Condition | None:
    if data is None:
        return None
    by_id = {n["id"]: n for n in data["nodes"]}
    if len(by_id) != len(data["nodes"]) or data["root"] not in by_id:
        raise ValueError("invalid node IDs")
    def visit(identifier: str, active: set[str]):
        if identifier in active:
            raise ValueError("AST transport cycle")
        if identifier not in by_id:
            raise ValueError("missing AST node")
        record = by_id[identifier]
        active = active | {identifier}
        if record["kind"] in {"ALL", "ANY", "NOT"}:
            return Logic.model_validate({"kind": record["kind"],
                "children": [visit(child, active) for child in record.get("children", [])]})
        if record["kind"] == "PREDICATE":
            if record.get("children"):
                raise ValueError("predicate cannot have children")
            return Predicate.model_validate({k: v for k, v in record.items() if k not in {"id", "children"}})
        raise ValueError("unknown AST kind")
    return visit(data["root"], set())


def restore_ruleset(data: dict) -> dict:
    restored = {**data, "rules": []}
    for item in data["rules"]:
        row = item.copy()
        row["conditions"] = restore(row.pop("conditions_transport", None))
        row["conditions"] = row["conditions"].model_dump(mode="json") if row["conditions"] else None
        row["exceptions"] = [restore(x).model_dump(mode="json") for x in row.pop("exceptions_transport", [])]
        restored["rules"].append(row)
    return restored


FLAT_AST = {"type": "object", "properties": {
    "root": {"type": "string"},
    "nodes": {"type": "array", "items": {"type": "object", "properties": {
        "id": {"type": "string"}, "kind": {"type": "string", "enum": ["ALL", "ANY", "NOT", "PREDICATE"]},
        "children": {"type": "array", "items": {"type": "string"}},
        "variable": {"type": ["string", "null"]},
        "operator": {"type": ["string", "null"]},
        "value": {}, "lower": {"type": ["number", "null"]}, "upper": {"type": ["number", "null"]},
        "lower_inclusive": {"type": ["boolean", "null"]},
        "upper_inclusive": {"type": ["boolean", "null"]},
        "unit": {"type": ["string", "null"]}},
        "required": ["id", "kind", "children"], "additionalProperties": False}}},
    "required": ["root", "nodes"], "additionalProperties": False}


def transport_schema(ruleset_schema: dict) -> dict:
    import copy
    schema = copy.deepcopy(ruleset_schema)
    item = schema["properties"]["rules"]["items"]
    props = item["properties"]
    props.pop("conditions", None)
    props.pop("exceptions", None)
    props["conditions_transport"] = {"anyOf": [FLAT_AST, {"type": "null"}]}
    props["exceptions_transport"] = {"type": "array", "items": FLAT_AST}
    required = set(item.get("required", [])) - {"conditions", "exceptions"}
    required.update({"conditions_transport", "exceptions_transport"})
    item["required"] = sorted(required)
    # Unused recursive definitions are removed; the transport graph is flat.
    for name in ("Logic", "Predicate"):
        schema.get("$defs", {}).pop(name, None)
    return schema
