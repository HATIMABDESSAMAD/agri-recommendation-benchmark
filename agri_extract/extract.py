"""E1–E8 strategies. Calls and intermediate responses are always retained."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .documents import DocumentArtifact
from .providers import Provider
from .schema import RULES
from .transport import restore_ruleset, transport_schema


PROMPT_ROOT = Path(__file__).resolve().parent / "prompts"


def prompt(name: str, version: str = "v1") -> str:
    if not version.replace("_", "").replace("-", "").isalnum():
        raise ValueError("invalid prompt version")
    from .prepare import ROOT
    path = ROOT / "prompts" / version / f"{name}.txt"
    if not path.exists():
        path = PROMPT_ROOT / version / f"{name}.txt"
    return path.read_text(encoding="utf-8")


def parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        opening_end = text.find("\n")
        if opening_end < 0:
            raise json.JSONDecodeError("unterminated JSON fence", text, 0)
        language = text[3:opening_end].strip().lower()
        if language not in {"", "json"}:
            raise json.JSONDecodeError("unexpected fence language", text, 0)
        closing = text.find("\n```", opening_end + 1)
        if closing < 0:
            raise json.JSONDecodeError("unterminated JSON fence", text, 0)
        body = text[opening_end + 1:closing].strip()
        trailing = text[closing + 4:].strip()
        if trailing.startswith(("{", "[", "```")):
            raise json.JSONDecodeError("second JSON payload after fence", text, closing + 4)
        return json.loads(body)
    return json.loads(text)


def _ruleset_schema(name: str) -> dict:
    # Native provider support can vary; strict local Pydantic validation is separate.
    item = RULES[name].model_json_schema()
    definitions = item.pop("$defs", {})
    return {"type": "object", "$defs": definitions, "properties": {
        "document_id": {"type": "string"},
        "rules": {"type": "array", "items": item}},
        "required": ["document_id", "rules"], "additionalProperties": False}


def _call(provider: Provider, stage: str, artifact: DocumentArtifact, schema: str,
          trace: list[dict], prior: Any = None, max_output_tokens: int = 8192,
          protocol: str = "common", prompt_version: str = "v1") -> dict:
    system = prompt(stage, prompt_version)
    flat_transport = protocol == "native" and schema in {"S3", "S4", "S5"} and stage in {
        "joint_extraction", "independent_extraction", "condition_extraction"}
    if flat_transport:
        system += "\nFor each rule, use conditions_transport and exceptions_transport flat node graphs. Do not output recursive conditions/exceptions."
    output_schema = transport_schema(_ruleset_schema(schema)) if flat_transport else _ruleset_schema(schema)
    user = (f"DOCUMENT_ID: {artifact.document_id}\nSCHEMA: {schema}\n"
            f"DOCUMENT:\n{artifact.llm_text()}\n\n"
            f"PRIOR_STAGE:\n{json.dumps(prior, ensure_ascii=False) if prior is not None else 'none'}\n"
            f"OUTPUT_SCHEMA:\n{json.dumps(output_schema, ensure_ascii=False)}")
    completion = provider.complete(system, user,
        json_schema=output_schema if protocol == "native" and stage in {
            "joint_extraction", "independent_extraction", "condition_extraction"} else None,
        max_output_tokens=max_output_tokens)
    entry = {"stage": stage, "provider": completion.provider, "model": completion.model,
             "endpoint": completion.endpoint, "prompt_version": prompt_version, "system_prompt": system,
             "user_prompt": user, "raw_response": completion.text,
             "input_tokens": completion.input_tokens, "output_tokens": completion.output_tokens,
             "latency_seconds": completion.latency_seconds, "stop_reason": completion.stop_reason,
             "temperature": getattr(provider, "temperature", None),
             "seed": getattr(provider, "seed", None), "max_output_tokens": max_output_tokens,
             "attempts": 1}
    trace.append(entry)
    if completion.stop_reason in {"max_tokens", "length", "model_context_window_exceeded"}:
        raise RuntimeError(f"TRUNCATED_OUTPUT: stop_reason={completion.stop_reason}")
    if completion.stop_reason not in {"end_turn", "stop", "stop_sequence"}:
        raise RuntimeError(f"INCOMPLETE_OUTPUT: stop_reason={completion.stop_reason}")
    parsed = parse_json(completion.text)
    entry["parsed_response"] = parsed
    if flat_transport:
        restored = restore_ruleset(parsed)
        entry["restored_response"] = restored
        return restored
    return parsed


def run_strategy(strategy: str, artifact: DocumentArtifact, providers: list[Provider],
                 schema: str, *, max_output_tokens: int = 8192,
                 protocol: str = "common", trace: list[dict] | None = None,
                 prompt_version: str = "v1") -> dict:
    if strategy not in {f"E{i}" for i in range(1, 9)}:
        raise ValueError(strategy)
    if schema not in RULES:
        raise ValueError(schema)
    if strategy in {"E3", "E4", "E5", "E6"} and len(providers) != 2:
        raise ValueError(f"{strategy} requires two providers")
    if strategy in {"E1", "E2", "E7", "E8"} and len(providers) != 1:
        raise ValueError(f"{strategy} requires one provider")
    names = [p.name for p in providers]
    if strategy == "E5" and names != ["mock", "mock"] and names != ["anthropic", "openai"]:
        raise ValueError("E5 requires Claude then GPT")
    if strategy == "E6" and names != ["mock", "mock"] and names != ["openai", "anthropic"]:
        raise ValueError("E6 requires GPT then Claude")
    trace = trace if trace is not None else []
    kwargs = {"max_output_tokens": max_output_tokens, "protocol": protocol,
              "prompt_version": prompt_version}
    result: dict = {"strategy": strategy, "schema": schema, "document_id": artifact.document_id,
                    "trace": trace, "outputs": [], "issues": []}
    if strategy == "E1":
        result["outputs"] = [_call(providers[0], "joint_extraction", artifact, schema, trace, **kwargs)]
    elif strategy in {"E2", "E5", "E6"}:
        first, second = (providers[0], providers[0]) if strategy == "E2" else providers
        candidates = _call(first, "recommendation_detection", artifact, schema, trace, **kwargs)
        result["candidates"] = candidates
        rules = _call(second, "condition_extraction", artifact, schema, trace, candidates, **kwargs)
        result["outputs"] = [rules]
        if len(rules.get("rules", [])) > len(candidates.get("candidates", [])):
            result["issues"].append("More rules than candidates: verify atomization and propagation")
    elif strategy == "E7":
        passages = _call(providers[0], "recommendation_detection", artifact, schema, trace, **kwargs)
        atoms = _call(providers[0], "atomic_decomposition", artifact, schema, trace, passages, **kwargs)
        rules = _call(providers[0], "condition_extraction", artifact, schema, trace, atoms, **kwargs)
        result.update({"candidates": passages, "atoms": atoms, "outputs": [rules]})
    elif strategy in {"E3", "E4", "E8"}:
        pair = [providers[0], providers[0]] if strategy == "E8" else providers
        a = _call(pair[0], "independent_extraction", artifact, schema, trace, **kwargs)
        b = _call(pair[1], "independent_extraction", artifact, schema, trace, **kwargs)
        result["outputs"] = [a, b]
        if strategy == "E4":
            verification = _call(pair[1], "evidence_verification", artifact,
                                           schema, trace, {"A": a, "B": b}, **kwargs)
            result["verification"] = verification
            supported = set(verification.get("supported_rule_ids", []))
            accepted, review = [], []
            seen = set()
            for candidate in a.get("rules", []) + b.get("rules", []):
                rid = candidate.get("rule_id")
                if rid in seen:
                    continue
                seen.add(rid)
                refs = candidate.get("evidence", [])
                source_supported = bool(refs) and all(
                    isinstance(ref.get("page_number"), int)
                    and 1 <= ref["page_number"] <= len(artifact.pages)
                    and ref.get("source_text", "").casefold() in artifact.pages[ref["page_number"] - 1].raw_text.casefold()
                    for ref in refs)
                if rid in supported and source_supported:
                    accepted.append(candidate)
                else:
                    review.append(rid)
            result["consolidation"] = {"rules": accepted, "review_required": review,
                "status": "PROVISIONAL_SOURCE_CHECK_ONLY"}
    return result
