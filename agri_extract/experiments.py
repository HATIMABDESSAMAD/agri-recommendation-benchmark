"""Machine-readable campaign matrix and immutable run records."""
from __future__ import annotations

import csv
import json
import os
import random
import re
import uuid
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from statistics import mean, stdev

from .alignment import align, compare
from .documents import DocumentArtifact
from .evaluation import GoldAgriculturalRuleSet, check_gold, evaluate
from .extract import run_strategy
from .providers import LiveProvider, MockProvider, Provider
from .schema import CanonicalEvaluationRuleSet, parse_ruleset, to_canonical
from .synthetic import responses
from .validation import validate
from .prepare import ROOT


def matrix() -> list[dict]:
    rows = json.loads((ROOT / "configs" / "matrix.json").read_text(encoding="utf-8"))["runs"]
    if len(rows) != 18 or [r["run"] for r in rows] != [f"R{i:02d}" for i in range(1, 19)]:
        raise ValueError("invalid initial matrix")
    return rows


def plan(models: dict | None = None) -> list[dict]:
    models = models or {}
    planned = []
    for row in matrix():
        name = row["provider"]
        entry = models.get(name, {})
        configured = bool(entry.get("model") and
            isinstance(entry.get("context_window_tokens"), int) and
            entry["context_window_tokens"] > 0 and
            (name != "deepinfra" or (entry.get("endpoint") and
             entry.get("api_protocol") == "openai-compatible" and
             entry.get("auth_scheme") == "bearer" and
             entry.get("underlying_model_family") and entry.get("api_key_env"))))
        key_name = entry.get("api_key_env") or (
            "ANTHROPIC_API_KEY" if name == "anthropic" else "OPENAI_API_KEY")
        status = ("BLOCKED_MISSING_CONFIGURATION" if not configured else
                  "BLOCKED_MISSING_CREDENTIAL" if not os.environ.get(key_name) else "READY")
        planned.append({**row, "status": status})
    return planned


def _save(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _record_usage(manifest: dict, calls: list[dict], providers: list[Provider],
                  verifier: Provider | None) -> None:
    manifest["calls"] = len(calls)
    manifest["input_tokens"] = sum(x["input_tokens"] for x in calls if x["input_tokens"] is not None) or None
    manifest["output_tokens"] = sum(x["output_tokens"] for x in calls if x["output_tokens"] is not None) or None
    manifest["latency_seconds"] = sum(x["latency_seconds"] for x in calls)
    price_by_model = {(p.name, p.model): p for p in [*providers, *([verifier] if verifier else [])]}
    costs = []
    for call in calls:
        p = price_by_model.get((call["provider"], call["model"]))
        if (p is None or call["input_tokens"] is None or call["output_tokens"] is None
                or getattr(p, "input_usd_per_million", None) is None
                or getattr(p, "output_usd_per_million", None) is None):
            costs = []
            break
        costs.append((call["input_tokens"] * p.input_usd_per_million +
                      call["output_tokens"] * p.output_usd_per_million) / 1_000_000)
    manifest["cost"] = sum(costs) if costs and len(costs) == len(calls) else None


def _final_ruleset(stage: dict) -> dict | None:
    states = stage["states"]
    if "V5" in states and states["V5"].get("ruleset_after") is not None:
        return states["V5"]["ruleset_after"]
    if "V2" in states:
        return states["V2"]["ruleset"]
    return states.get("V1")


def execute(config: dict, artifact: DocumentArtifact, providers: list[Provider],
            campaign: str, *, output_root: Path | None = None, gold: GoldAgriculturalRuleSet | None = None,
            max_output_tokens: int = 8192, protocol: str = "common",
            verifier: Provider | None = None, max_repairs: int = 2) -> Path:
    if config.get("parser", artifact.parser) != artifact.parser:
        raise ValueError("configured parser does not match document artifact")
    if max_repairs < 0 or max_output_tokens <= 0:
        raise ValueError("max_repairs must be nonnegative and max_output_tokens positive")
    if gold is not None:
        audit = check_gold(gold, artifact)
        if audit["status"] != "READY":
            raise ValueError(f"gold standard is not ready: {audit['errors']}")
    output_root = output_root or ROOT / "results"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_id = f"{config.get('run', config['strategy'])}_{artifact.document_id}_{stamp}_{uuid.uuid4().hex[:8]}"
    folder = output_root / campaign / run_id
    folder.mkdir(parents=True, exist_ok=False)
    manifest = {"run_id": run_id, "config": config, "document_id": artifact.document_id,
                "source_hash": artifact.file_hash, "created_utc": stamp,
                "models": [p.model for p in providers], "protocol": protocol,
                "inference_parameters": [{"temperature": getattr(p, "temperature", None),
                    "seed": getattr(p, "seed", None)} for p in providers],
                "model_snapshots": [getattr(p, "snapshot", None) for p in providers],
                "underlying_model_families": [getattr(p, "underlying_model_family", None) for p in providers],
                "max_output_tokens": max_output_tokens, "max_repairs": max_repairs,
                "status": "RUNNING", "cost": None,
                "versions": {name: version(name) for name in ("pydantic", "pypdf", "docling", "anthropic", "openai")}}
    _save(folder / "manifest.json", manifest)
    trace: list[dict] = []
    try:
        output = run_strategy(config["strategy"], artifact, providers, config["schema"],
                              max_output_tokens=max_output_tokens, protocol=protocol, trace=trace,
                              prompt_version=config.get("prompt_version", "v1"))
        _save(folder / "raw_trace.json", output)
        # Keep the model's final JSON separate from the complete request/response trace.
        if len(output["outputs"]) == 1:
            _save(folder / "raw.json", output["outputs"][0])
        else:
            for i, raw in enumerate(output["outputs"], 1):
                _save(folder / f"raw_{i}.json", raw)
        if output.get("candidates"):
            _save(folder / "candidates.json", output["candidates"])
        if output.get("atoms"):
            _save(folder / "atoms.json", output["atoms"])
        if output.get("consolidation"):
            _save(folder / "consolidation.json", output["consolidation"])
        validated = []
        for i, raw in enumerate(output["outputs"]):
            stage = validate(raw, config["schema"], artifact, config.get("validation", "V1"),
                             verifier=verifier, max_repairs=max_repairs,
                             prompt_version=config.get("prompt_version", "v1"),
                             max_output_tokens=max_output_tokens)
            _save(folder / f"validation_{i+1}.json", stage)
            validated.append(stage)
            suffix = f"_{i+1}" if len(output["outputs"]) > 1 else ""
            if "V1" in stage["states"]:
                _save(folder / f"validated{suffix}.json", stage["states"]["V1"])
                final = _final_ruleset(stage)
                _save(folder / f"final{suffix}.json", final)
            if "V3" in stage["states"]:
                _save(folder / f"evidence_checks{suffix}.json", stage["states"]["V3"])
            if "V4" in stage["states"]:
                _save(folder / f"coverage{suffix}.json", stage["states"]["V4"])
            if "V5" in stage["states"]:
                _save(folder / f"repairs{suffix}.json", stage["states"]["V5"])
                if stage["states"]["V5"].get("repairs_applied"):
                    _save(folder / f"repaired{suffix}.json", final)
        source_pattern = re.compile(r"\b(fertili[sz]er|urea|DAP|NPS|manure|compost|nitrogen|phosphorus|potassium)\b", re.I)
        source_passages = [{"page_number": page.number, "element_id": element.element_id,
                            "source_text": element.original}
                            for page in artifact.pages for element in page.elements
                            if source_pattern.search(element.cleaned)]
        evidence = {"schema": config["schema"], "status": "SOURCE_PASSAGES_UNLINKED",
                    "rule_evidence": [], "source_passages": source_passages,
                    "note": "S1-S3 cannot link rules to source references; passages are audit candidates only. Use S4/S5 for cited evidence."}
        if config["schema"] in {"S4", "S5"}:
            evidence = {"schema": config["schema"], "status": "MODEL_SUPPLIED_UNVERIFIED",
                        "rule_evidence": [{"output_index": i+1, "rule_id": rule.get("rule_id"),
                            "evidence": rule.get("evidence", []),
                            "field_evidence": rule.get("field_evidence", {})}
                            for i, stage in enumerate(validated) if stage["status"] == "SCHEMA_VALID"
                            for rule in stage["states"]["V1"]["rules"]]}
        _save(folder / "evidence.json", evidence)
        if len(output["outputs"]) == 2:
            a = to_canonical(parse_ruleset(config["schema"], output["outputs"][0]))
            b = to_canonical(parse_ruleset(config["schema"], output["outputs"][1]))
            comparison = compare(a.rules, b.rules)
            _save(folder / "alignment.json", comparison)
            if gold:
                matched = align(a.rules, b.rules)
                matched_right = {pair["right"] for pair in matched["matches"]}
                matched_left = {pair["left"] for pair in matched["matches"]}
                def candidate_set(label: str, rules: list) -> CanonicalEvaluationRuleSet:
                    return CanonicalEvaluationRuleSet(document_id=artifact.document_id,
                        source_schema=config["schema"],
                        rules=[r.model_copy(update={"rule_id": f"{label}:{r.rule_id}"}) for r in rules])
                union = candidate_set("A", a.rules)
                union.rules.extend(r.model_copy(update={"rule_id": f"B:{r.rule_id}"})
                                   for r in b.rules if r.rule_id not in matched_right)
                intersection = candidate_set("A", [r for r in a.rules if r.rule_id in matched_left])
                _save(folder / "evaluation_multi.json", {
                    "status": "CANDIDATE_SET_COMPARISON",
                    "interpretation": "Union and intersection are analytic candidate sets, not verified consolidated recommendations.",
                    "extractor_a": evaluate(a, gold, artifact.file_hash),
                    "extractor_b": evaluate(b, gold, artifact.file_hash),
                    "unverified_union": evaluate(union, gold, artifact.file_hash),
                    "matched_intersection": evaluate(intersection, gold, artifact.file_hash)})
        if gold and len(output["outputs"]) == 1 and (
                validated[0]["status"] == "SCHEMA_VALID" or
                (validated[0]["status"] == "RAW_ONLY" and validated[0].get("raw_schema_valid"))):
            raw_canonical = to_canonical(parse_ruleset(config["schema"], output["outputs"][0]))
            final_rules = _final_ruleset(validated[0]) or output["outputs"][0]
            final_canonical = to_canonical(parse_ruleset(config["schema"], final_rules))
            _save(folder / "evaluation_raw.json", evaluate(raw_canonical, gold, artifact.file_hash))
            _save(folder / "evaluation.json", evaluate(final_canonical, gold, artifact.file_hash))
            by_stage = {}
            states = validated[0]["states"]
            for stage_name, state in states.items():
                candidate = (state.get("ruleset_after") or state.get("ruleset") if isinstance(state, dict) else None)
                if candidate is None and stage_name in {"V0", "V1"}:
                    candidate = state
                if candidate is None:
                    candidate = states.get("V2", {}).get("ruleset", states.get("V1", states["V0"]))
                try:
                    by_stage[stage_name] = evaluate(to_canonical(parse_ruleset(config["schema"], candidate)),
                                                    gold, artifact.file_hash)
                except Exception as exc:
                    by_stage[stage_name] = {"status": "SCHEMA_INVALID", "error": str(exc)[:500]}
            _save(folder / "evaluation_by_stage.json", by_stage)
        if any(v["status"] == "VERIFICATION_FAILED" for v in validated):
            manifest["status"] = "VERIFICATION_FAILED"
        elif any(v["status"] not in {"SCHEMA_VALID", "RAW_ONLY"} or
                 (v["status"] == "RAW_ONLY" and not v.get("raw_schema_valid")) for v in validated):
            manifest["status"] = "SCHEMA_INVALID"
        else:
            manifest["status"] = "SIMULATED" if all(p.name == "mock" for p in providers) else "COMPLETED"
        all_calls = output["trace"] + [call for result in validated for call in result.get("calls", [])]
        _record_usage(manifest, all_calls, providers, verifier)
    except Exception as exc:
        name = type(exc).__name__
        message = str(exc)
        context_error = any(fragment in message.lower() for fragment in (
            "context_length_exceeded", "maximum context length", "prompt is too long",
            "input is too long", "context window exceeded"))
        credit_error = any(fragment in message.lower() for fragment in (
            "credit balance is too low", "no credits remaining", "insufficient_quota"))
        manifest["status"] = ("TRUNCATED_OUTPUT" if "TRUNCATED_OUTPUT" in message else
                              "CONTEXT_LIMIT_ESTIMATE_EXCEEDED" if "CONTEXT_LIMIT_ESTIMATE_EXCEEDED" in message else
                              "CONTEXT_LIMIT_EXCEEDED" if context_error else
                              "INCOMPLETE_OUTPUT" if "INCOMPLETE_OUTPUT" in message else
                              "JSON_INVALID" if name == "JSONDecodeError" else
                              "BLOCKED_MISSING_CONFIGURATION" if "BLOCKED_MISSING_CONFIGURATION" in message else
                              "INSUFFICIENT_CREDITS" if credit_error else
                              "RATE_LIMITED" if "RateLimit" in name else
                              "TIMED_OUT" if "Timeout" in name else "FAILED")
        manifest["error"] = f"{name}: {message}"
        _save(folder / "partial_trace.json", trace)
        _record_usage(manifest, trace, providers, verifier)
    _save(folder / "cost.json", {"currency": "USD", "estimated_cost": manifest.get("cost"),
        "input_tokens": manifest.get("input_tokens"), "output_tokens": manifest.get("output_tokens"),
        "calls": manifest.get("calls"),
        "status": "UNKNOWN" if manifest.get("cost") is None else "ESTIMATED_FROM_CONFIGURED_RATES"})
    (folder / "run.log").write_text(
        f"{stamp} run={run_id} document={artifact.document_id} strategy={config['strategy']} "
        f"schema={config['schema']} validation={config.get('validation', 'V1')} "
        f"status={manifest['status']} calls={manifest.get('calls', 0)}\n"
        + (f"error={manifest['error']}\n" if manifest.get("error") else ""), encoding="utf-8")
    (folder / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return folder


def simulate_matrix(campaign: str, output_root: Path | None = None, resume: bool = False) -> list[Path]:
    from .synthetic import artifact
    art = artifact(parser="P4")
    paths = []
    for row in matrix():
        if resume:
            existing = [p for p in ((output_root or ROOT / "results") / campaign).glob("*/manifest.json")
                        if (d := json.loads(p.read_text(encoding="utf-8"))).get("config", {}).get("run") == row["run"]
                        and d.get("status") == "SIMULATED"]
            if existing:
                paths.append(existing[0].parent)
                continue
        mock = MockProvider(responses(row["strategy"], row["schema"]), model=f"mock-{row['provider']}")
        paths.append(execute(row, art, [mock], campaign, output_root=output_root))
    return paths


def repeatability(rows: list[dict], bootstrap_samples: int = 1000) -> list[dict]:
    """Summarize repeated scored runs; never infer a score from unscored runs."""
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (row["document_id"], row["model"], row["strategy"], row["schema"],
               row["parser"], row["validation"])
        groups.setdefault(key, []).append(row)
    result = []
    for key, group in sorted(groups.items(), key=lambda item: str(item[0])):
        scores = [r["recommendation_f1"] for r in group if r["recommendation_f1"] is not None]
        interval = None
        if len(scores) >= 2:
            rng = random.Random(0)
            samples = sorted(mean(rng.choices(scores, k=len(scores))) for _ in range(bootstrap_samples))
            interval = [samples[int(0.025 * bootstrap_samples)],
                        samples[min(bootstrap_samples - 1, int(0.975 * bootstrap_samples))]]
        result.append({"configuration": dict(zip(
            ("document_id", "model", "strategy", "schema", "parser", "validation"), key)),
            "run_count": len(group), "scored_run_count": len(scores),
            "mean_recommendation_f1": mean(scores) if scores else None,
            "sample_stdev_recommendation_f1": stdev(scores) if len(scores) >= 2 else None,
            "bootstrap_95_ci_recommendation_f1": interval,
            "interpretation": "descriptive within-document repeatability; not evidence of cross-document generalization"})
    return result


def report(campaign: str, output_root: Path | None = None) -> tuple[Path, Path]:
    base = (output_root or ROOT / "results") / campaign
    rows = []
    for file in base.glob("*/manifest.json"):
        d = json.loads(file.read_text(encoding="utf-8"))
        rows.append({"run_id": d["run_id"], "document_id": d["document_id"],
                     "model": ",".join(d.get("models", [])),
                     "strategy": d["config"]["strategy"], "schema": d["config"]["schema"],
                     "parser": d["config"].get("parser"),
                     "validation": d["config"].get("validation"), "status": d["status"],
                     "cost": d.get("cost"), "latency_seconds": d.get("latency_seconds"),
                     "recommendation_precision": None, "recommendation_recall": None,
                     "recommendation_f1": None, "condition_f1": None,
                     "exception_f1": None, "numeric_accuracy": None,
                     "schema_validity": None})
        validations = list(file.parent.glob("validation_*.json"))
        if validations:
            v = json.loads(validations[0].read_text(encoding="utf-8"))
            rows[-1]["schema_validity"] = 1.0 if (
                v["status"] == "SCHEMA_VALID" or
                (v["status"] == "RAW_ONLY" and v.get("raw_schema_valid"))) else 0.0
        evaluation = file.parent / "evaluation.json"
        if evaluation.exists():
            e = json.loads(evaluation.read_text(encoding="utf-8"))
            if e["status"] == "EVALUATED":
                m = e["metrics"]
                rows[-1].update(recommendation_precision=m["recommendation"]["precision"],
                    recommendation_recall=m["recommendation"]["recall"],
                    recommendation_f1=m["recommendation"]["f1"],
                    condition_f1=m["condition"]["f1"], exception_f1=m["exception"]["f1"],
                    numeric_accuracy=m["numeric_accuracy"])
    rows.sort(key=lambda x: x["run_id"])
    json_path, csv_path = base / "report.json", base / "report.csv"
    json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["run_id"])
        writer.writeheader(); writer.writerows(rows)
    tables = {
        "table_1_global": rows,
        "table_2_e1_vs_e2": [r for r in rows if r["strategy"] in {"E1", "E2"}],
        "table_3_schemas": [r for r in rows if r["schema"] in {f"S{i}" for i in range(1, 6)}],
        "table_4_validation": [
            {**r, "stage_evaluation": json.loads((base / r["run_id"] /
                "evaluation_by_stage.json").read_text(encoding="utf-8"))
                if (base / r["run_id"] / "evaluation_by_stage.json").exists() else None}
            for r in rows],
        "table_5_multi_llm": [
            {**r, "candidate_set_evaluation": json.loads((base / r["run_id"] /
                "evaluation_multi.json").read_text(encoding="utf-8"))
                if (base / r["run_id"] / "evaluation_multi.json").exists() else None}
            for r in rows if r["strategy"] in {"E3", "E4", "E8"}],
        "table_6_parsing": rows,
        "table_7_errors": [],
        "repeatability": repeatability(rows),
        "notice": "Null means not evaluated or unknown; synthetic runs are never scientific scores.",
    }
    for file in base.glob("*/validation_*.json"):
        data = json.loads(file.read_text(encoding="utf-8"))
        for error in data.get("errors", []):
            tables["table_7_errors"].append({"run_id": file.parent.name, "type": "schema", "detail": error})
    for file in base.glob("*/evidence_checks*.json"):
        data = json.loads(file.read_text(encoding="utf-8"))
        for check in data.get("checks", []):
            if check.get("status") in {"CONTRADICTED", "INSUFFICIENT_EVIDENCE", "AMBIGUOUS"}:
                tables["table_7_errors"].append({"run_id": file.parent.name,
                    "type": "evidence", "detail": check})
    for file in base.glob("*/coverage*.json"):
        data = json.loads(file.read_text(encoding="utf-8"))
        for candidate in data.get("potentially_unmatched_passages", []):
            tables["table_7_errors"].append({"run_id": file.parent.name,
                "type": "unlinked_coverage_candidate", "detail": candidate})
    for file in base.glob("*/evaluation.json"):
        data = json.loads(file.read_text(encoding="utf-8"))
        if data.get("status") == "EVALUATED":
            match = data["matching"]
            for rule_id in match["left_only"]:
                tables["table_7_errors"].append({"run_id": file.parent.name,
                    "type": "unmatched_prediction", "detail": rule_id})
            for rule_id in match["right_only"]:
                tables["table_7_errors"].append({"run_id": file.parent.name,
                    "type": "omission", "detail": rule_id})
            for pair in match["borderline"]:
                tables["table_7_errors"].append({"run_id": file.parent.name,
                    "type": "borderline_match", "detail": pair})
            for pair in match.get("contested", []):
                tables["table_7_errors"].append({"run_id": file.parent.name,
                    "type": "contested_match", "detail": pair})
    (base / "report_tables.json").write_text(json.dumps(tables, ensure_ascii=False, indent=2), encoding="utf-8")
    (base / "repeatability.json").write_text(json.dumps(tables["repeatability"], ensure_ascii=False, indent=2), encoding="utf-8")
    return json_path, csv_path


def compare_campaigns(first: str, second: str, output_root: Path | None = None) -> list[dict]:
    root = output_root or ROOT / "results"
    def load(campaign: str):
        path = root / campaign / "report.json"
        if not path.exists():
            report(campaign, output_root=root)
        return json.loads(path.read_text(encoding="utf-8"))
    left, right = load(first), load(second)
    def key(row):
        return (row["document_id"], row["strategy"], row["schema"], row["model"], row["parser"], row["validation"])
    a = {key(row): row for row in left}
    b = {key(row): row for row in right}
    output = []
    for k in sorted(set(a) | set(b), key=str):
        x, y = a.get(k), b.get(k)
        sx = x.get("recommendation_f1") if x else None
        sy = y.get("recommendation_f1") if y else None
        output.append({"configuration": k, "first_status": x.get("status") if x else None,
                       "second_status": y.get("status") if y else None,
                       "first_f1": sx, "second_f1": sy,
                       "f1_difference": sy - sx if sx is not None and sy is not None else None})
    return output
