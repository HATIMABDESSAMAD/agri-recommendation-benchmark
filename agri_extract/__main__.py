"""Research CLI. No paid call runs without --live."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .documents import DocumentArtifact, convert, save_artifact
from .evaluation import GoldAgriculturalRuleSet, check_gold, evaluate, gold_template
from .experiments import ROOT, compare_campaigns, execute, matrix, plan, report, simulate_matrix
from .prepare import DOCUMENTS
from .providers import LiveProvider
from .providers import MockProvider
from .registry import register
from .schema import parse_ruleset, to_canonical
from .synthetic import artifact as synthetic_artifact, responses as synthetic_responses
from .validation import validate


def main():
    parser = argparse.ArgumentParser(prog="python -m agri_extract")
    sub = parser.add_subparsers(dest="cmd", required=True)
    reg = sub.add_parser("register")
    reg.add_argument("--document", choices=DOCUMENTS)
    parse = sub.add_parser("parse")
    parse.add_argument("--document", choices=DOCUMENTS, required=True)
    parse.add_argument("--pipeline", choices=[f"P{i}" for i in range(5)], required=True)
    planning = sub.add_parser("plan")
    planning.add_argument("--models", type=Path)
    sim = sub.add_parser("simulate-matrix")
    sim.add_argument("--campaign", default="synthetic_demo")
    sim.add_argument("--resume", action="store_true")
    sim_custom = sub.add_parser("simulate-configs")
    sim_custom.add_argument("--configs", type=Path, required=True)
    sim_custom.add_argument("--campaign", required=True)
    run = sub.add_parser("run")
    run.add_argument("--artifact", type=Path, required=True)
    run.add_argument("--strategy", choices=[f"E{i}" for i in range(1, 9)], required=True)
    run.add_argument("--schema", choices=[f"S{i}" for i in range(1, 6)], required=True)
    run.add_argument("--validation", choices=[f"V{i}" for i in range(6)], default="V1")
    run.add_argument("--provider", choices=["anthropic", "openai", "deepinfra"], nargs="+", required=True)
    run.add_argument("--models", type=Path, required=True)
    run.add_argument("--gold", type=Path)
    run.add_argument("--campaign", required=True)
    run.add_argument("--protocol", choices=["common", "native"], default="common")
    run.add_argument("--prompt-version", default="v1")
    run.add_argument("--max-output-tokens", type=int, default=8192)
    run.add_argument("--max-requests", type=int, default=5)
    run.add_argument("--max-repairs", type=int, default=2)
    run.add_argument("--verification-mode", choices=["V-Same", "V-Cross"])
    run.add_argument("--verifier-provider", choices=["anthropic", "openai", "deepinfra"])
    run.add_argument("--live", action="store_true")
    run.add_argument("--source-rights-confirmed", action="store_true")
    batch = sub.add_parser("run-matrix")
    batch.add_argument("--artifact", type=Path, required=True)
    batch.add_argument("--models", type=Path, required=True)
    batch.add_argument("--gold", type=Path)
    batch.add_argument("--campaign", required=True)
    batch.add_argument("--run-id", choices=[f"R{i:02d}" for i in range(1, 19)])
    batch.add_argument("--max-requests", type=int, required=True)
    batch.add_argument("--max-repairs", type=int, default=2)
    batch.add_argument("--max-output-tokens", type=int, default=8192)
    batch.add_argument("--protocol", choices=["common", "native"], default="common")
    batch.add_argument("--live", action="store_true")
    batch.add_argument("--source-rights-confirmed", action="store_true")
    custom = sub.add_parser("run-batch")
    custom.add_argument("--artifact", type=Path, required=True)
    custom.add_argument("--configs", type=Path, required=True)
    custom.add_argument("--models", type=Path, required=True)
    custom.add_argument("--gold", type=Path)
    custom.add_argument("--campaign", required=True)
    custom.add_argument("--max-requests", type=int, required=True)
    custom.add_argument("--max-repairs", type=int, default=2)
    custom.add_argument("--max-output-tokens", type=int, default=8192)
    custom.add_argument("--protocol", choices=["common", "native"], default="common")
    custom.add_argument("--resume", action="store_true")
    custom.add_argument("--live", action="store_true")
    custom.add_argument("--source-rights-confirmed", action="store_true")
    gold = sub.add_parser("gold-template")
    gold.add_argument("--artifact", type=Path, required=True)
    gold.add_argument("--split", choices=["development", "validation", "test", "unassigned"], default="unassigned")
    gold_check = sub.add_parser("gold-check")
    gold_check.add_argument("--artifact", type=Path, required=True)
    gold_check.add_argument("--gold", type=Path, required=True)
    val = sub.add_parser("validate")
    val.add_argument("--artifact", type=Path, required=True)
    val.add_argument("--rules", type=Path, required=True)
    val.add_argument("--schema", choices=[f"S{i}" for i in range(1, 6)], required=True)
    val.add_argument("--level", choices=[f"V{i}" for i in range(6)], default="V1")
    ev = sub.add_parser("evaluate")
    ev.add_argument("--artifact", type=Path, required=True)
    ev.add_argument("--rules", type=Path, required=True)
    ev.add_argument("--schema", choices=[f"S{i}" for i in range(1, 6)], required=True)
    ev.add_argument("--gold", type=Path, required=True)
    rep = sub.add_parser("report")
    rep.add_argument("--campaign", required=True)
    comp = sub.add_parser("compare-campaigns")
    comp.add_argument("first")
    comp.add_argument("second")
    args = parser.parse_args()

    if args.cmd == "register":
        ids = [args.document] if args.document else DOCUMENTS
        out = [register(i).model_dump(mode="json") for i in ids]
        path = ROOT / "artifacts" / "document_registry.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        print(path)
    elif args.cmd == "parse":
        art = convert(args.document, args.pipeline)
        path = save_artifact(art, ROOT / "artifacts" / "documents")
        print(f"{path} | {len(art.pages)} pages | warnings={sum(len(p.warnings) for p in art.pages)}")
    elif args.cmd == "plan":
        models = json.loads(args.models.read_text(encoding="utf-8")) if args.models else None
        print(json.dumps(plan(models), indent=2, ensure_ascii=False))
    elif args.cmd == "simulate-matrix":
        paths = simulate_matrix(args.campaign, resume=args.resume)
        print(f"{len(paths)} synthetic runs; no API calls")
        print(*paths, sep="\n")
    elif args.cmd == "simulate-configs":
        rows = json.loads(args.configs.read_text(encoding="utf-8"))["runs"]
        for row in rows:
            art = synthetic_artifact(parser=row.get("parser", "P0"))
            mock = MockProvider(synthetic_responses(row["strategy"], row["schema"]),
                                model="mock-synthetic-v1")
            pair = [mock, mock] if row["strategy"] in {"E3", "E4", "E5", "E6"} else [mock]
            verifier = (MockProvider([json.dumps({"checks": []})],
                model="mock-synthetic-v1" if row.get("verification_mode") == "V-Same" else
                      "mock-synthetic-verifier-v2") if row.get("verification_mode") else None)
            print(execute(row, art, pair, args.campaign, verifier=verifier))
    elif args.cmd == "run":
        if not args.live:
            raise PermissionError("Real providers require --live; no paid call was made")
        art = DocumentArtifact.model_validate_json(args.artifact.read_text(encoding="utf-8"))
        gold_data = None
        if args.gold:
            gold_data = GoldAgriculturalRuleSet.model_validate_json(args.gold.read_text(encoding="utf-8"))
            audit = check_gold(gold_data, art)
            if audit["status"] != "READY":
                raise ValueError(f"gold-check failed before API calls: {audit['errors']}")
        if art.document_id in DOCUMENTS:
            record = register(art.document_id)
            if record.file_hash != art.file_hash or record.license_verification_status != "STATED_IN_PDF":
                raise PermissionError("source hash or documented license check failed")
        elif not args.source_rights_confirmed:
            raise PermissionError("unregistered source requires --source-rights-confirmed")
        models = json.loads(args.models.read_text(encoding="utf-8"))
        providers = []
        for name in args.provider:
            entry = models.get(name, {})
            if not entry.get("model"):
                raise ValueError(f"BLOCKED_MISSING_CONFIGURATION: {name}.model")
            if not isinstance(entry.get("context_window_tokens"), int) or entry["context_window_tokens"] <= 0:
                raise ValueError(f"BLOCKED_MISSING_CONFIGURATION: {name}.context_window_tokens")
            if args.protocol == "native" and entry.get("supports_json_schema") is not True:
                raise ValueError(f"BLOCKED_MISSING_CONFIGURATION: {name}.supports_json_schema must be confirmed")
            providers.append(LiveProvider(name, entry["model"], allow_live=True,
                endpoint=entry.get("endpoint"), api_key_env=entry.get("api_key_env"),
                protocol=args.protocol, snapshot=entry.get("snapshot"),
                underlying_model_family=entry.get("underlying_model_family"),
                input_usd_per_million=entry.get("input_usd_per_million"),
                output_usd_per_million=entry.get("output_usd_per_million"),
                api_protocol=entry.get("api_protocol"), auth_scheme=entry.get("auth_scheme"),
                context_window_tokens=entry.get("context_window_tokens"),
                temperature=entry.get("temperature"), seed=entry.get("seed")))
        config = {"strategy": args.strategy, "schema": args.schema, "validation": args.validation,
                  "parser": art.parser, "providers": args.provider,
                  "prompt_version": args.prompt_version}
        planned_calls = {"E1":1,"E2":2,"E3":2,"E4":3,"E5":2,"E6":2,"E7":3,"E8":2}[args.strategy]
        if args.max_repairs < 0:
            raise ValueError("max-repairs must be nonnegative")
        if args.verification_mode and args.validation in {"V3", "V4", "V5"}:
            planned_calls += (2 if args.strategy in {"E3", "E4", "E8"} else 1) * (
                1 + args.max_repairs if args.validation == "V5" else 1)
        if args.max_requests < planned_calls:
            raise ValueError(f"max-requests {args.max_requests} below planned {planned_calls}; no calls made")
        verifier = None
        if args.verification_mode == "V-Same":
            verifier = providers[0]
        elif args.verification_mode == "V-Cross":
            name = args.verifier_provider
            if not name or name == args.provider[0]:
                raise ValueError("V-Cross requires a different --verifier-provider")
            entry = models.get(name, {})
            if not entry.get("model"):
                raise ValueError(f"BLOCKED_MISSING_CONFIGURATION: {name}.model")
            if not isinstance(entry.get("context_window_tokens"), int) or entry["context_window_tokens"] <= 0:
                raise ValueError(f"BLOCKED_MISSING_CONFIGURATION: {name}.context_window_tokens")
            verifier = LiveProvider(name, entry["model"], allow_live=True,
                endpoint=entry.get("endpoint"), api_key_env=entry.get("api_key_env"),
                protocol=args.protocol, snapshot=entry.get("snapshot"),
                underlying_model_family=entry.get("underlying_model_family"),
                input_usd_per_million=entry.get("input_usd_per_million"),
                output_usd_per_million=entry.get("output_usd_per_million"),
                api_protocol=entry.get("api_protocol"), auth_scheme=entry.get("auth_scheme"),
                context_window_tokens=entry.get("context_window_tokens"),
                temperature=entry.get("temperature"), seed=entry.get("seed"))
        config["verification_mode"] = args.verification_mode
        print(execute(config, art, providers, args.campaign,
                      max_output_tokens=args.max_output_tokens, protocol=args.protocol,
                      verifier=verifier, max_repairs=args.max_repairs, gold=gold_data))
    elif args.cmd in {"run-matrix", "run-batch"}:
        if not args.live:
            raise PermissionError("Batch API calls require explicit --live; no paid call was made")
        art = DocumentArtifact.model_validate_json(args.artifact.read_text(encoding="utf-8"))
        gold_data = None
        if args.gold:
            gold_data = GoldAgriculturalRuleSet.model_validate_json(args.gold.read_text(encoding="utf-8"))
            audit = check_gold(gold_data, art)
            if audit["status"] != "READY":
                raise ValueError(f"gold-check failed before API calls: {audit['errors']}")
        if art.document_id in DOCUMENTS:
            record = register(art.document_id)
            if record.file_hash != art.file_hash or record.license_verification_status != "STATED_IN_PDF":
                raise PermissionError("source hash or documented license check failed")
        elif not args.source_rights_confirmed:
            raise PermissionError("unregistered source requires --source-rights-confirmed")
        if args.cmd == "run-matrix" and art.parser != "P4":
            raise ValueError("Initial R01–R18 matrix requires P4 artifact")
        models = json.loads(args.models.read_text(encoding="utf-8"))
        rows = ([r for r in matrix() if not args.run_id or r["run"] == args.run_id]
                if args.cmd == "run-matrix" else json.loads(args.configs.read_text(encoding="utf-8"))["runs"])
        if args.max_repairs < 0:
            raise ValueError("max-repairs must be nonnegative")
        def names(row):
            return row.get("providers") or [row["provider"]]
        def has_credential(name):
            entry = models.get(name, {})
            key_name = entry.get("api_key_env") or (
                "ANTHROPIC_API_KEY" if name == "anthropic" else "OPENAI_API_KEY")
            return bool(os.environ.get(key_name))
        def calls(row):
            base = {"E1":1,"E2":2,"E3":2,"E4":3,"E5":2,"E6":2,"E7":3,"E8":2}[row["strategy"]]
            level = row.get("validation", "V1")
            if row.get("verification_mode") and level in {"V3", "V4", "V5"}:
                outputs = 2 if row["strategy"] in {"E3", "E4", "E8"} else 1
                base += outputs * (1 + args.max_repairs if level == "V5" else 1)
            return base
        needed = sum(calls(r) for r in rows if all(models.get(n, {}).get("model")
            and isinstance(models.get(n, {}).get("context_window_tokens"), int)
            and models[n]["context_window_tokens"] > 0 and has_credential(n)
            for n in names(r)))
        if args.max_requests < needed:
            raise ValueError(f"max-requests {args.max_requests} below planned {needed}; no calls made")
        blocked = []
        for row in rows:
            selected_protocol = row.get("protocol", args.protocol)
            if row.get("parser", art.parser) != art.parser:
                raise ValueError(f"{row.get('run')} parser mismatch with artifact")
            if args.cmd == "run-batch" and args.resume:
                previous = [p for p in (ROOT / "results" / args.campaign).glob("*/manifest.json")
                            if (d := json.loads(p.read_text(encoding="utf-8"))).get("config") == row
                            and d.get("source_hash") == art.file_hash and d.get("status") == "COMPLETED"]
                if previous:
                    print(f"{row.get('run')} RESUMED_EXISTING {previous[0].parent}")
                    continue
            if any(not models.get(n, {}).get("model") or
                   not isinstance(models.get(n, {}).get("context_window_tokens"), int) or
                   models[n]["context_window_tokens"] <= 0 or
                   (n == "deepinfra" and (not models.get(n, {}).get("endpoint") or
                    models.get(n, {}).get("api_protocol") != "openai-compatible" or
                    models.get(n, {}).get("auth_scheme") != "bearer" or
                    not models.get(n, {}).get("underlying_model_family"))) for n in names(row)):
                blocked.append({"config": row, "status": "BLOCKED_MISSING_CONFIGURATION"})
                print(f"{row.get('run')} BLOCKED_MISSING_CONFIGURATION")
                continue
            if selected_protocol == "native" and any(models.get(n, {}).get("supports_json_schema") is not True for n in names(row)):
                blocked.append({"config": row, "status": "BLOCKED_MODEL_CAPABILITY_UNCONFIRMED"})
                print(f"{row.get('run')} BLOCKED_MODEL_CAPABILITY_UNCONFIRMED")
                continue
            if any(not has_credential(name) for name in names(row)):
                blocked.append({"config": row, "status": "BLOCKED_MISSING_CREDENTIAL"})
                print(f"{row.get('run')} BLOCKED_MISSING_CREDENTIAL")
                continue
            providers = []
            for name in names(row):
                entry = models[name]
                providers.append(LiveProvider(name, entry["model"], allow_live=True,
                    endpoint=entry.get("endpoint"), api_key_env=entry.get("api_key_env"),
                    protocol=selected_protocol,
                    snapshot=entry.get("snapshot"),
                    underlying_model_family=entry.get("underlying_model_family"),
                    input_usd_per_million=entry.get("input_usd_per_million"),
                    output_usd_per_million=entry.get("output_usd_per_million"),
                    api_protocol=entry.get("api_protocol"), auth_scheme=entry.get("auth_scheme"),
                    context_window_tokens=entry.get("context_window_tokens"),
                    temperature=entry.get("temperature"), seed=entry.get("seed")))
            verifier = None
            mode = row.get("verification_mode")
            if mode == "V-Same":
                verifier = providers[0]
            elif mode == "V-Cross":
                name = row.get("verifier_provider")
                if not name or name == names(row)[0]:
                    raise ValueError("V-Cross requires a distinct verifier_provider")
                entry = models.get(name, {})
                if not entry.get("model") or not isinstance(entry.get("context_window_tokens"), int):
                    blocked.append({"config": row, "status": "BLOCKED_MISSING_CONFIGURATION"})
                    continue
                if not has_credential(name):
                    blocked.append({"config": row, "status": "BLOCKED_MISSING_CREDENTIAL"})
                    print(f"{row.get('run')} BLOCKED_MISSING_CREDENTIAL")
                    continue
                verifier = LiveProvider(name, entry["model"], allow_live=True,
                    endpoint=entry.get("endpoint"), api_key_env=entry.get("api_key_env"),
                    protocol=selected_protocol,
                    snapshot=entry.get("snapshot"),
                    underlying_model_family=entry.get("underlying_model_family"),
                    input_usd_per_million=entry.get("input_usd_per_million"),
                    output_usd_per_million=entry.get("output_usd_per_million"),
                    api_protocol=entry.get("api_protocol"), auth_scheme=entry.get("auth_scheme"),
                    context_window_tokens=entry.get("context_window_tokens"),
                    temperature=entry.get("temperature"), seed=entry.get("seed"))
            print(execute(row, art, providers, args.campaign,
                          max_output_tokens=args.max_output_tokens, verifier=verifier,
                          protocol=selected_protocol, max_repairs=args.max_repairs,
                          gold=gold_data))
        if blocked:
            path = ROOT / "results" / args.campaign / "blocked_configurations.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(blocked, ensure_ascii=False, indent=2), encoding="utf-8")
            print(path)
    elif args.cmd == "gold-template":
        art = DocumentArtifact.model_validate_json(args.artifact.read_text(encoding="utf-8"))
        template = gold_template(art.document_id, art.file_hash, len(art.pages), args.split)
        path = ROOT / "artifacts" / "gold" / f"{art.document_id}_{art.file_hash[:8]}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(path)
        path.write_text(template.model_dump_json(indent=2), encoding="utf-8")
        print(path)
    elif args.cmd == "gold-check":
        art = DocumentArtifact.model_validate_json(args.artifact.read_text(encoding="utf-8"))
        gold_data = GoldAgriculturalRuleSet.model_validate_json(args.gold.read_text(encoding="utf-8"))
        print(json.dumps(check_gold(gold_data, art), indent=2, ensure_ascii=False))
    elif args.cmd in {"validate", "evaluate"}:
        art = DocumentArtifact.model_validate_json(args.artifact.read_text(encoding="utf-8"))
        raw = json.loads(args.rules.read_text(encoding="utf-8"))
        if args.cmd == "validate":
            print(json.dumps(validate(raw, args.schema, art, args.level), indent=2, ensure_ascii=False))
        else:
            pred = to_canonical(parse_ruleset(args.schema, raw))
            gold_data = GoldAgriculturalRuleSet.model_validate_json(args.gold.read_text(encoding="utf-8"))
            audit = check_gold(gold_data, art)
            result = evaluate(pred, gold_data, art.file_hash) if audit["status"] == "READY" else {
                "status": "NOT_EVALUATED", "gold_check": audit, "metrics": None}
            print(json.dumps(result, indent=2, ensure_ascii=False))
    elif args.cmd == "report":
        print(*report(args.campaign), sep="\n")
    elif args.cmd == "compare-campaigns":
        print(json.dumps(compare_campaigns(args.first, args.second), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
