"""Label-blind V5 benchmark for three DeepInfra models and three fixed methods."""
from __future__ import annotations

import argparse
import getpass
import hashlib
import importlib.util
import json
import os
from pathlib import Path

from agri_extract.documents import DocumentArtifact
from agri_extract.s5_matcher import decide
from agri_extract.s5_query import extraction_messages, parse_s5_query
from agri_extract.s5_reasoner import SYSTEM_PROMPT, messages as reasoner_messages, parse_decision
from agri_extract.s5_working_copy import validated_copy

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/annotation_v5_multimodel"
MODEL_IDS = ("Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-27B",
             "Qwen/Qwen3-235B-A22B-Instruct-2507")
CAP_USD = 3.0

spec = importlib.util.spec_from_file_location("baseline_support_archive", ROOT / "agri_extract/baseline_support.py")
support = importlib.util.module_from_spec(spec)
spec.loader.exec_module(support)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def slug(model: str) -> str:
    return model.replace("/", "__")


def complete(client, ledger: dict, model: dict, qid: str, method: str,
             messages: list[dict], max_tokens: int, reserve_input: int,
             destination: Path) -> dict:
    if destination.exists():
        reply = json.loads(destination.read_text(encoding="utf-8"))
        if reply.get("request_sha256") != hashlib.sha256(json.dumps(messages, ensure_ascii=False).encode()).hexdigest():
            raise ValueError(f"Saved request differs for {model['id']} {qid} {method}")
        return reply
    estimated_input = (sum(len(item["content"].encode("utf-8")) for item in messages) + 2) // 3
    if estimated_input + max_tokens > model["context_tokens"]:
        raise ValueError(f"Estimated context overflow for {model['id']} {qid} {method}")
    reserve = 1.5 * (reserve_input * model["input_price"] +
                     max_tokens * model["output_price"]) / 1_000_000
    if ledger["reserved_usd"] + reserve > CAP_USD:
        raise RuntimeError(f"Budget reservation cap reached before {model['id']} {qid} {method}")
    ledger["reserved_usd"] += reserve
    save(OUT / "budget_ledger.json", ledger)
    reply = client.complete(model["id"], messages, max_tokens)
    reply.update({"model": model["id"], "question_id": qid, "method": method,
                  "request_sha256": hashlib.sha256(json.dumps(messages, ensure_ascii=False).encode()).hexdigest(),
                  "reserved_usd": reserve, "cost_usd": support.cost(reply, model)})
    ledger["observed_usd"] += reply["cost_usd"]
    save(destination, reply)
    save(OUT / "budget_ledger.json", ledger)
    print(f"{slug(model['id'])} {qid} {method}: saved; total USD {ledger['observed_usd']:.6f}", flush=True)
    return reply


def text(reply: dict) -> str:
    if reply.get("error") or not reply.get("raw", {}).get("choices"):
        raise ValueError("API error or missing completion")
    choice = reply["raw"]["choices"][0]
    if choice.get("finish_reason") == "length":
        raise ValueError("Truncated output")
    return choice["message"]["content"] or ""


def decision(path: Path, qid: str, parsed: dict | None, error: Exception | None = None) -> None:
    if error is None:
        save(path, {"question_id": qid, "decision": parsed})
    else:
        save(path, {"question_id": qid,
                    "error": {"type": type(error).__name__, "message": str(error)[:500]}})


def run(check_only: bool) -> None:
    blind_path = ROOT / "data/private/annotation_v5_blind.jsonl"
    rules_path = ROOT / "data/private/verified_s5_rules.json"
    artifact_path = ROOT / "artifacts/documents/garlic_P4_tables.json"
    prompt_path = ROOT / "configs/s5_query_system_prompt_v1.txt"
    blind = [json.loads(line) for line in blind_path.read_text(encoding="utf-8").splitlines() if line]
    if len(blind) != 15 or len({q["id"] for q in blind}) != 15:
        raise ValueError("Expected 15 unique blind questions")
    rules, encodings = validated_copy(json.loads(rules_path.read_text(encoding="utf-8")))
    document = DocumentArtifact.model_validate_json(artifact_path.read_text(encoding="utf-8")).llm_text()
    old_manifest = json.loads((ROOT / "results/direct_baseline_archive/manifest.json").read_text(encoding="utf-8"))
    if old_manifest["document_text_sha256"] != hashlib.sha256(document.encode()).hexdigest():
        raise ValueError("Prepared PDF text differs from prior benchmark")
    if old_manifest["prompt_sha256"]["direct"] != hashlib.sha256(support.DIRECT_SYSTEM.encode()).hexdigest():
        raise ValueError("Direct system prompt differs from prior benchmark")
    if old_manifest["api_parameters"] != support.API_PARAMS:
        raise ValueError("API request parameters differ from prior benchmark")
    prompt = prompt_path.read_text(encoding="utf-8")
    catalog = {m["id"]: m for m in json.loads((ROOT / "data/private/study_code/configs__deepinfra_models_snapshot.json").read_text(encoding="utf-8"))["models"]}
    if any(model_id not in catalog for model_id in MODEL_IDS):
        raise ValueError("Selected model missing from local catalog snapshot")
    maximum_direct_input = max((len(document.encode()) + len(q["question"].encode()) + len(support.DIRECT_SYSTEM.encode()) + 100) // 3 for q in blind)
    maximum_reasoner_input = max((len(json.dumps({"question": q["question"], "s5_ruleset": rules}, ensure_ascii=False).encode()) + 10000) // 3 for q in blind)
    for model_id in MODEL_IDS:
        window = catalog[model_id]["context_tokens"]
        if maximum_direct_input + 768 > window or maximum_reasoner_input + 2048 > window:
            raise ValueError(f"Full input exceeds context estimate for {model_id}")
    if check_only:
        print(json.dumps({"models": MODEL_IDS, "questions": len(blind),
                          "maximum_direct_input_estimate": maximum_direct_input,
                          "maximum_reasoner_input_estimate": maximum_reasoner_input,
                          "context_tokens": {m: catalog[m]["context_tokens"] for m in MODEL_IDS},
                          "budget_cap_usd": CAP_USD, "api_calls": 0}, indent=2))
        return
    save(OUT / "run_manifest.json", {"models": MODEL_IDS,
        "blind_sha256": sha(blind_path), "rules_sha256": sha(rules_path),
        "pdf_artifact_sha256": sha(artifact_path),
        "document_text_sha256": hashlib.sha256(document.encode()).hexdigest(),
        "direct_system_prompt_sha256": hashlib.sha256(support.DIRECT_SYSTEM.encode()).hexdigest(),
        "direct_user_format": "JSON object with pdf_text and question",
        "s5query_prompt_sha256": sha(prompt_path),
        "s5reasoner_system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(),
        "api_parameters": support.API_PARAMS, "output_caps": {"direct": 768, "s5query": 4096, "s5_reasoner": 2048},
        "s5_encoding_audit": encodings, "gold_opened_by_runner": False,
        "budget_cap_usd": CAP_USD})
    ledger_path = OUT / "budget_ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.exists() else {
        "cap_usd": CAP_USD, "reserved_usd": 0.0, "observed_usd": 0.0}
    if ledger["cap_usd"] != CAP_USD:
        raise ValueError("Saved budget cap differs")
    key = os.getenv("DEEPINFRA_API_KEY") or getpass.getpass("DeepInfra key (hidden): ")
    client = support.DeepInfra(key)
    for model_id in MODEL_IDS:
        model = catalog[model_id]
        folder = OUT / slug(model_id)
        compat = [{"role": "system", "content": "Return JSON only."},
                  {"role": "user", "content": "Return exactly {\"ok\":true}."}]
        compat_reply = complete(client, ledger, model, "preflight", "compat", compat, 30, 100,
                                folder / "preflight_compat.json")
        try:
            if json.loads(text(compat_reply)).get("ok") is not True:
                raise ValueError("Compatibility response did not return ok=true")
        except Exception as exc:
            save(folder / "model_blocked.json", {"error": str(exc)[:500]})
            print(f"{slug(model_id)} blocked by compatibility test", flush=True)
            continue
        for q in blind:
            qid = q["id"]
            user = json.dumps({"pdf_text": document, "question": q["question"]}, ensure_ascii=False)
            messages = [{"role": "system", "content": support.DIRECT_SYSTEM},
                        {"role": "user", "content": user}]
            reply = complete(client, ledger, model, qid, "direct", messages, 768, 25000,
                             folder / "direct/raw" / f"{qid}.json")
            try:
                parsed = json.loads(text(reply))
                if parsed.get("label") not in {"ANSWER", "ASK", "ABSTAIN"}:
                    raise ValueError("Invalid direct label")
                decision(folder / "direct/decisions" / f"{qid}.json", qid, parsed)
            except Exception as exc:
                decision(folder / "direct/decisions" / f"{qid}.json", qid, None, exc)
        for q in blind:
            qid = q["id"]
            messages = extraction_messages(q["question"], rules, prompt)
            reply = complete(client, ledger, model, qid, "s5query", messages, 4096, 40000,
                             folder / "matcher/raw_s5query" / f"{qid}.json")
            try:
                query = parse_s5_query(text(reply), q["question"])
                query_path = folder / "matcher/s5queries" / f"{qid}.json"
                save(query_path, query.model_dump())
                result = decide(query.engine_input(), rules)
                decision(folder / "matcher/decisions" / f"{qid}.json", qid, result)
            except Exception as exc:
                decision(folder / "matcher/decisions" / f"{qid}.json", qid, None, exc)
                decision(folder / "reasoner/decisions" / f"{qid}.json", qid, None,
                         ValueError("Invalid S5Query; reasoner call not made"))
                continue
            reasoner_request = reasoner_messages(q["question"], query, rules)
            reasoner_reply = complete(client, ledger, model, qid, "s5_reasoner", reasoner_request,
                                      2048, 100000, folder / "reasoner/raw" / f"{qid}.json")
            try:
                parsed = parse_decision(text(reasoner_reply), rules)
                decision(folder / "reasoner/decisions" / f"{qid}.json", qid, parsed)
            except Exception as exc:
                decision(folder / "reasoner/decisions" / f"{qid}.json", qid, None, exc)
        print(f"{slug(model_id)} finished; observed total USD {ledger['observed_usd']:.6f}", flush=True)
    print(f"Campaign finished; observed total USD {ledger['observed_usd']:.6f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    run(parser.parse_args().check)
