"""Private, label-blind DeepInfra continuation for revised questions."""
from __future__ import annotations

import getpass
import hashlib
import importlib.util
import json
import os
from pathlib import Path

from agri_extract.documents import DocumentArtifact
from agri_extract.s5_matcher import decide
from agri_extract.s5_query import S5Query, extraction_messages, parse_s5_query
from agri_extract.s5_reasoner import messages as reasoner_messages, parse_decision
from agri_extract.s5_working_copy import validated_copy

ROOT = Path(__file__).resolve().parents[2]
OLD_MATCHER = ROOT / "results/s5_matcher_single_model_deepseek_exploratory"
OUT = ROOT / "results/annotation_v5_live"
MODEL = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
CHANGED = {"G01", "G04", "G06", "G08"}
CAP_USD = 0.25

spec = importlib.util.spec_from_file_location("baseline_support_archive", ROOT / "agri_extract/baseline_support.py")
support = importlib.util.module_from_spec(spec)
spec.loader.exec_module(support)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temp.replace(path)


def content(reply: dict) -> str:
    return reply["raw"]["choices"][0]["message"]["content"] or ""


def invoke(client, model_info: dict, ledger: dict, ledger_path: Path,
           path: Path, qid: str, method: str, messages: list[dict],
           max_tokens: int, reserve_input: int) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    reserve = 1.5 * (reserve_input * model_info["input_price"] +
                     max_tokens * model_info["output_price"]) / 1_000_000
    if ledger["reserved_usd"] + reserve > CAP_USD:
        raise RuntimeError(f"Budget reservation cap reached before {qid} {method}")
    ledger["reserved_usd"] += reserve
    save(ledger_path, ledger)
    reply = client.complete(MODEL, messages, max_tokens)
    reply.update({"model": MODEL, "question_id": qid, "method": method,
                  "request_sha256": hashlib.sha256(json.dumps(messages, ensure_ascii=False).encode()).hexdigest(),
                  "cost_usd": support.cost(reply, model_info), "reserved_usd": reserve})
    ledger["observed_usd"] += reply["cost_usd"]
    save(path, reply)
    save(ledger_path, ledger)
    print(f"{qid} {method}: saved, observed total USD {ledger['observed_usd']:.6f}", flush=True)
    if "error" in reply:
        raise RuntimeError(f"API request failed for {qid} {method}; inspect saved raw response")
    return reply


def valid_content(reply: dict) -> str:
    if "error" in reply or not reply.get("raw", {}).get("choices"):
        raise ValueError("API response has no completion")
    if reply["raw"]["choices"][0].get("finish_reason") == "length":
        raise ValueError("Output was truncated")
    return content(reply)


def main() -> None:
    blind_path = ROOT / "data/private/annotation_v5_blind.jsonl"
    prior_path = ROOT / "data/private/original_questions_blind.jsonl"
    rules_path = ROOT / "data/private/verified_s5_rules.json"
    pdf_artifact_path = ROOT / "artifacts/documents/garlic_P4_tables.json"
    blind = [json.loads(line) for line in blind_path.read_text(encoding="utf-8").splitlines() if line]
    prior = {item["id"]: item for item in (json.loads(line) for line in prior_path.read_text(encoding="utf-8").splitlines() if line)}
    if len(blind) != 15 or len({q["id"] for q in blind}) != 15:
        raise ValueError("Expected 15 unique blind questions")
    if {q["id"] for q in blind if q["question"] != prior[q["id"]]["question"]} != CHANGED:
        raise ValueError("Changed question set differs from preflight")
    old_manifest = json.loads((ROOT / "results/direct_baseline_archive/manifest.json").read_text(encoding="utf-8"))
    if old_manifest["files_sha256"].get("garlic_P4_tables.json") != digest(pdf_artifact_path):
        raise ValueError("Prepared PDF artifact differs from the original baseline")
    if old_manifest["prompt_sha256"]["direct"] != hashlib.sha256(support.DIRECT_SYSTEM.encode()).hexdigest():
        raise ValueError("Direct system prompt differs from the original baseline")
    if old_manifest["api_parameters"] != support.API_PARAMS:
        raise ValueError("API parameters differ from the original baseline")
    old_matcher_manifest = json.loads((OLD_MATCHER / "run_manifest.json").read_text(encoding="utf-8"))
    if old_matcher_manifest["question_sha256"] != digest(prior_path) or old_matcher_manifest["s5_sha256"] != digest(rules_path):
        raise ValueError("Saved S5Query provenance differs from prior questions or S5 rules")
    rules, changes = validated_copy(json.loads(rules_path.read_text(encoding="utf-8")))
    artifact = DocumentArtifact.model_validate_json(pdf_artifact_path.read_text(encoding="utf-8"))
    document = artifact.llm_text()
    if hashlib.sha256(document.encode()).hexdigest() != old_manifest["document_text_sha256"]:
        raise ValueError("Full PDF text differs from the original baseline")
    model_info = next(model for model in json.loads((ROOT / "data/private/study_code/configs__deepinfra_models_snapshot.json").read_text(encoding="utf-8"))["models"] if model["id"] == MODEL)
    query_prompt_path = ROOT / "configs/s5_query_system_prompt_v1.txt"
    query_prompt = query_prompt_path.read_text(encoding="utf-8")
    if digest(query_prompt_path) != old_matcher_manifest["s5_query_prompt_sha256"]:
        raise ValueError("S5Query prompt differs from the saved run")
    save(OUT / "run_manifest.json", {"model": MODEL, "blind_sha256": digest(blind_path),
         "prior_blind_sha256": digest(prior_path), "rules_sha256": digest(rules_path),
         "pdf_artifact_sha256": digest(pdf_artifact_path),
         "direct_system_prompt_sha256": hashlib.sha256(support.DIRECT_SYSTEM.encode()).hexdigest(),
         "direct_user_format": "JSON object with pdf_text and question",
         "direct_user_format_matches_old_run": False,
         "s5_query_prompt_sha256": digest(query_prompt_path),
         "reasoner_system_prompt_sha256": hashlib.sha256(__import__("agri_extract.s5_reasoner", fromlist=["SYSTEM_PROMPT"]).SYSTEM_PROMPT.encode()).hexdigest(),
         "changed_question_ids": sorted(CHANGED), "gold_opened_by_runner": False,
         "s5_encoding_audit": changes, "cap_usd": CAP_USD})
    ledger_path = OUT / "budget_ledger.json"
    ledger = json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.exists() else {
        "cap_usd": CAP_USD, "reserved_usd": 0.0, "observed_usd": 0.0}
    key = os.getenv("DEEPINFRA_API_KEY") or getpass.getpass("DeepInfra key (hidden): ")
    client = support.DeepInfra(key)
    for question in blind:
        qid = question["id"]
        user = json.dumps({"pdf_text": document, "question": question["question"]}, ensure_ascii=False)
        messages = [{"role": "system", "content": support.DIRECT_SYSTEM},
                    {"role": "user", "content": user}]
        path = OUT / "direct/raw" / f"{qid}_direct.json"
        try:
            reply = invoke(client, model_info, ledger, ledger_path, path, qid, "direct", messages, 768, 25000)
            parsed = json.loads(valid_content(reply))
            if parsed.get("label") not in {"ANSWER", "ASK", "ABSTAIN"}:
                raise ValueError("Invalid direct label")
            save(OUT / "direct/decisions" / f"{qid}.json", {"question_id": qid, "decision": parsed})
        except Exception as exc:
            save(OUT / "direct/decisions" / f"{qid}.json", {"question_id": qid,
                 "error": {"type": type(exc).__name__, "message": str(exc)[:500]}})
            if "API request failed" in str(exc) or "Budget reservation cap" in str(exc):
                raise
    for question in blind:
        qid = question["id"]
        if qid not in CHANGED:
            continue
        messages = extraction_messages(question["question"], rules, query_prompt)
        path = OUT / "matcher/raw_s5query" / f"{qid}.json"
        try:
            reply = invoke(client, model_info, ledger, ledger_path, path, qid, "s5query", messages, 4096, 40000)
            parsed = parse_s5_query(valid_content(reply), question["question"])
            query_path = OUT / "matcher/s5queries" / f"{qid}.json"
            save(query_path, parsed.model_dump())
            save(OUT / "matcher/decisions" / f"{qid}.json", {"question_id": qid,
                 "decision": decide(parsed.engine_input(), rules), "s5query_sha256": digest(query_path)})
        except Exception as exc:
            save(OUT / "matcher/decisions" / f"{qid}.json", {"question_id": qid,
                 "error": {"type": type(exc).__name__, "message": str(exc)[:500]}})
            if "API request failed" in str(exc) or "Budget reservation cap" in str(exc):
                raise
            continue
        reasoner_request = reasoner_messages(question["question"], parsed, rules)
        reasoner_path = OUT / "reasoner/raw" / f"{qid}.json"
        try:
            reasoner_reply = invoke(client, model_info, ledger, ledger_path, reasoner_path, qid,
                                    "s5_reasoner", reasoner_request, 2048, 100000)
            decision = parse_decision(valid_content(reasoner_reply), rules)
            save(OUT / "reasoner/decisions" / f"{qid}.json", {"question_id": qid,
                 "decision": decision, "s5query_sha256": digest(query_path)})
        except Exception as exc:
            save(OUT / "reasoner/decisions" / f"{qid}.json", {"question_id": qid,
                 "error": {"type": type(exc).__name__, "message": str(exc)[:500]}})
            if "API request failed" in str(exc) or "Budget reservation cap" in str(exc):
                raise
    print(f"Completed. New observed API cost USD {ledger['observed_usd']:.6f}")


if __name__ == "__main__":
    main()
