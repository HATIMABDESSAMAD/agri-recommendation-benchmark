"""Frozen direct-baseline identity and DeepInfra request helpers."""
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BLIND = ROOT / "data/private/garlic_questions_blind.jsonl"
GOLD = ROOT / "data/private/garlic_questions_gold.jsonl"
RULES = ROOT / "Garlic_PROF_S5_12_QA.json"
PDF = ROOT / "docs/Bestfitmanual_MUCASCAPE_Garlic_Sep2019final.pdf"
P4 = ROOT / "artifacts/documents/garlic_P4_tables.json"
OUT = ROOT / "results/direct_baseline_archive"
MAX_OUTPUT = 768
API_PARAMS = {"temperature": 0, "reasoning_effort": "none"}
DIRECT_SYSTEM = """You are assessing a garlic fertilizer question against the supplied full PDF text.
Use only the PDF. Return exactly one JSON object with keys label (ANSWER, ASK, or
ABSTAIN), response, justification, cited_pages (array of integers).
ANSWER only if the document and farmer facts support a precise, safe answer.
ASK if a necessary farmer fact is missing; response must ask a specific question.
ABSTAIN if the requested advice is unsupported by the document. Never invent
dates, doses, conversions, soil thresholds, or substitutions. Distinguish
fertilizer product mass from nutrient mass. Quote no gold labels."""


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def content(reply: dict) -> str:
    return reply["raw"]["choices"][0]["message"]["content"] or ""


def cost(reply: dict, model: dict) -> float:
    usage = reply.get("raw", {}).get("usage") or {}
    if usage.get("estimated_cost") is not None:
        return float(usage["estimated_cost"])
    return (usage.get("prompt_tokens", 0) * model["input_price"] +
            usage.get("completion_tokens", 0) * model["output_price"]) / 1_000_000


def parse_object(value: str) -> dict:
    result = json.loads(value)
    if not isinstance(result, dict):
        raise ValueError("Expected JSON object")
    return result


class DeepInfra:
    def __init__(self, key: str):
        if not key:
            raise ValueError("DEEPINFRA_API_KEY is required")
        self.key = key

    def complete(self, model: str, messages: list[dict], max_tokens: int = MAX_OUTPUT) -> dict:
        body = json.dumps({"model": model, "messages": messages, "max_tokens": max_tokens,
                           **API_PARAMS}).encode("utf-8")
        request = urllib.request.Request("https://api.deepinfra.com/v1/openai/chat/completions",
            data=body, headers={"Authorization": "Bearer " + self.key,
                                "Content-Type": "application/json"})
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                raw = response.read().decode("utf-8", errors="replace")
            return {"raw": json.loads(raw), "latency_s": time.monotonic() - started}
        except urllib.error.HTTPError as exc:
            return {"error": {"type": "HTTPError", "status": exc.code,
                              "body": exc.read().decode("utf-8", errors="replace")[:2000]},
                    "latency_s": time.monotonic() - started}
        except Exception as exc:
            return {"error": {"type": type(exc).__name__, "message": str(exc)[:500]},
                    "latency_s": time.monotonic() - started}
