"""Offline end-to-end smoke test for every strategy and schema on registered artifacts.

Responses are deliberately empty mock rule sets. This tests plumbing, not extraction
quality, and does not create or modify human gold annotations.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

from agri_extract.documents import DocumentArtifact
from agri_extract.experiments import execute
from agri_extract.prepare import ROOT, DOCUMENTS
from agri_extract.providers import MockProvider


def _providers(strategy: str, document_id: str) -> list[MockProvider]:
    rules = json.dumps({"document_id": document_id, "rules": []})
    candidates = json.dumps({"candidates": []})
    atoms = json.dumps({"atoms": []})
    verification = json.dumps({"agreements": [], "disagreements": [],
                               "supported_rule_ids": [], "review_required": []})
    if strategy == "E1":
        return [MockProvider([rules])]
    if strategy == "E2":
        return [MockProvider([candidates, rules])]
    if strategy == "E3":
        return [MockProvider([rules]), MockProvider([rules])]
    if strategy == "E4":
        return [MockProvider([rules]), MockProvider([rules, verification])]
    if strategy in {"E5", "E6"}:
        return [MockProvider([candidates]), MockProvider([rules])]
    if strategy == "E7":
        return [MockProvider([candidates, atoms, rules])]
    if strategy == "E8":
        return [MockProvider([rules, rules])]
    raise ValueError(strategy)


def _hashes(paths: list[Path]) -> dict[str, str]:
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def main() -> None:
    if not DOCUMENTS:
        raise RuntimeError("No local documents configured; set AGRI_DOCUMENT_CONFIG or configs/documents.local.json")
    gold_paths = sorted((ROOT / "artifacts" / "gold").glob("*.json"))
    before = _hashes(gold_paths)
    summary = {"kind": "MOCK_PLUMBING_TEST", "paid_api_calls": 0,
               "human_gold_modified": False, "documents": []}
    with tempfile.TemporaryDirectory(prefix="agri_mock_smoke_") as temp:
        output_root = Path(temp)
        for document_id in DOCUMENTS:
            p4 = DocumentArtifact.model_validate_json((ROOT / "artifacts" / "documents" /
                f"{document_id}_P4_tables.json").read_text(encoding="utf-8"))
            runs = 0
            for schema in ("S1", "S2", "S3", "S4", "S5"):
                for strategy in ("E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8"):
                    config = {"strategy": strategy, "schema": schema,
                              "validation": "V1", "parser": "P4"}
                    folder = execute(config, p4, _providers(strategy, document_id),
                                     "mock", output_root=output_root)
                    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
                    assert manifest["status"] == "SIMULATED", (config, manifest)
                    assert (folder / "raw_trace.json").exists()
                    assert (folder / "validated.json").exists() or strategy in {"E3", "E4", "E8"}
                    runs += 1
            for level in ("V0", "V1", "V2", "V3", "V4", "V5"):
                config = {"strategy": "E1", "schema": "S4", "validation": level,
                          "parser": "P4"}
                folder = execute(config, p4, _providers("E1", document_id),
                                 "mock", output_root=output_root)
                manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
                assert manifest["status"] == "SIMULATED", (config, manifest)
                runs += 1
            for level in ("V3", "V4", "V5"):
                for mode in ("V-Same", "V-Cross"):
                    config = {"strategy": "E1", "schema": "S4", "validation": level,
                              "verification_mode": mode, "parser": "P4"}
                    extractor = _providers("E1", document_id)[0]
                    verifier = MockProvider([json.dumps({"checks": []})],
                        model=extractor.model if mode == "V-Same" else "mock-verifier-v2")
                    folder = execute(config, p4, [extractor], "mock", output_root=output_root,
                                     verifier=verifier)
                    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
                    assert manifest["status"] == "SIMULATED", (config, manifest)
                    assert (folder / "evidence_checks.json").exists()
                    runs += 1
            for parser in ("P0", "P1", "P2", "P3"):
                parsed = DocumentArtifact.model_validate_json((ROOT / "artifacts" / "documents" /
                    f"{document_id}_{parser}.json").read_text(encoding="utf-8"))
                config = {"strategy": "E1", "schema": "S3", "validation": "V1", "parser": parser}
                folder = execute(config, parsed, _providers("E1", document_id),
                                 "mock", output_root=output_root)
                assert json.loads((folder / "manifest.json").read_text())["status"] == "SIMULATED"
                runs += 1
            summary["documents"].append({"document_id": document_id,
                "p4_strategy_schema_runs": 40, "p4_validation_runs": 6,
                "p4_verifier_mode_runs": 6, "p0_to_p3_reference_runs": 4,
                "completed_mock_runs": runs})
    summary["human_gold_modified"] = before != _hashes(gold_paths)
    assert not summary["human_gold_modified"]
    path = ROOT / "artifacts" / "audits" / "mock_smoke_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
