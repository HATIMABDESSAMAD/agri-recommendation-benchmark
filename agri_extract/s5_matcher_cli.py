"""Offline S5Query pipeline and source freeze. No API adapter is imported."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .s5_matcher import decide
from .s5_query import extraction_messages, parse_s5_query
from .s5_working_copy import validated_copy

ROOT = Path(__file__).resolve().parents[1]
FILES = [ROOT / "agri_extract" / name for name in
         ("s5_matcher.py", "s5_query.py", "s5_matcher_cli.py", "schema.py")]
FILES += [ROOT / "configs/s5_query_system_prompt_v1.txt"]
MANIFEST_ROOT = ROOT / "results/s5_matcher_frozen"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest_path(rules_path: Path) -> Path:
    key = hashlib.sha256(str(rules_path.resolve()).encode("utf-8")).hexdigest()[:16]
    return MANIFEST_ROOT / key / "frozen_manifest.json"


def frozen(rules_path: Path) -> dict:
    files = [*FILES, rules_path.resolve()]
    def location(path: Path) -> str:
        try:
            return path.relative_to(ROOT).as_posix()
        except ValueError:
            return path.as_posix()
    return {"version": "S5Matcher-1.0.0", "files_sha256":
            {location(p): sha(p) for p in files},
            "gold_labels_used": False, "api_calls": False}


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline S5Query to S5 matching")
    parser.add_argument("command", choices=["freeze", "messages", "decide"])
    parser.add_argument("--question")
    parser.add_argument("--query", type=Path)
    parser.add_argument("--rules", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    manifest = manifest_path(args.rules)
    if args.command == "freeze":
        snapshot = frozen(args.rules)
        manifest.parent.mkdir(parents=True, exist_ok=True)
        if manifest.exists():
            if json.loads(manifest.read_text(encoding="utf-8")) != snapshot:
                raise RuntimeError("Frozen S5Matcher source or S5 changed")
        else:
            manifest.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
        print("S5Matcher source and S5 frozen")
        return
    if not manifest.exists() or json.loads(manifest.read_text(encoding="utf-8")) != frozen(args.rules):
        raise RuntimeError("Run freeze first; S5Matcher source or S5 changed")
    ruleset, changes = validated_copy(json.loads(args.rules.read_text(encoding="utf-8")))
    if args.command == "messages":
        if not args.question:
            parser.error("messages requires --question")
        prompt = (ROOT / "configs/s5_query_system_prompt_v1.txt").read_text(encoding="utf-8")
        result = extraction_messages(args.question, ruleset, prompt)
    else:
        if not args.query:
            parser.error("decide requires --query")
        data = json.loads(args.query.read_text(encoding="utf-8"))
        query = parse_s5_query(json.dumps(data, ensure_ascii=False), data["original_question"])
        result = {"decision": decide(query.engine_input(), ruleset),
                  "query": query.model_dump(), "source_rules_sha256": sha(args.rules),
                  "s5_encoding_audit": changes}
    output = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output, encoding="utf-8")
    else:
        print(output)


if __name__ == "__main__":
    main()
