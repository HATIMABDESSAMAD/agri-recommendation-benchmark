"""Workspace source paths and conservative numeric comparison helpers."""
from __future__ import annotations

import os
import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(os.environ.get("AGRI_PROJECT_ROOT", Path.cwd())).resolve()
DOCUMENT_CONFIG_PATH = Path(os.environ.get("AGRI_DOCUMENT_CONFIG", ROOT / "configs/documents.local.json"))
DOCUMENT_CONFIG = (json.loads(DOCUMENT_CONFIG_PATH.read_text(encoding="utf-8"))
                   if DOCUMENT_CONFIG_PATH.exists() else {})
DOCUMENTS = {document_id: (ROOT / entry["path"]).resolve()
             for document_id, entry in DOCUMENT_CONFIG.items()}


def numeric_tokens(text: str) -> Counter[str]:
    """Compare literal numbers without converting chemical bases or punctuation."""
    return Counter(re.findall(r"(?<![\w.])\d+(?:[.,]\d+)?%?(?![\w.])", text))
