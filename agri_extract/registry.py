"""Document eligibility evidence; unknown fields remain unknown."""
from __future__ import annotations

import hashlib
from pathlib import Path
from pydantic import Field
from pypdf import PdfReader
from .schema import Strict
from .prepare import DOCUMENTS, DOCUMENT_CONFIG


class DocumentRecord(Strict):
    document_id: str
    title: str
    publisher: str
    source_url: str | None = None
    crop: str
    country: str
    publication_year: int
    version: str
    license_name: str | None = None
    license_url: str | None = None
    license_evidence: str | None = None
    license_verification_status: str
    page_count: int
    file_hash: str
    language: str
    eligibility_status: str
    eligibility_evidence: dict[str, str] = Field(default_factory=dict)


def register(document_id: str, path: Path | None = None) -> DocumentRecord:
    path = path or DOCUMENTS[document_id]
    config = DOCUMENT_CONFIG[document_id]
    reader = PdfReader(path)
    title = config.get("title") or (reader.metadata.title if reader.metadata else None) or path.stem
    crop = config.get("crop") or "unspecified"
    year = config.get("publication_year") or 0
    edition = config.get("version") or "unspecified"
    cover = reader.pages[0].extract_text() or ""
    rights = reader.pages[1].extract_text() or "" if len(reader.pages) > 1 else ""
    fertilizer_page = config.get("fertilizer_page")
    context_page = config.get("context_page")
    fertilizer_text = reader.pages[fertilizer_page - 1].extract_text() or "" if fertilizer_page else ""
    context_text = reader.pages[context_page - 1].extract_text() or "" if context_page else ""
    license_present = "Creative Commons Attribution 4.0 International" in rights
    evidence = next((line.strip() for line in rights.splitlines() if "Creative Commons Attribution 4.0 International" in line), None)
    return DocumentRecord(document_id=document_id, title=title, publisher=config.get("publisher") or "unspecified",
        crop=crop, country=config.get("country") or "unspecified", publication_year=year, version=edition,
        license_name="CC BY 4.0" if license_present else None,
        license_url="https://creativecommons.org/licenses/by/4.0/" if license_present else None,
        license_evidence=evidence,
        license_verification_status="STATED_IN_PDF" if license_present else "NOT_VERIFIED",
        page_count=len(reader.pages), file_hash=hashlib.sha256(path.read_bytes()).hexdigest(),
        language="en", eligibility_status="CANDIDATE_REVIEWED_LOCAL_PDF" if license_present else "PENDING_LICENSE",
        eligibility_evidence={"C1": config.get("publisher_evidence") or "Publisher not independently verified",
            "C2": cover[:180].replace("\n", " "),
            "C3": f"PDF p. {fertilizer_page}: " + fertilizer_text[:220].replace("\n", " ") if fertilizer_page else "No fertilizer page configured",
            "C4": f"PDF p. {context_page}: " + context_text[:220].replace("\n", " ") if context_page else "No context page configured",
            "C5": evidence or "No explicit license text found"})
