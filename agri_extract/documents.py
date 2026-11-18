"""Auditable document pipelines P0–P4. No API transmission occurs here."""
from __future__ import annotations

import hashlib
import json
import re
import tempfile
from importlib.metadata import version
from pathlib import Path
from typing import Literal

from pydantic import Field
from pypdf import PdfReader, PdfWriter

from .prepare import DOCUMENTS, numeric_tokens
from .schema import Strict


class Transform(Strict):
    element_id: str
    kind: str
    original: str
    cleaned: str
    justification: str
    warnings: list[str] = Field(default_factory=list)


class Element(Strict):
    element_id: str
    page_number: int
    kind: Literal["title", "paragraph", "table", "note", "ocr"]
    section_id: str | None = None
    original: str
    cleaned: str
    table_id: str | None = None
    cells: list[list[str]] | None = None
    bbox: list[float] | None = None


class Page(Strict):
    number: int
    raw_text: str
    structured_text: str
    prepared_text: str
    ocr_candidate: bool
    ocr_used: bool
    elements: list[Element]
    numeric_missing: list[str] = Field(default_factory=list)
    numeric_added: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class DocumentArtifact(Strict):
    document_id: str
    file_name: str
    file_hash: str
    parser: Literal["P0", "P1", "P2", "P3", "P4"]
    version: str
    pages: list[Page]
    transforms: list[Transform] = Field(default_factory=list)
    docling_tables: list[dict] = Field(default_factory=list)
    conversion: dict[str, str | int | bool | None]
    warnings: list[str] = Field(default_factory=list)

    def llm_text(self) -> str:
        return "\n\n".join(f"=== PDF PAGE {p.number} ===\n" + "\n\n".join(
            f"[{e.element_id}{' TABLE' if e.kind == 'table' else ''}] {e.cleaned}" for e in p.elements)
            for p in self.pages)


def clean_text(text: str) -> str:
    for old, new in {"\u00a0": " ", "\u200b": "", "\ufb01": "fi", "\ufb02": "fl"}.items():
        text = text.replace(old, new)
    return "\n".join(re.sub(r"[ \t]+", " ", line).strip()
                     for line in text.splitlines() if line.strip())


def _blocks(text: str) -> list[str]:
    blocks, current, table_mode = [], [], False
    for line in text.splitlines():
        is_table = line.lstrip().startswith("|")
        if not line.strip() or (current and is_table != table_mode):
            if current:
                blocks.append("\n".join(current))
                current = []
            if not line.strip():
                continue
        current.append(line)
        table_mode = is_table
    if current:
        blocks.append("\n".join(current))
    return blocks


def _elements(document_id: str, page: int, text: str, cleaned: bool) -> tuple[list[Element], list[Transform]]:
    blocks = _blocks(text.strip())
    out: list[Element] = []
    transforms: list[Transform] = []
    for index, block in enumerate(filter(str.strip, blocks), 1):
        element_id = f"{document_id}:p{page:03d}:e{index:04d}"
        lines = block.splitlines()
        table = bool(lines and lines[0].strip().startswith("|"))
        kind = "table" if table else "title" if block.startswith("#") else "note" if re.match(r"(?i)^(note|figure|table)\s*\d", block) else "paragraph"
        normalized = clean_text(block) if cleaned else block
        cells = None
        if table:
            cells = [[x.strip() for x in line.strip().strip("|").split("|")]
                     for line in lines if line.strip().startswith("|") and not re.fullmatch(r"[|:\- ]+", line.strip())]
        out.append(Element(element_id=element_id, page_number=page, kind=kind,
                           original=block, cleaned=normalized,
                           table_id=element_id if table else None, cells=cells))
        if normalized != block:
            transforms.append(Transform(element_id=element_id, kind="whitespace_only",
                original=block, cleaned=normalized, justification="Deterministic whitespace normalization"))
    return out, transforms


def convert(document_id: str, pipeline: str, path: Path | None = None,
            ocr_threshold: int = 80, ocr_languages: list[str] | None = None) -> DocumentArtifact:
    if pipeline not in {"P0", "P1", "P2", "P3", "P4"}:
        raise ValueError(pipeline)
    path = path or DOCUMENTS[document_id]
    reader = PdfReader(path)
    native = [(p.extract_text() or "") for p in reader.pages]
    ocr_pages = [i + 1 for i, text in enumerate(native) if len(text.strip()) < ocr_threshold]
    structured = native[:]
    docling_tables: list[dict] = []
    warnings: list[str] = []
    use_ocr = pipeline in {"P3", "P4"} and bool(ocr_pages)
    successful_ocr: set[int] = set()
    if pipeline != "P0":
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
        from docling.document_converter import DocumentConverter, PdfFormatOption
        options = PdfPipelineOptions(do_table_structure=pipeline in {"P2", "P3", "P4"}, do_ocr=False)
        if pipeline in {"P2", "P3", "P4"}:
            options.table_structure_options.mode = TableFormerMode.ACCURATE
        converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)})
        result = converter.convert(path)
        for table in getattr(result.document, "tables", []):
            docling_tables.append({"self_ref": getattr(table, "self_ref", None),
                "provenance": [p.model_dump(mode="json") for p in getattr(table, "prov", [])],
                "num_rows": table.data.num_rows, "num_cols": table.data.num_cols,
                "cells": [cell.model_dump(mode="json") for cell in table.data.table_cells]})
        for i in range(len(native)):
            rendered = result.document.export_to_markdown(page_no=i + 1)
            if rendered.strip():
                structured[i] = rendered
            else:
                warnings.append(f"page {i+1}: empty Docling output; native retained")
        if use_ocr:
            ocr_options = PdfPipelineOptions(do_table_structure=True, do_ocr=True)
            ocr_options.table_structure_options.mode = TableFormerMode.ACCURATE
            if ocr_languages:
                ocr_options.ocr_options.lang = ocr_languages
            ocr_converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=ocr_options)})
            with tempfile.TemporaryDirectory(prefix="agri_ocr_") as temporary:
                for page_no in ocr_pages:
                    writer = PdfWriter()
                    writer.add_page(reader.pages[page_no - 1])
                    selected_page = Path(temporary) / f"page_{page_no}.pdf"
                    with selected_page.open("wb") as handle:
                        writer.write(handle)
                    try:
                        ocr_result = ocr_converter.convert(selected_page)
                        rendered = ocr_result.document.export_to_markdown(page_no=1)
                        if rendered.strip():
                            structured[page_no - 1] = rendered
                            successful_ocr.add(page_no)
                        else:
                            warnings.append(f"page {page_no}: OCR returned empty text")
                    except Exception as exc:
                        warnings.append(f"page {page_no}: OCR failed: {type(exc).__name__}: {exc}")
    pages: list[Page] = []
    transforms: list[Transform] = []
    for i, (raw, parsed) in enumerate(zip(native, structured), 1):
        missing = list((numeric_tokens(raw) - numeric_tokens(parsed)).elements())
        added = list((numeric_tokens(parsed) - numeric_tokens(raw)).elements())
        source = parsed
        page_warnings = []
        if pipeline == "P4" and len(missing) > 1:
            source = raw
            page_warnings.append(f"Docling omitted numeric tokens {missing}; native text used")
        elif pipeline in {"P1", "P2", "P3"} and missing:
            page_warnings.append(f"Docling omitted numeric tokens {missing}; no fallback in {pipeline}")
        items, changes = _elements(document_id, i, source, pipeline == "P4")
        if i in successful_ocr:
            for element in items:
                if element.kind == "paragraph":
                    element.kind = "ocr"
        transforms.extend(changes)
        pages.append(Page(number=i, raw_text=raw, structured_text=parsed,
            prepared_text=clean_text(source) if pipeline == "P4" else source,
            ocr_candidate=i in ocr_pages, ocr_used=i in successful_ocr,
            elements=items, numeric_missing=missing, numeric_added=added, warnings=page_warnings))
    current_section = None
    for page in pages:
        for element in page.elements:
            if element.kind == "title":
                current_section = element.element_id
            element.section_id = current_section
    return DocumentArtifact(document_id=document_id, file_name=path.name,
        file_hash=hashlib.sha256(path.read_bytes()).hexdigest(), parser=pipeline, version="1",
        pages=pages, transforms=transforms,
        docling_tables=docling_tables,
        conversion={"tool": "pypdf" if pipeline == "P0" else "docling",
                    "pypdf_version": version("pypdf"), "docling_version": version("docling") if pipeline != "P0" else None,
                    "table_structure": pipeline in {"P2", "P3", "P4"}, "ocr_enabled": use_ocr,
                    "tableformer_mode": "ACCURATE" if pipeline in {"P2", "P3", "P4"} else None,
                    "ocr_engine": "auto" if use_ocr else None,
                    "ocr_languages": json.dumps(ocr_languages or []),
                    "ocr_threshold": ocr_threshold, "ocr_pages": json.dumps(ocr_pages),
                    "quality_status": "REVIEW_REQUIRED" if any(p.warnings for p in pages) else "AUTOMATED_CHECKS_ONLY"},
        warnings=warnings)


def save_artifact(artifact: DocumentArtifact, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{artifact.document_id}_{artifact.parser}.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")
    return path
