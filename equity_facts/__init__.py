"""Issuer financial-statement extraction (EQX v3 worker B): scanned PDF -> validated financial facts.

Public interface (spec-v3-document-pipeline.md §7), called by ai/equity_pipeline.py:
    EXTRACTOR_VERSION
    extract_document(conn, document, root, *, run_id, heartbeat) -> dict
    derive_and_cross_validate(conn, symbol, *, run_id) -> dict
Neither function commits; the orchestrator owns the transaction per document.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from .extract import EXTRACTOR_VERSION, OcrEngine, extract_with
from .persist import PgRepo

__all__ = ["EXTRACTOR_VERSION", "derive_and_cross_validate", "extract_document"]


def extract_document(conn, document: dict, root: Path, *, run_id: str | None = None,
                     heartbeat: Callable[[], None] = lambda: None) -> dict:
    """Extract facts from one EquitySourceDocuments row (dict). Idempotent per (DocumentId, EXTRACTOR_VERSION)."""
    return extract_with(PgRepo(conn), OcrEngine(), document, Path(root), run_id=run_id, heartbeat=heartbeat)


def derive_and_cross_validate(conn, symbol: str, *, run_id: str | None = None) -> dict:
    """Cross-document checks + derived facts for one symbol (idempotent by FactKey). Does not commit."""
    from .derive import run

    return run(PgRepo(conn), symbol.upper(), run_id=run_id)
