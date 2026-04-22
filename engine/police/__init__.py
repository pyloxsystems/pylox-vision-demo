"""Pylox Vision — Law Enforcement Module.

Features built to make Pylox the infrastructure South Florida police depend on:

    report_narrative   — auto-generated prosecutor-ready incident paragraph
    tactical_brief     — structured responder briefing (suspect count, direction, tools)
    evidence_cert      — FL Statute 90.902(11) self-authenticating certification PDFs
    case_management    — BSO/Miami-Dade case number binding + evidence holds
    responder_view     — signed JWT mobile live-view links for responding officers
    bolo_scan          — natural language search over historical incidents
    vehicle_ai         — secondary Gemini pass for make/model/color/year extraction
    officer_api        — FastAPI router exposing everything under /api/v2/police/*

All features are buildable on existing hardware with the existing Gemini API key.
No external approvals required for the features in this package.

See /home/acme-corpai/.claude/plans/florida-police-features.md for the full strategy.
"""

__all__ = [
    "report_narrative",
    "tactical_brief",
    "evidence_cert",
    "case_management",
    "responder_view",
    "bolo_scan",
    "vehicle_ai",
    "officer_api",
]
