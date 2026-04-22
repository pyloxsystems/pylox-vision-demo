"""BOLO Auto-Scan — Natural Language Search Over Historical Incidents.

When a BOLO (Be On the Look Out) hits the airwaves — "red Ford F-150 missing
passenger mirror, last seen Pompano area" — a detective can query Pylox in
plain English and get matching hits from every protected site in seconds,
instead of calling each business manually.

This v1 implementation runs an in-memory TF-IDF match over historical incident
narrations + tactical brief text + extracted vehicle details. It's fast, it
uses zero external services, and it works today.

A v2 upgrade path exists: swap the ranker for Gemini embeddings (already have
the API key) or a local sentence-transformer model. Keep the same interface so
callers don't need to change.

Usage:
    from engine.police.bolo_scan import search_bolo
    hits = search_bolo("red ford f-150 missing passenger mirror", window_hours=72)
    # returns list of ranked incident matches with scores
"""

import logging
import re
import time
import math
from collections import Counter
from typing import Optional

from engine import database as db

logger = logging.getLogger("pylox-v2.police.bolo")


_WORD_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "of", "in", "on", "at", "to", "for",
    "with", "by", "from", "is", "was", "were", "be", "been", "being", "this",
    "that", "these", "those", "it", "its", "as", "if", "then", "than", "so",
    "no", "not", "do", "does", "did", "has", "have", "had", "seen", "last",
}


def _tokenize(text: str) -> list:
    if not text:
        return []
    return [
        w for w in _WORD_RE.findall(text.lower())
        if len(w) > 1 and w not in _STOPWORDS
    ]


def _build_document(incident_id: int) -> str:
    """Assemble a searchable text document for an incident.

    Combines narrations + tactical brief + vehicle details + police narrative.
    """
    incident = db.get_incident(incident_id)
    if not incident:
        return ""

    parts = []
    parts.append(f"camera {incident.get('camera', '')}")
    parts.append(f"trigger {incident.get('trigger_source', '')}")
    parts.append(f"resolution {incident.get('resolution', '')}")

    for n in incident.get("narrations", []):
        parts.append(n.get("narration", ""))
        parts.append(n.get("action", ""))

    police = db.get_incident_police_data(incident_id)
    if police:
        if police.get("report_narrative"):
            parts.append(police["report_narrative"])
        if police.get("tactical_brief"):
            try:
                import json as _j
                tb = _j.loads(police["tactical_brief"])
                for s in tb.get("suspects", []):
                    parts.append(s.get("description", ""))
                    parts.append(" ".join(s.get("weapons", [])))
                    parts.append(" ".join(s.get("tools_visible", [])))
                    parts.append(s.get("distinguishing_features", ""))
                    parts.append(s.get("direction_of_travel", ""))
                parts.append(tb.get("entry_point", ""))
                parts.append(tb.get("exit_route_predicted", ""))
                parts.append(tb.get("last_known_location", ""))
            except Exception:
                pass
        if police.get("vehicle_details"):
            try:
                import json as _j
                vd = _j.loads(police["vehicle_details"])
                for k in ("make", "model", "color", "year_range", "body_type",
                          "license_plate", "plate_state", "distinguishing_features"):
                    v = vd.get(k)
                    if v:
                        parts.append(str(v))
            except Exception:
                pass

    return " ".join(parts)


def _score_incident(
    query_tokens: list,
    doc_tokens: list,
    df: dict,
    total_docs: int,
) -> float:
    """TF-IDF-ish cosine-like score between query and document."""
    if not query_tokens or not doc_tokens:
        return 0.0
    doc_counts = Counter(doc_tokens)
    doc_len = max(1, sum(doc_counts.values()))
    score = 0.0
    for qt in query_tokens:
        tf = doc_counts.get(qt, 0) / doc_len
        if tf == 0:
            continue
        idf = math.log((total_docs + 1) / (df.get(qt, 0) + 1)) + 1.0
        score += tf * idf
    # Boost multi-word literal matches (e.g. "f 150" in both)
    query_set = set(query_tokens)
    doc_set = set(doc_tokens)
    overlap = len(query_set & doc_set) / max(1, len(query_set))
    score *= 0.5 + overlap
    return score


def search_bolo(
    query: str,
    window_hours: int = 72,
    camera: str = None,
    limit: int = 20,
    min_score: float = 0.05,
) -> list:
    """Run a BOLO search over historical incidents.

    Args:
        query: natural language BOLO description
        window_hours: how far back to search (default 72h)
        camera: optional camera filter
        limit: max results
        min_score: minimum relevance score to return

    Returns:
        list of dicts: {incident, score, matched_terms, excerpt}
    """
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    since = time.time() - (window_hours * 3600)
    candidates = db.get_incidents(camera=camera, since=since, limit=500)

    if not candidates:
        return []

    # Build document for each candidate
    docs = []
    for c in candidates:
        doc_text = _build_document(c["id"])
        tokens = _tokenize(doc_text)
        if tokens:
            docs.append({"incident": c, "tokens": tokens, "text": doc_text})

    if not docs:
        return []

    # Compute document frequency for IDF
    df = Counter()
    for d in docs:
        for token in set(d["tokens"]):
            df[token] += 1
    total_docs = len(docs)

    # Score all
    scored = []
    for d in docs:
        score = _score_incident(query_tokens, d["tokens"], df, total_docs)
        if score < min_score:
            continue
        matched = [qt for qt in set(query_tokens) if qt in d["tokens"]]
        excerpt = _build_excerpt(d["text"], matched)
        scored.append({
            "incident_id": d["incident"]["id"],
            "camera": d["incident"]["camera"],
            "start_time": d["incident"]["start_time"],
            "end_time": d["incident"].get("end_time"),
            "max_threat": d["incident"].get("max_threat"),
            "resolution": d["incident"].get("resolution"),
            "score": round(score, 4),
            "matched_terms": matched,
            "excerpt": excerpt,
        })

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:limit]


def _build_excerpt(text: str, matched_terms: list, window: int = 60) -> str:
    """Build a short excerpt around the first match for display."""
    if not text:
        return ""
    lower = text.lower()
    for term in matched_terms:
        idx = lower.find(term)
        if idx >= 0:
            start = max(0, idx - window // 2)
            end = min(len(text), idx + len(term) + window // 2)
            excerpt = text[start:end].strip()
            if start > 0:
                excerpt = "... " + excerpt
            if end < len(text):
                excerpt = excerpt + " ..."
            return excerpt
    return (text[:120] + "...") if len(text) > 120 else text


def vehicle_plate_lookup(plate: str, window_hours: int = 168) -> list:
    """Quick lookup: find all incidents where this plate was seen."""
    if not plate:
        return []
    plate_clean = plate.upper().replace("-", "").replace(" ", "")
    return search_bolo(
        query=plate_clean,
        window_hours=window_hours,
        min_score=0.1,
        limit=50,
    )
