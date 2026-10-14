from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

TRIGGER1_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "processed"
    / "trigger1_drought_talukas_2026.json"
)

# Canonical district alias mappings for Maharashtra (historical vs renamed)
DISTRICT_ALIASES = {
    "ahmednagar": "ahilyanagar",
    "ahilyanagar": "ahilyanagar",
    "aurangabad": "chhatrapati sambhajinagar",
    "chhatrapati sambhajinagar": "chhatrapati sambhajinagar",
    "osmanabad": "dharashiv",
    "dharashiv": "dharashiv",
    "beed": "beed",
    "bid": "beed",
}


def _norm(name: str | None) -> str:
    if not name:
        return ""
    cleaned = str(name).strip().lower()
    return DISTRICT_ALIASES.get(cleaned, cleaned)


@lru_cache(maxsize=1)
def load_trigger1_taluka_registry() -> list[dict[str, Any]]:
    """Loads the 265 Trigger-1 drought talukas designated for Kharif 2026."""
    if not TRIGGER1_PATH.exists():
        return []
    try:
        data = json.loads(TRIGGER1_PATH.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []


@lru_cache(maxsize=1)
def _taluka_lookup_map() -> dict[str, dict[str, Any]]:
    registry = load_trigger1_taluka_registry()
    lookup: dict[str, dict[str, Any]] = {}
    for entry in registry:
        dist_norm = _norm(entry.get("district"))
        tal_norm = _norm(entry.get("taluka"))
        # Key by district + taluka
        lookup[f"{dist_norm}::{tal_norm}"] = entry
        # Also key by taluka alone for fast fallback
        if tal_norm not in lookup:
            lookup[tal_norm] = entry
    return lookup


def is_trigger1_taluka(taluka: str, district: str | None = None) -> bool:
    """Returns True if the given taluka is included in the official 2026 Trigger-1 drought register."""
    if not taluka:
        return False
    lookup = _taluka_lookup_map()
    tal_norm = _norm(taluka)
    if district:
        dist_norm = _norm(district)
        if f"{dist_norm}::{tal_norm}" in lookup:
            return True
    return tal_norm in lookup


def get_drought_taluka_info(taluka: str, district: str | None = None) -> dict[str, Any] | None:
    """Retrieves full drought declaration and public evidence details for a taluka."""
    if not taluka:
        return None
    lookup = _taluka_lookup_map()
    tal_norm = _norm(taluka)
    if district:
        dist_norm = _norm(district)
        info = lookup.get(f"{dist_norm}::{tal_norm}")
        if info:
            return dict(info)
    info = lookup.get(tal_norm)
    return dict(info) if info else None


def get_division_summary() -> dict[str, int]:
    """Returns count of Trigger-1 talukas by revenue division."""
    registry = load_trigger1_taluka_registry()
    summary: dict[str, int] = {}
    for entry in registry:
        div = entry.get("revenue_division", "Other")
        summary[div] = summary.get(div, 0) + 1
    return summary
