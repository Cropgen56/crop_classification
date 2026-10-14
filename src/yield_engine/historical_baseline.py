from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.labeling.temporal_reference import canonicalize_crop_name
from .trigger1_talukas import _norm

BASELINE_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "processed"
    / "crop_yield_reference_baseline.json"
)

# Standard statewide fallback reference yields (kg/ha) if historical files missing
DEFAULT_FALLBACK_YIELDS_KG_HA: dict[str, float] = {
    "soybean": 1250.0,
    "cotton": 450.0,
    "paddy": 2100.0,
    "rice": 2100.0,
    "maize": 2400.0,
    "tur": 850.0,
    "arhar": 850.0,
    "pigeonpea": 850.0,
    "bajra": 780.0,
    "jowar": 820.0,
    "sorghum": 820.0,
    "groundnut": 1150.0,
    "gram": 950.0,
    "chickpea": 950.0,
    "wheat": 1650.0,
    "sugarcane": 82000.0,
    "onion": 14500.0,
}


@lru_cache(maxsize=1)
def load_yield_baseline() -> dict[str, Any]:
    """Loads precomputed district & statewide 5-year reference yields."""
    if not BASELINE_PATH.exists():
        return {}
    try:
        data = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def _match_crop_key(crop_name: str, candidate_keys: list[str]) -> str | None:
    canonical = canonicalize_crop_name(crop_name).lower()
    for key in candidate_keys:
        k_norm = canonicalize_crop_name(key).lower()
        if canonical == k_norm:
            return key
        if canonical in k_norm or k_norm in canonical:
            return key
    return None


def get_reference_yield_kg_ha(crop_name: str, district: str | None = None) -> float:
    """
    Returns the normal 5-year reference yield in kg/ha for a given crop and district.
    Falls back to state average, then agronomic defaults.
    """
    baseline = load_yield_baseline()
    canonical = canonicalize_crop_name(crop_name).lower()

    # 1. Check official 2026 anchor overrides (e.g., Beed Soybean = 1534.2 kg/ha)
    if district:
        dist_norm = _norm(district)
        anchors = baseline.get("official_2026_anchors", {})
        for anchor_key, anchor_val in anchors.items():
            if "::" in anchor_key:
                a_dist, a_crop = anchor_key.split("::", 1)
                if _norm(a_dist) == dist_norm and canonical in canonicalize_crop_name(a_crop).lower():
                    if "reference_yield_kg_ha" in anchor_val:
                        return float(anchor_val["reference_yield_kg_ha"])

    # 2. Check district-crop average
    dist_map = baseline.get("district_crop_averages_kg_ha", {})
    if district and dist_map:
        dist_norm = _norm(district)
        for key, val in dist_map.items():
            if "::" in key:
                d, c = key.split("::", 1)
                if _norm(d) == dist_norm and canonical in canonicalize_crop_name(c).lower():
                    if val and float(val) > 0:
                        return float(val)

    # 3. Check state crop average
    state_map = baseline.get("state_averages_kg_ha", {})
    if state_map:
        matched_key = _match_crop_key(crop_name, list(state_map.keys()))
        if matched_key and state_map[matched_key] and float(state_map[matched_key]) > 0:
            return float(state_map[matched_key])

    # 4. Fallback defaults
    for k, v in DEFAULT_FALLBACK_YIELDS_KG_HA.items():
        if k in canonical:
            return v

    return 1000.0


def get_2026_anchor_deficit(crop_name: str, district: str | None = None) -> float | None:
    """Returns official early 2026 CCE yield deficit percentage if documented for this district-crop."""
    if not district:
        return None
    baseline = load_yield_baseline()
    dist_norm = _norm(district)
    canonical = canonicalize_crop_name(crop_name).lower()

    anchors = baseline.get("official_2026_anchors", {})
    for anchor_key, anchor_val in anchors.items():
        if "::" in anchor_key:
            a_dist, a_crop = anchor_key.split("::", 1)
            if _norm(a_dist) == dist_norm and canonical in canonicalize_crop_name(a_crop).lower():
                return float(anchor_val.get("deficit_pct", 0.0))
    return None
