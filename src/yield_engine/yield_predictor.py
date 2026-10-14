from __future__ import annotations

import logging
from typing import Any

from src.labeling.temporal_reference import canonicalize_crop_name
from .drought_indicators import compute_satellite_drought_indicators
from .historical_baseline import (
    get_2026_anchor_deficit,
    get_reference_yield_kg_ha,
)
from .trigger1_talukas import get_drought_taluka_info, is_trigger1_taluka

logger = logging.getLogger(__name__)

# Crop vulnerability factor to drought stress in Maharashtra
# Represents maximum yield loss fraction under extreme drought (score = 1.0)
CROP_DROUGHT_VULNERABILITY: dict[str, float] = {
    "tur": 0.82,
    "arhar": 0.82,
    "pigeonpea": 0.82,
    "soybean": 0.80,
    "cotton": 0.72,
    "bajra": 0.68,
    "maize": 0.65,
    "groundnut": 0.55,
    "jowar": 0.45,
    "sorghum": 0.45,
    "paddy": 0.40,
    "rice": 0.40,
    "wheat": 0.50,
    "gram": 0.48,
    "sugarcane": 0.50,
}


def predict_field_yield(
    crop_name: str,
    district: str | None = None,
    taluka: str | None = None,
    satellite: dict[str, Any] | list[dict[str, Any]] | None = None,
    field_area_ha: float | None = None,
) -> dict[str, Any]:
    """
    Predicts crop yield, percentage deficit vs 5-year baseline, and production loss
    using Sentinel-1/2 remote sensing observations and official 2026 drought anchors.

    Args:
        crop_name: Classified or reported crop name
        district: District name (e.g. 'Beed', 'Yavatmal', 'Nashik')
        taluka: Taluka name (e.g. 'Ambejogai', 'Yeola')
        satellite: Dictionary containing 'timeseries' or direct list of monthly/dekad records
        field_area_ha: Field area in hectares (optional, for total production estimation)

    Returns:
        Structured dictionary conforming to the official government yield deficit formulas.
    """
    canonical_crop = canonicalize_crop_name(crop_name)
    crop_key = canonical_crop.lower()

    # 1. 5-year reference baseline yield (kg/ha and t/ha)
    ref_yield_kg_ha = get_reference_yield_kg_ha(canonical_crop, district)
    ref_yield_t_ha = round(ref_yield_kg_ha / 1000.0, 3)

    # 2. Trigger-1 administrative status
    in_trigger1 = is_trigger1_taluka(taluka or "", district)
    taluka_drought_info = get_drought_taluka_info(taluka or "", district) if in_trigger1 else None

    # 3. Extract timeseries
    timeseries = None
    if isinstance(satellite, dict):
        timeseries = satellite.get("timeseries")
    elif isinstance(satellite, list):
        timeseries = satellite

    indicators = compute_satellite_drought_indicators(timeseries, canonical_crop)
    has_satellite = indicators.get("satellite_available", False)

    # Vulnerability factor
    vuln = next((v for k, v in CROP_DROUGHT_VULNERABILITY.items() if k in crop_key), 0.60)

    # Check if there is an official 2026 anchor point (e.g. Beed Soybean -73.3%)
    anchor_deficit_pct = get_2026_anchor_deficit(canonical_crop, district)

    # 4. Compute yield deficit %
    if has_satellite:
        stress_score = float(indicators.get("drought_stress_score", 0.0))
        # Deficit fraction scaled by crop vulnerability
        base_deficit_fraction = min(0.88, stress_score * vuln)

        if anchor_deficit_pct is not None:
            # Calibrate satellite score with ground anchor
            # If stress is high and anchor exists, smoothly merge
            anchor_fraction = abs(anchor_deficit_pct) / 100.0
            deficit_fraction = 0.65 * base_deficit_fraction + 0.35 * anchor_fraction
        else:
            deficit_fraction = base_deficit_fraction

        yield_deficit_pct = -round(deficit_fraction * 100.0, 1)
    else:
        # Fallback when satellite observation is absent
        if anchor_deficit_pct is not None:
            yield_deficit_pct = round(anchor_deficit_pct, 1)
        elif in_trigger1:
            # In a notified drought taluka without satellite data, apply default moderate-severe deficit
            yield_deficit_pct = -round(vuln * 60.0, 1)
        else:
            yield_deficit_pct = 0.0

    # 5. Estimated 2026 harvest yield
    loss_fraction = abs(yield_deficit_pct) / 100.0
    pred_yield_kg_ha = round(max(0.0, ref_yield_kg_ha * (1.0 - loss_fraction)), 1)
    pred_yield_t_ha = round(pred_yield_kg_ha / 1000.0, 3)

    # 6. Area and Production Loss in tonnes (Official formula from Section 1)
    area = max(0.0, float(field_area_ha or 0.0))
    if area > 0:
        est_prod_tonnes = round((area * pred_yield_kg_ha) / 1000.0, 3)
        prod_loss_tonnes = round((area * ref_yield_kg_ha * loss_fraction) / 1000.0, 3)
    else:
        est_prod_tonnes = None
        prod_loss_tonnes = None

    # Severity category
    abs_loss = abs(yield_deficit_pct)
    if abs_loss >= 60.0:
        damage_category = "Severe Crop Failure"
    elif abs_loss >= 35.0:
        damage_category = "Substantial Loss"
    elif abs_loss >= 20.0:
        damage_category = "Moderate Deficit"
    else:
        damage_category = "Normal / Minor Variation"

    return {
        "crop": canonical_crop,
        "district": district or "Unknown",
        "taluka": taluka or "Unknown",
        "is_trigger1_drought_taluka": in_trigger1,
        "predicted_yield_kg_ha": pred_yield_kg_ha,
        "predicted_yield_t_ha": pred_yield_t_ha,
        "reference_yield_kg_ha": ref_yield_kg_ha,
        "reference_yield_t_ha": ref_yield_t_ha,
        "yield_deficit_pct": yield_deficit_pct,
        "damage_category": damage_category,
        "area_ha": round(area, 3) if area > 0 else None,
        "estimated_production_tonnes": est_prod_tonnes,
        "production_loss_tonnes": prod_loss_tonnes,
        "drought_indicators": indicators,
        "public_evidence": taluka_drought_info.get("public_evidence") if taluka_drought_info else None,
    }
