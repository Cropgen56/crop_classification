from __future__ import annotations

import math
from typing import Any

from src.labeling.temporal_reference import canonicalize_crop_name

# Indicative normal peak NDVI expectations for healthy Maharashtra crops
HEALTHY_PEAK_NDVI: dict[str, float] = {
    "soybean": 0.82,
    "cotton": 0.78,
    "paddy": 0.85,
    "rice": 0.85,
    "maize": 0.82,
    "tur": 0.75,
    "arhar": 0.75,
    "pigeonpea": 0.75,
    "bajra": 0.72,
    "jowar": 0.74,
    "sorghum": 0.74,
    "groundnut": 0.76,
    "sugarcane": 0.88,
    "gram": 0.78,
    "wheat": 0.80,
}

# Indicative normal mature VH backscatter in dB (Sentinel-1 C-band)
HEALTHY_PEAK_VH_DB: dict[str, float] = {
    "cotton": -14.0,
    "sugarcane": -12.5,
    "maize": -13.5,
    "soybean": -15.5,
    "paddy": -14.5,
    "tur": -15.0,
    "jowar": -14.5,
    "bajra": -16.0,
    "groundnut": -17.0,
}


def compute_satellite_drought_indicators(
    timeseries: list[dict[str, Any]] | None,
    crop_name: str | None = None,
) -> dict[str, Any]:
    """
    Computes remote-sensing drought stress indicators from Sentinel-1 and Sentinel-2 timeseries.

    Returns:
        Dictionary containing:
        - peak_ndvi
        - integral_ndvi (iNDVI)
        - ndmi_min (critical water deficit)
        - ndmi_mean
        - ndre_mean
        - s1_vh_peak (dB)
        - s1_vh_mean (dB)
        - s1_crosspol_ratio
        - drought_stress_score (0.0 = optimal vigor, 1.0 = catastrophic drought collapse)
        - stress_level ("Normal", "Mild", "Moderate", "Severe", "Extreme")
    """
    if not timeseries:
        return {
            "satellite_available": False,
            "peak_ndvi": None,
            "integral_ndvi": None,
            "ndmi_min": None,
            "ndmi_mean": None,
            "ndre_mean": None,
            "s1_vh_peak": None,
            "s1_vh_mean": None,
            "s1_crosspol_ratio": None,
            "drought_stress_score": 0.0,
            "stress_level": "Unobserved",
        }

    canonical = canonicalize_crop_name(crop_name or "").lower()

    ndvis = [float(t["ndvi"]) for t in timeseries if t.get("ndvi") is not None and not math.isnan(float(t["ndvi"]))]
    ndmis = [float(t["ndmi"]) for t in timeseries if t.get("ndmi") is not None and not math.isnan(float(t["ndmi"]))]
    ndres = [float(t["ndre"]) for t in timeseries if t.get("ndre") is not None and not math.isnan(float(t["ndre"]))]
    vhs = [float(t["vh"]) for t in timeseries if t.get("vh") is not None and not math.isnan(float(t["vh"]))]
    vvs = [float(t["vv"]) for t in timeseries if t.get("vv") is not None and not math.isnan(float(t["vv"]))]

    peak_ndvi = max(ndvis) if ndvis else 0.0
    integral_ndvi = sum(ndvis) if ndvis else 0.0
    ndmi_min = min(ndmis) if ndmis else 0.0
    ndmi_mean = (sum(ndmis) / len(ndmis)) if ndmis else 0.0
    ndre_mean = (sum(ndres) / len(ndres)) if ndres else 0.0
    s1_vh_peak = max(vhs) if vhs else -25.0
    s1_vh_mean = (sum(vhs) / len(vhs)) if vhs else -25.0

    # Cross-polarization difference in dB: VH - VV
    # For healthy dense canopies, VH - VV is typically around -6 to -8 dB
    # For bare soil / stunted sparse crops, VH - VV drops to -11 to -14 dB
    if vhs and vvs and len(vhs) == len(vvs):
        diffs = [h - v for h, v in zip(vhs, vvs)]
        s1_crosspol_ratio = sum(diffs) / len(diffs)
    else:
        s1_crosspol_ratio = None

    # Expected reference benchmarks for this crop
    exp_peak_ndvi = next((v for k, v in HEALTHY_PEAK_NDVI.items() if k in canonical), 0.78)
    exp_peak_vh = next((v for k, v in HEALTHY_PEAK_VH_DB.items() if k in canonical), -15.0)

    # 1. Canopy vigor deficit (NDVI penalty)
    # If peak NDVI is far below expected healthy peak:
    ndvi_deficit = max(0.0, min(1.0, (exp_peak_ndvi - peak_ndvi) / max(0.2, exp_peak_ndvi - 0.35)))

    # 2. Moisture stress deficit (NDMI penalty)
    # Healthy crop: NDMI > 0.25 (penalty = 0.0)
    # Water stress: NDMI between 0.0 and 0.15
    # Severe desiccation: NDMI < 0.0 (penalty = 1.0 at NDMI <= -0.15)
    if ndmis:
        ndmi_penalty = max(0.0, min(1.0, (0.25 - ndmi_min) / 0.40))
    else:
        ndmi_penalty = ndvi_deficit

    # 3. Biomass stunting deficit (Sentinel-1 VH penalty)
    # Healthy mature canopy: around -14 to -15 dB
    # Severe stunted canopy: <= -21 dB
    if vhs:
        vh_penalty = max(0.0, min(1.0, (exp_peak_vh - s1_vh_peak) / 6.0))
    else:
        vh_penalty = ndvi_deficit

    # 4. Seasonal cumulative biomass deficit
    # Normal 4-month kharif cumulative NDVI is approx 2.0 to 2.8
    # Severe drought cumulative NDVI drops below 1.4
    expected_integral = exp_peak_ndvi * max(len(ndvis) * 0.65, 2.0)
    if expected_integral > 0 and integral_ndvi > 0:
        integral_deficit = max(0.0, min(1.0, (expected_integral - integral_ndvi) / expected_integral))
    else:
        integral_deficit = ndvi_deficit

    # Weighted composite drought stress score: [0.0 = optimal, 1.0 = catastrophic failure]
    drought_stress_score = (
        0.35 * ndmi_penalty
        + 0.30 * ndvi_deficit
        + 0.20 * vh_penalty
        + 0.15 * integral_deficit
    )
    drought_stress_score = round(max(0.0, min(1.0, drought_stress_score)), 4)

    if drought_stress_score >= 0.70:
        stress_level = "Extreme"
    elif drought_stress_score >= 0.50:
        stress_level = "Severe"
    elif drought_stress_score >= 0.30:
        stress_level = "Moderate"
    elif drought_stress_score >= 0.15:
        stress_level = "Mild"
    else:
        stress_level = "Normal"

    return {
        "satellite_available": True,
        "peak_ndvi": round(peak_ndvi, 3),
        "integral_ndvi": round(integral_ndvi, 3),
        "ndmi_min": round(ndmi_min, 3),
        "ndmi_mean": round(ndmi_mean, 3),
        "ndre_mean": round(ndre_mean, 3),
        "s1_vh_peak": round(s1_vh_peak, 2),
        "s1_vh_mean": round(s1_vh_mean, 2),
        "s1_crosspol_ratio": round(s1_crosspol_ratio, 2) if s1_crosspol_ratio is not None else None,
        "drought_stress_score": drought_stress_score,
        "stress_level": stress_level,
    }
