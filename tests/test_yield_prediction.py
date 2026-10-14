from __future__ import annotations

import pytest

from src.service.classification_service import classify_village_request
from src.yield_engine.drought_indicators import compute_satellite_drought_indicators
from src.yield_engine.historical_baseline import (
    get_2026_anchor_deficit,
    get_reference_yield_kg_ha,
)
from src.yield_engine.trigger1_talukas import (
    get_division_summary,
    get_drought_taluka_info,
    is_trigger1_taluka,
    load_trigger1_taluka_registry,
)
from src.yield_engine.yield_predictor import predict_field_yield


def test_trigger1_talukas_coverage():
    registry = load_trigger1_taluka_registry()
    assert len(registry) == 265, "Expected exactly 265 Trigger-1 talukas from 2026 gazette"

    summary = get_division_summary()
    assert summary.get("Konkan") == 12
    assert summary.get("Nashik") == 49
    assert summary.get("Pune") == 35
    assert summary.get("Chhatrapati Sambhajinagar") == 74
    assert summary.get("Amravati") == 56
    assert summary.get("Nagpur") == 39

    # Known talukas from document
    assert is_trigger1_taluka("Ambajogai", "Beed") or is_trigger1_taluka("Ambejogai", "Beed")
    assert is_trigger1_taluka("Kaij", "Beed")
    assert is_trigger1_taluka("Yeola", "Nashik")
    assert is_trigger1_taluka("Yavatmal", "Yavatmal")
    assert not is_trigger1_taluka("FictionalTaluka123", "Mumbai")


def test_historical_baseline_reference_yields():
    # Official 2026 anchor check for Beed Soybean
    beed_soy_ref = get_reference_yield_kg_ha("Soybean", "Beed")
    assert pytest.approx(beed_soy_ref, rel=1e-2) == 1534.2

    # Check cotton and tur reference yields
    cotton_ref = get_reference_yield_kg_ha("Cotton", "Yavatmal")
    assert cotton_ref > 0.0

    tur_ref = get_reference_yield_kg_ha("Tur (Arhar)", "Yavatmal")
    assert tur_ref > 0.0

    # Anchor deficit lookup
    beed_deficit = get_2026_anchor_deficit("Soybean", "Beed")
    assert beed_deficit == -73.3


def test_drought_indicators_under_severe_stress():
    # Simulated 2026 drought timeseries for Beed / Marathwada
    drought_series = [
        {"date": "2026-06", "ndvi": 0.22, "ndre": 0.18, "gndvi": 0.20, "evi": 0.15, "ndmi": -0.05, "vv": -12.0, "vh": -22.5},
        {"date": "2026-07", "ndvi": 0.35, "ndre": 0.26, "gndvi": 0.30, "evi": 0.22, "ndmi": -0.02, "vv": -11.5, "vh": -21.0},
        {"date": "2026-08", "ndvi": 0.48, "ndre": 0.34, "gndvi": 0.38, "evi": 0.30, "ndmi": -0.08, "vv": -11.0, "vh": -19.5},
        {"date": "2026-09", "ndvi": 0.32, "ndre": 0.22, "gndvi": 0.25, "evi": 0.20, "ndmi": -0.15, "vv": -12.5, "vh": -21.5},
    ]

    indicators = compute_satellite_drought_indicators(drought_series, "Soybean")
    assert indicators["satellite_available"] is True
    assert indicators["peak_ndvi"] == 0.48
    assert indicators["ndmi_min"] == -0.15
    assert indicators["drought_stress_score"] >= 0.50
    assert indicators["stress_level"] in ["Severe", "Extreme"]


def test_drought_indicators_under_healthy_conditions():
    healthy_series = [
        {"date": "2024-06", "ndvi": 0.25, "ndre": 0.20, "gndvi": 0.22, "evi": 0.18, "ndmi": 0.15, "vv": -12.0, "vh": -20.0},
        {"date": "2024-07", "ndvi": 0.60, "ndre": 0.48, "gndvi": 0.52, "evi": 0.45, "ndmi": 0.30, "vv": -10.0, "vh": -16.5},
        {"date": "2024-08", "ndvi": 0.84, "ndre": 0.68, "gndvi": 0.72, "evi": 0.62, "ndmi": 0.38, "vv": -9.0, "vh": -14.2},
        {"date": "2024-09", "ndvi": 0.72, "ndre": 0.55, "gndvi": 0.60, "evi": 0.50, "ndmi": 0.25, "vv": -10.5, "vh": -15.8},
    ]

    indicators = compute_satellite_drought_indicators(healthy_series, "Soybean")
    assert indicators["peak_ndvi"] == 0.84
    assert indicators["ndmi_min"] > 0.0
    assert indicators["drought_stress_score"] < 0.20
    assert indicators["stress_level"] == "Normal"


def test_predict_field_yield_calibrated_against_2026_cce():
    drought_series = [
        {"date": "2026-06", "ndvi": 0.22, "ndre": 0.18, "gndvi": 0.20, "evi": 0.15, "ndmi": -0.05, "vv": -12.0, "vh": -22.5},
        {"date": "2026-07", "ndvi": 0.35, "ndre": 0.26, "gndvi": 0.30, "evi": 0.22, "ndmi": -0.02, "vv": -11.5, "vh": -21.0},
        {"date": "2026-08", "ndvi": 0.48, "ndre": 0.34, "gndvi": 0.38, "evi": 0.30, "ndmi": -0.08, "vv": -11.0, "vh": -19.5},
        {"date": "2026-09", "ndvi": 0.32, "ndre": 0.22, "gndvi": 0.25, "evi": 0.20, "ndmi": -0.15, "vv": -12.5, "vh": -21.5},
    ]

    pred = predict_field_yield(
        crop_name="Soybean",
        district="Beed",
        taluka="Kaij",
        satellite=drought_series,
        field_area_ha=2.5,
    )

    assert pred["is_trigger1_drought_taluka"] is True
    assert pred["reference_yield_kg_ha"] == 1534.2
    # Beed Soybean CCE in docx was 410 kg/ha (-73.3%). Our model prediction should reflect acute failure.
    assert pred["predicted_yield_kg_ha"] < 550.0
    assert pred["yield_deficit_pct"] <= -60.0
    assert pred["damage_category"] == "Severe Crop Failure"
    assert pred["area_ha"] == 2.5
    assert pred["production_loss_tonnes"] is not None
    assert pred["production_loss_tonnes"] > 2.0


def test_classification_service_includes_yield_and_drought_assessment():
    payload = {
        "village_id": "v_beed_001",
        "district_name": "Beed",
        "taluka_name": "Kaij",
        "fields": [
            {
                "field_id": "f_soy_1",
                "dominant_top_crop": "Soybean",
                "area_hectares": 3.0,
                "satellite": {
                    "source": "gee_sentinel_monthly",
                    "timeseries": [
                        {"date": "2026-06", "ndvi": 0.25, "ndre": 0.20, "gndvi": 0.22, "evi": 0.18, "ndmi": -0.02, "vv": -12.0, "vh": -22.0},
                        {"date": "2026-07", "ndvi": 0.40, "ndre": 0.30, "gndvi": 0.35, "evi": 0.25, "ndmi": -0.05, "vv": -11.0, "vh": -20.5},
                        {"date": "2026-08", "ndvi": 0.50, "ndre": 0.38, "gndvi": 0.40, "evi": 0.32, "ndmi": -0.10, "vv": -10.5, "vh": -19.0},
                        {"date": "2026-09", "ndvi": 0.35, "ndre": 0.25, "gndvi": 0.28, "evi": 0.22, "ndmi": -0.12, "vv": -12.0, "vh": -21.0},
                    ],
                },
            }
        ],
    }

    result = classify_village_request(payload)
    assert "drought_assessment" in result
    drought = result["drought_assessment"]
    assert drought["is_trigger1_drought_area"] is True
    assert drought["district"] == "Beed"
    assert drought["taluka"] == "Kaij"
    assert drought["average_yield_deficit_pct"] < -50.0

    field_pred = result["field_predictions"][0]
    assert "yield_prediction" in field_pred
    yp = field_pred["yield_prediction"]
    assert yp["predicted_yield_kg_ha"] > 0.0
    assert yp["damage_category"] == "Severe Crop Failure"

    # Verify each top candidate also has its crop-specific yield prediction
    assert len(field_pred["top_3_candidates"]) > 0
    for cand in field_pred["top_3_candidates"]:
        assert "yield_prediction" in cand
        assert cand["yield_prediction"]["predicted_yield_kg_ha"] > 0.0
        assert "yield_deficit_pct" in cand["yield_prediction"]
