import pytest
from src.ingestion.district_priors import get_district_priors, get_plausible_crops
from src.labeling.temporal_clustering import (
    rank_crops_for_temporal_series,
    _detect_intercrop_candidates,
)


def test_amravati_priors_loading():
    priors = get_district_priors("Amravati")
    assert bool(priors) is True
    assert "soybean" in priors
    assert "cotton" in priors
    assert "paddy" in priors

    # In Amravati, soybean and cotton are dominant kharif crops
    assert priors["soybean"] > 0.35
    assert priors["cotton"] > 0.25
    # Paddy is virtually non-existent in Amravati (< 1% share)
    assert priors["paddy"] < 0.05


def test_district_normalization_and_fallback():
    # Case insensitivity
    priors_lower = get_district_priors("amravati")
    priors_upper = get_district_priors("AMRAVATI")
    priors_title = get_district_priors("Amravati")

    assert priors_lower == priors_upper == priors_title

    # Unknown or empty district returns empty dict
    assert get_district_priors("") == {}
    assert get_district_priors("Unknown_District_XYZ") == {}


def test_amravati_plausible_crops():
    plausible = get_plausible_crops("Amravati")
    assert "soybean" in plausible
    assert "cotton" in plausible
    assert "" not in plausible
    assert "unknown" not in plausible


def test_bayesian_prior_suppresses_paddy_ranking():
    # Reference synthetic observation
    # A timeseries of 6 months
    timeseries = [
        {"date": "2024-06", "ndvi": 0.22, "vv": -15.0, "vh": -22.0},
        {"date": "2024-07", "ndvi": 0.45, "vv": -12.0, "vh": -18.0},
        {"date": "2024-08", "ndvi": 0.72, "vv": -9.0, "vh": -15.0},
        {"date": "2024-09", "ndvi": 0.78, "vv": -10.0, "vh": -16.0},
        {"date": "2024-10", "ndvi": 0.40, "vv": -14.0, "vh": -20.0},
        {"date": "2024-11", "ndvi": 0.25, "vv": -16.0, "vh": -22.0},
    ]

    priors = get_district_priors("Amravati")

    # Without district priors (uniform prior)
    unweighted = rank_crops_for_temporal_series(timeseries, district_priors=None)

    # With Amravati district priors
    weighted = rank_crops_for_temporal_series(timeseries, district_priors=priors)

    # In weighted results, posterior = raw_score * prior
    for item in weighted:
        assert "score" in item
        assert "raw_score" in item
        assert "prior_weight" in item

    # If paddy is in unweighted, verify its rank or score drops dramatically in weighted
    paddy_unweighted = next((c for c in unweighted if c["crop"] == "paddy"), None)
    paddy_weighted = next((c for c in weighted if c["crop"] == "paddy"), None)

    if paddy_unweighted and paddy_weighted:
        # Paddy score should collapse by the prior ratio (paddy prior ~0.0098)
        assert paddy_weighted["score"] < paddy_unweighted["score"] * 0.1


def test_intercrop_candidate_heuristic():
    priors = get_district_priors("Amravati")
    # Synthetic timeseries typical of Soybean + Tur (Soybean harvested Oct, Tur peaks Nov-Dec)
    intercrop_ts = [
        {"date": "2024-06", "ndvi": 0.25},
        {"date": "2024-07", "ndvi": 0.55},
        {"date": "2024-08", "ndvi": 0.78},  # Soybean peak
        {"date": "2024-09", "ndvi": 0.65},
        {"date": "2024-10", "ndvi": 0.50},  # Soybean harvested, Tur standing
        {"date": "2024-11", "ndvi": 0.58},  # Tur secondary peak / shoulder
        {"date": "2024-12", "ndvi": 0.42},
        {"date": "2025-01", "ndvi": 0.20},
    ]

    candidates = _detect_intercrop_candidates(intercrop_ts, priors)
    assert len(candidates) > 0
    crop_names = [c["crop"] for c in candidates]
    assert any("tur" in name for name in crop_names)
