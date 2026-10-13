from __future__ import annotations

from typing import Any

from src.feature_engineering.sentinel_field_extractor import build_village_field_profiles
from src.ingestion.village_geojson import normalize_village_fields
from src.rules.field_candidate_generator import classify_field_profile
from src.rules.village_classifier import classify_village

REQUEST_VERSION = "0.7.0"


def _build_field_prediction(profile: dict[str, Any]) -> dict[str, Any]:
    field_result = classify_field_profile(profile)
    top_3 = []
    for rank, candidate in enumerate(field_result["ranked_candidates"][:3], start=1):
        top_3.append({
            "rank": rank,
            "crop": candidate["crop"],
            "score": candidate["score"],
            "season": candidate.get("season"),
        })

    return {
        "field_id": profile.get("field_id"),
        "best_crop": field_result["best_crop"],
        "best_score": field_result["best_score"],
        "season_label": field_result["season_label"],
        "top_3_candidates": top_3,
    }


def classify_village_request(payload: dict[str, Any]) -> dict[str, Any]:
    village_id = payload.get("village_id", "unknown")
    fields = payload.get("fields", [])

    if isinstance(payload, dict) and "fields" not in payload and "type" in payload and payload.get("type") == "FeatureCollection":
        fields = payload

    field_profiles = build_village_field_profiles(fields)
    if not field_profiles:
        field_profiles = normalize_village_fields(fields)

    field_predictions = [_build_field_prediction(profile) for profile in field_profiles]

    village = {
        "village_id": village_id,
        "fields": field_profiles,
    }

    result = classify_village(village)
    response = {
        "request_version": REQUEST_VERSION,
        "village_id": village_id,
        "dominant_crop": result.get("dominant_crop"),
        "dominant_share": result.get("dominant_share"),
        "field_count": result.get("field_count"),
        "summary": result.get("summary"),
        "top_candidates": result.get("top_candidates", []),
        "crop_mix": result.get("crop_mix", []),
        "field_predictions": field_predictions,
    }
    return response
