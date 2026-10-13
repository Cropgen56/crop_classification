from __future__ import annotations

from typing import Any

from src.rules.village_classifier import classify_village

REQUEST_VERSION = "0.7.0"


def classify_village_request(payload: dict[str, Any]) -> dict[str, Any]:
    village_id = payload.get("village_id", "unknown")
    fields = payload.get("fields", [])

    if not isinstance(fields, list):
        raise ValueError("The 'fields' value must be a list of field profile objects.")

    village = {
        "village_id": village_id,
        "fields": fields,
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
    }
    return response
