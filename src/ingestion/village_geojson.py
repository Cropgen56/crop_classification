from __future__ import annotations

from typing import Any


def _normalize_feature(feature: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(feature, dict):
        raise ValueError("Each field entry must be a dictionary.")

    properties = feature.get("properties") if isinstance(feature.get("properties"), dict) else {}
    normalized = dict(feature)
    normalized.update(properties)

    if "field_id" not in normalized and feature.get("id") is not None:
        normalized["field_id"] = feature.get("id")

    if "dominant_top_crop" not in normalized and "top_crop" in normalized:
        normalized["dominant_top_crop"] = normalized["top_crop"]

    return normalized


def normalize_village_fields(fields: Any) -> list[dict[str, Any]]:
    if fields is None:
        return []

    if isinstance(fields, dict):
        if fields.get("type") == "FeatureCollection":
            return [
                _normalize_feature(feature)
                for feature in fields.get("features", [])
                if isinstance(feature, dict)
            ]

        if any(key in fields for key in ("properties", "geometry", "field_id", "dominant_top_crop", "top_crop")):
            return [_normalize_feature(fields)]

    if isinstance(fields, list):
        normalized: list[dict[str, Any]] = []
        for item in fields:
            if isinstance(item, dict) and item.get("type") == "Feature":
                normalized.append(_normalize_feature(item))
            elif isinstance(item, dict):
                normalized.append(_normalize_feature(item))
            else:
                raise ValueError("Each field entry must be a dictionary.")
        return normalized

    raise ValueError("The 'fields' value must be a FeatureCollection, a list of field features, or a single field object.")
