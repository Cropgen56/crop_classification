from __future__ import annotations

from typing import Any


def _normalize_feature(feature: dict[str, Any], index: int = 0) -> dict[str, Any]:
    if not isinstance(feature, dict):
        raise ValueError("Each field entry must be a dictionary.")

    properties = feature.get("properties") if isinstance(feature.get("properties"), dict) else {}
    normalized = dict(feature)
    normalized.update(properties)

    if "field_id" not in normalized:
        if feature.get("id") is not None:
            normalized["field_id"] = feature.get("id")
        elif "fid" in properties:
            normalized["field_id"] = properties["fid"]
        elif "FID" in properties:
            normalized["field_id"] = properties["FID"]
        elif "id" in properties:
            normalized["field_id"] = properties["id"]
        elif "ID" in properties:
            normalized["field_id"] = properties["ID"]
        else:
            normalized["field_id"] = f"field_{index + 1}"

    if "dominant_top_crop" not in normalized and "top_crop" in normalized:
        normalized["dominant_top_crop"] = normalized["top_crop"]

    return normalized


def normalize_village_fields(fields: Any) -> list[dict[str, Any]]:
    if fields is None:
        return []

    if isinstance(fields, dict):
        if fields.get("type") == "FeatureCollection":
            return [
                _normalize_feature(feature, index)
                for index, feature in enumerate(fields.get("features", []))
                if isinstance(feature, dict)
            ]

        if any(key in fields for key in ("properties", "geometry", "field_id", "dominant_top_crop", "top_crop")):
            return [_normalize_feature(fields, 0)]

    if isinstance(fields, list):
        normalized: list[dict[str, Any]] = []
        for index, item in enumerate(fields):
            if isinstance(item, dict):
                normalized.append(_normalize_feature(item, index))
            else:
                raise ValueError("Each field entry must be a dictionary.")
        return normalized

    raise ValueError("The 'fields' value must be a FeatureCollection, a list of field features, or a single field object.")
