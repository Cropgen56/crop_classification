from __future__ import annotations

import os
from typing import Any

import ee
from shapely.geometry import shape


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_crop_name(value: Any) -> str:
    if value is None:
        return "unknown"
    text = str(value).strip()
    if not text:
        return "unknown"
    return text


def _infer_default_shares(dominant_crop: str) -> dict[str, float]:
    crop = _normalize_crop_name(dominant_crop).lower()
    if "paddy" in crop or "rice" in crop:
        return {"avg_kharif_share": 0.72, "avg_rabi_share": 0.12, "avg_zaid_share": 0.04, "avg_other_share": 0.12}
    if "cotton" in crop:
        return {"avg_kharif_share": 0.56, "avg_rabi_share": 0.18, "avg_zaid_share": 0.04, "avg_other_share": 0.22}
    if "wheat" in crop or "gram" in crop or "mustard" in crop:
        return {"avg_kharif_share": 0.18, "avg_rabi_share": 0.58, "avg_zaid_share": 0.09, "avg_other_share": 0.15}
    if "sugarcane" in crop or "banana" in crop or "orange" in crop or "mango" in crop or "sapota" in crop:
        return {"avg_kharif_share": 0.22, "avg_rabi_share": 0.16, "avg_zaid_share": 0.06, "avg_other_share": 0.56}
    return {"avg_kharif_share": 0.42, "avg_rabi_share": 0.28, "avg_zaid_share": 0.08, "avg_other_share": 0.22}


def _feature_to_profile(feature: dict[str, Any], index: int) -> dict[str, Any]:
    if not isinstance(feature, dict):
        raise ValueError("Each field must be a dictionary.")

    properties = feature.get("properties") if isinstance(feature.get("properties"), dict) else {}
    if not properties:
        properties = {
            key: value
            for key, value in feature.items()
            if key not in {"field_id", "id", "geometry", "type", "properties", "satellite"}
        }

    geom = feature.get("geometry") if isinstance(feature.get("geometry"), dict) else None
    field_id = (
        feature.get("field_id")
        or feature.get("id")
        or properties.get("field_id")
        or f"field_{index + 1}"
    )
    dominant_crop = properties.get("dominant_top_crop") or properties.get("top_crop") or "unknown"

    default_shares = _infer_default_shares(dominant_crop)
    profile = {
        "field_id": str(field_id),
        "dominant_top_crop": str(dominant_crop),
        "avg_kharif_share": _safe_float(properties.get("avg_kharif_share", default_shares["avg_kharif_share"])),
        "avg_rabi_share": _safe_float(properties.get("avg_rabi_share", default_shares["avg_rabi_share"])),
        "avg_zaid_share": _safe_float(properties.get("avg_zaid_share", default_shares["avg_zaid_share"])),
        "avg_other_share": _safe_float(properties.get("avg_other_share", default_shares["avg_other_share"])),
        "top_crop_share": _safe_float(properties.get("top_crop_share", max(default_shares.values()))),
    }

    if geom is not None:
        try:
            poly = shape(geom)
            area_km2 = abs(poly.area) * 111_320.0 * 111_320.0 * max(0.0, abs(poly.centroid.y) if hasattr(poly.centroid, "y") else 0.0)
            profile["area_km2"] = round(max(area_km2, 0.0) / 1_000_000.0, 6)
        except Exception:
            profile["area_km2"] = 0.0
    else:
        profile["area_km2"] = 0.0

    profile["satellite"] = {
        "source": "synthetic_fallback",
        "ndvi_mean": 0.58,
        "ndre_mean": 0.29,
        "evi_mean": 0.41,
        "s1_vv_mean": -12.5,
        "s1_vh_mean": -19.2,
        "polarization_ratio": 6.7,
        "acquisition_count": 8,
    }

    profile.update({key: value for key, value in properties.items() if key not in profile and key not in {"field_id"}})
    return profile


def _try_gee_field_profile(feature: dict[str, Any], index: int, start_date: str, end_date: str) -> dict[str, Any] | None:
    geometry = feature.get("geometry") if isinstance(feature.get("geometry"), dict) else None
    if geometry is None:
        return None

    credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not credentials_path or not os.path.exists(credentials_path):
        return None

    try:
        ee.Initialize(project=None)
    except Exception:
        try:
            ee.Initialize(credentials=ee.ServiceAccountCredentials(None, credentials_path), project=None)
        except Exception:
            return None

    try:
        geom = ee.Geometry(geometry)
        s2 = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(geom)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 35))
        )

        if s2.size().getInfo() == 0:
            return None

        sentinel = s2.mean()
        nir = sentinel.select("B8")
        red = sentinel.select("B4")
        blue = sentinel.select("B2")
        ndvi = nir.subtract(red).divide(nir.add(red).clamp(0.0001)).rename("NDVI")
        ndre = nir.subtract(sentinel.select("B5")).divide(nir.add(sentinel.select("B5")).clamp(0.0001)).rename("NDRE")
        evi = sentinel.expression(
            "2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))",
            {"NIR": nir, "RED": red, "BLUE": blue},
        ).rename("EVI")

        s1 = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(geom)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .select(["VV", "VH"])
        )

        vv_mean = s1.select("VV").mean().reduceRegion(ee.Reducer.mean(), geom, 10, maxPixels=1e9).get("VV")
        vh_mean = s1.select("VH").mean().reduceRegion(ee.Reducer.mean(), geom, 10, maxPixels=1e9).get("VH")
        ndvi_mean = ndvi.reduceRegion(ee.Reducer.mean(), geom, 10, maxPixels=1e9).get("NDVI")
        ndre_mean = ndre.reduceRegion(ee.Reducer.mean(), geom, 10, maxPixels=1e9).get("NDRE")
        evi_mean = evi.reduceRegion(ee.Reducer.mean(), geom, 10, maxPixels=1e9).get("EVI")

        profile = _feature_to_profile(feature, index)
        profile["satellite"] = {
            "source": "gee_sentinel",
            "ndvi_mean": _safe_float(ndvi_mean),
            "ndre_mean": _safe_float(ndre_mean),
            "evi_mean": _safe_float(evi_mean),
            "s1_vv_mean": _safe_float(vv_mean),
            "s1_vh_mean": _safe_float(vh_mean),
            "polarization_ratio": round(_safe_float(vv_mean) - _safe_float(vh_mean), 4),
            "acquisition_count": int(s2.size().getInfo()),
        }
        return profile
    except Exception:
        return None


def build_village_field_profiles(village_payload: Any, start_date: str = "2024-01-01", end_date: str = "2024-12-31") -> list[dict[str, Any]]:
    if isinstance(village_payload, dict) and village_payload.get("type") == "FeatureCollection":
        features = village_payload.get("features", [])
    elif isinstance(village_payload, list):
        features = village_payload
    elif isinstance(village_payload, dict):
        if "features" in village_payload:
            features = village_payload["features"]
        elif any(key in village_payload for key in ("geometry", "properties", "field_id", "dominant_top_crop")):
            features = [village_payload]
        else:
            features = []
    else:
        features = []

    profiles: list[dict[str, Any]] = []
    for index, feature in enumerate(features):
        if not isinstance(feature, dict):
            continue
        profile = _try_gee_field_profile(feature, index, start_date, end_date)
        if profile is None:
            profile = _feature_to_profile(feature, index)
        profiles.append(profile)
    return profiles


__all__ = ["build_village_field_profiles"]
