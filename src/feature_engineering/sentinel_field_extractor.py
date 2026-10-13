from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import ee
import pandas as pd
from shapely.geometry import shape

from src.labeling.temporal_reference import canonicalize_crop_name

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _monthly_windows(start_date: str, end_date: str) -> list[tuple[str, str, str]]:
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if end < start:
        raise ValueError("end_date must be on or after start_date")

    windows = []
    cursor = start.replace(day=1)
    while cursor <= end:
        next_month = cursor + pd.offsets.MonthBegin(1)
        window_start = max(start, cursor)
        window_end = min(end + pd.Timedelta(days=1), next_month)
        windows.append((window_start.strftime("%Y-%m-%d"), window_end.strftime("%Y-%m-%d"), cursor.strftime("%Y-%m")))
        cursor = next_month
    return windows


@lru_cache(maxsize=4)
def _initialize_earth_engine(credentials_path: str) -> bool:
    try:
        credential_info = json.loads(Path(credentials_path).read_text())
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT") or credential_info.get("project_id")
        if not project_id:
            return False
        credentials = ee.ServiceAccountCredentials(credential_info.get("client_email"), credentials_path)
        ee.Initialize(credentials=credentials, project=project_id)
        return True
    except Exception:
        return False


def _infer_default_shares(dominant_crop: str) -> dict[str, float]:
    crop = canonicalize_crop_name(dominant_crop)
    crop = crop.lower()
    if "paddy" in crop or "rice" in crop:
        return {"avg_kharif_share": 0.74, "avg_rabi_share": 0.12, "avg_zaid_share": 0.04, "avg_other_share": 0.10}
    if "soybean" in crop:
        return {"avg_kharif_share": 0.68, "avg_rabi_share": 0.14, "avg_zaid_share": 0.06, "avg_other_share": 0.12}
    if "cotton" in crop:
        return {"avg_kharif_share": 0.58, "avg_rabi_share": 0.16, "avg_zaid_share": 0.04, "avg_other_share": 0.22}
    if "jowar" in crop or "sorghum" in crop:
        return {"avg_kharif_share": 0.54, "avg_rabi_share": 0.18, "avg_zaid_share": 0.07, "avg_other_share": 0.21}
    if "wheat" in crop or "gram" in crop or "mustard" in crop:
        return {"avg_kharif_share": 0.18, "avg_rabi_share": 0.62, "avg_zaid_share": 0.10, "avg_other_share": 0.10}
    if "sugarcane" in crop or "banana" in crop or "orange" in crop or "mango" in crop or "sapota" in crop:
        return {"avg_kharif_share": 0.22, "avg_rabi_share": 0.18, "avg_zaid_share": 0.06, "avg_other_share": 0.54}
    if "tur" in crop or "maize" in crop or "bajra" in crop or "groundnut" in crop:
        return {"avg_kharif_share": 0.60, "avg_rabi_share": 0.15, "avg_zaid_share": 0.07, "avg_other_share": 0.18}
    return {"avg_kharif_share": 0.42, "avg_rabi_share": 0.24, "avg_zaid_share": 0.08, "avg_other_share": 0.26}


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
            profile["centroid_lon"] = float(poly.centroid.x)
            profile["centroid_lat"] = float(poly.centroid.y)
        except Exception:
            profile["area_km2"] = 0.0
            profile["centroid_lon"] = None
            profile["centroid_lat"] = None
    else:
        profile["area_km2"] = 0.0
        profile["centroid_lon"] = None
        profile["centroid_lat"] = None

    profile["satellite"] = {
        "source": "unavailable",
        "timeseries": [],
        "reason": "Sentinel observations were not fetched; no satellite values are being substituted.",
        "ndvi_mean": None,
        "ndre_mean": None,
        "gndvi_mean": None,
        "evi_mean": None,
        "ndmi_mean": None,
        "s1_vv_mean": None,
        "s1_vh_mean": None,
    }

    profile.update({key: value for key, value in properties.items() if key not in profile and key not in {"field_id"}})
    return profile


def _try_gee_village_profiles(
    features: list[dict[str, Any]],
    start_date: str,
    end_date: str,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[int, dict[str, Any]]:
    credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not credentials_path or not os.path.exists(credentials_path):
        return {}
    if not _initialize_earth_engine(credentials_path):
        return {}

    try:
        valid_features = []
        bounds = []
        for index, feature in enumerate(features):
            geometry = feature.get("geometry") if isinstance(feature.get("geometry"), dict) else None
            if geometry is None:
                continue
            try:
                min_x, min_y, max_x, max_y = shape(geometry).bounds
            except Exception:
                continue
            valid_features.append(
                ee.Feature(ee.Geometry(geometry), {"source_index": index})
            )
            bounds.append((min_x, min_y, max_x, max_y))
        if not valid_features:
            return {}

        region = ee.Geometry.BBox(
            min(bound[0] for bound in bounds),
            min(bound[1] for bound in bounds),
            max(bound[2] for bound in bounds),
            max(bound[3] for bound in bounds),
        )
        field_collection = ee.FeatureCollection(valid_features)
        s2 = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(region)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 80))
        )
        s1 = (
            ee.ImageCollection("COPERNICUS/S1_GRD")
            .filterBounds(region)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV"))
            .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
            .filter(ee.Filter.eq("instrumentMode", "IW"))
            .select(["VV", "VH"])
        )

        s2_bands = ["NDVI", "NDRE", "GNDVI", "EVI", "NDMI"]
        empty_s2 = ee.Image.constant([0] * len(s2_bands)).rename(s2_bands).updateMask(ee.Image.constant(0))
        empty_s1 = ee.Image.constant([0, 0]).rename(["VV", "VH"]).updateMask(ee.Image.constant(0))

        def add_s2_indices(image: Any) -> Any:
            scl = image.select("SCL")
            cloud_mask = scl.neq(0).And(scl.neq(1)).And(scl.neq(3)).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))
            reflectance = image.select(["B2", "B3", "B4", "B5", "B8", "B11"]).multiply(0.0001).updateMask(cloud_mask)
            blue = reflectance.select("B2")
            green = reflectance.select("B3")
            red = reflectance.select("B4")
            red_edge = reflectance.select("B5")
            nir = reflectance.select("B8")
            swir = reflectance.select("B11")
            ndvi = nir.subtract(red).divide(nir.add(red).max(0.0001)).rename("NDVI")
            ndre = nir.subtract(red_edge).divide(nir.add(red_edge).max(0.0001)).rename("NDRE")
            gndvi = nir.subtract(green).divide(nir.add(green).max(0.0001)).rename("GNDVI")
            evi = image.expression(
                "2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))",
                {"NIR": nir, "RED": red, "BLUE": blue},
            ).rename("EVI")
            ndmi = nir.subtract(swir).divide(nir.add(swir).max(0.0001)).rename("NDMI")
            return ee.Image.cat([ndvi, ndre, gndvi, evi, ndmi]).copyProperties(image, ["system:time_start"])

        monthly_images = []
        for window_start, window_end, month in _monthly_windows(start_date, end_date):
            s2_month = s2.filterDate(window_start, window_end).map(add_s2_indices)
            s1_month = s1.filterDate(window_start, window_end)
            s2_composite = s2_month.merge(ee.ImageCollection.fromImages([empty_s2])).median()
            s1_composite = s1_month.merge(ee.ImageCollection.fromImages([empty_s1])).mean()
            s2_count = ee.Image.constant(s2_month.size()).rename(f"s2_count_{month}")
            s1_count = ee.Image.constant(s1_month.size()).rename(f"s1_count_{month}")
            monthly_images.append(
                ee.Image.cat([
                    s2_composite.rename([f"{band}_{month}" for band in s2_bands]),
                    s1_composite.rename([f"VV_{month}", f"VH_{month}"]),
                    s2_count,
                    s1_count,
                ])
            )

        stacked = ee.Image.cat(monthly_images)
        if progress_callback:
            progress_callback({
                "stage": "satellite_extraction",
                "completed": 0,
                "total": len(features),
                "message": f"Fetching monthly Sentinel-1/2 data for {len(valid_features)} field polygons in one batch.",
            })

        reduced = stacked.reduceRegions(
            collection=field_collection,
            reducer=ee.Reducer.mean(),
            scale=20,
            tileScale=4,
        ).getInfo()

        profiles = {}
        windows = _monthly_windows(start_date, end_date)
        for result_feature in reduced.get("features", []):
            properties = result_feature.get("properties", {})
            try:
                source_index = int(properties["source_index"])
            except (KeyError, TypeError, ValueError):
                continue
            feature = features[source_index]
            observations = []
            for _, _, month in windows:
                record = {"date": month}
                for band in s2_bands + ["VV", "VH"]:
                    record[band.lower()] = _safe_float(properties.get(f"{band}_{month}"), None)
                record["s2_count"] = int(_safe_float(properties.get(f"s2_count_{month}"), 0))
                record["s1_count"] = int(_safe_float(properties.get(f"s1_count_{month}"), 0))
                observations.append(record)

            observed = [row for row in observations if any(row.get(name) is not None for name in ["ndvi", "vv", "vh"])]
            if not observed:
                continue

            profile = _feature_to_profile(feature, source_index)

            def mean_band(name: str) -> float | None:
                values_for_band = [row[name] for row in observations if row.get(name) is not None]
                return round(sum(values_for_band) / len(values_for_band), 5) if values_for_band else None

            profile["satellite"] = {
                "source": "gee_sentinel_monthly",
                "start_date": start_date,
                "end_date": end_date,
                "timeseries": observations,
                "observation_count": len(observed),
                "s2_observation_count": sum(row["s2_count"] for row in observations),
                "s1_observation_count": sum(row["s1_count"] for row in observations),
                "ndvi_mean": mean_band("ndvi"),
                "ndre_mean": mean_band("ndre"),
                "gndvi_mean": mean_band("gndvi"),
                "evi_mean": mean_band("evi"),
                "ndmi_mean": mean_band("ndmi"),
                "s1_vv_mean": mean_band("vv"),
                "s1_vh_mean": mean_band("vh"),
            }
            profiles[source_index] = profile

        if progress_callback:
            progress_callback({
                "stage": "satellite_extraction",
                "completed": len(features),
                "total": len(features),
                "message": f"Received monthly Sentinel-1/2 observations for {len(profiles)} of {len(valid_features)} polygons.",
            })
        return profiles
    except Exception:
        return {}


def build_village_field_profiles(
    village_payload: Any,
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
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

    _monthly_windows(start_date, end_date)
    satellite_profiles = _try_gee_village_profiles(
        features,
        start_date,
        end_date,
        progress_callback=progress_callback,
    )
    profiles: list[dict[str, Any]] = []
    for index, feature in enumerate(features):
        if not isinstance(feature, dict):
            continue
        profile = satellite_profiles.get(index)
        if profile is None:
            profile = _feature_to_profile(feature, index)
            credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
            if not credentials_path or not os.path.exists(credentials_path):
                profile["satellite"]["reason"] = "GOOGLE_APPLICATION_CREDENTIALS is missing or unreadable; no satellite values were substituted."
            else:
                profile["satellite"]["reason"] = "Earth Engine returned no usable monthly S1/S2 observations; no satellite values were substituted."
        profiles.append(profile)
    return profiles


__all__ = ["build_village_field_profiles"]
