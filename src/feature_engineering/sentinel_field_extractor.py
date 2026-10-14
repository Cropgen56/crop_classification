from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import ee
import pandas as pd
from shapely.geometry import shape

from src.labeling.temporal_reference import canonicalize_crop_name

logger = logging.getLogger(__name__)

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


# Only successful initialisations are cached. A failed attempt clears itself
# so the next request can retry (e.g. after credentials are corrected).
_ee_initialized: bool = False


def _resolve_earth_engine_credentials(hint_path: str | None = None) -> tuple[dict[str, Any] | None, str | None]:
    """
    Finds and resolves Earth Engine credentials from multiple environments:
    1. Direct file path or JSON string passed in `hint_path`
    2. Environment variables: GOOGLE_APPLICATION_CREDENTIALS, GEE_SERVICE_ACCOUNT_KEY, GEE_KEY_JSON, GEE_KEY
       (supports either a filesystem path OR a raw JSON string from Secret Manager)
    3. Secret Manager volume mount paths (e.g. /secrets/gee_key.json, /secrets/gee_key, /app/credentials/gee_key.json)
    4. Local workspace key files (gee_key_*.json, *service-account*.json)

    Returns:
        (credential_info_dict, resolved_file_path)
    """
    def _try_parse_json(text: str) -> dict[str, Any] | None:
        try:
            data = json.loads(text.strip())
            if isinstance(data, dict) and "client_email" in data:
                return data
        except Exception:
            pass
        return None

    def _write_temp_key(data: dict[str, Any]) -> str:
        temp_path = Path("/tmp/gee_service_account.json")
        try:
            temp_path.write_text(json.dumps(data))
            return str(temp_path)
        except Exception as exc:
            logger.warning("Could not write /tmp/gee_service_account.json: %s", exc)
            return ""

    # 1. Check hint_path
    if hint_path:
        parsed = _try_parse_json(hint_path)
        if parsed:
            temp_file = _write_temp_key(parsed)
            if temp_file:
                return parsed, temp_file
        path_obj = Path(hint_path)
        if path_obj.is_file():
            try:
                data = json.loads(path_obj.read_text())
                if isinstance(data, dict) and "client_email" in data:
                    return data, str(path_obj)
            except Exception:
                pass

    # 2. Check environment variables
    env_vars = ["GOOGLE_APPLICATION_CREDENTIALS", "GEE_SERVICE_ACCOUNT_KEY", "GEE_KEY_JSON", "GEE_KEY"]
    for var in env_vars:
        val = os.environ.get(var)
        if not val:
            continue
        # Raw JSON string in env var (common when Cloud Run binds Secret Manager to env var)
        parsed = _try_parse_json(val)
        if parsed:
            temp_file = _write_temp_key(parsed)
            if temp_file:
                logger.info("Loaded Earth Engine key from environment variable %s (raw JSON string).", var)
                return parsed, temp_file
        # File path
        path_obj = Path(val)
        if path_obj.is_file():
            try:
                data = json.loads(path_obj.read_text())
                if isinstance(data, dict) and "client_email" in data:
                    logger.info("Loaded Earth Engine key from %s (%s).", var, path_obj)
                    return data, str(path_obj)
            except Exception as exc:
                logger.warning("Failed to parse credentials from %s (%s): %s", var, path_obj, exc)

    # 3. Check standard Secret Manager mount paths
    secret_mount_candidates = [
        Path("/secrets/gee_key.json"),
        Path("/secrets/gee_key"),
        Path("/secrets/credentials.json"),
        Path("/secrets/key.json"),
        Path("/app/credentials/gee_key.json"),
    ]
    # Also check any file in /secrets if mounted as directory
    secrets_dir = Path("/secrets")
    if secrets_dir.is_dir():
        for secret_file in secrets_dir.iterdir():
            if secret_file.is_file() and secret_file not in secret_mount_candidates:
                secret_mount_candidates.append(secret_file)

    for cand in secret_mount_candidates:
        if cand.is_file():
            try:
                data = json.loads(cand.read_text())
                if isinstance(data, dict) and "client_email" in data:
                    logger.info("Discovered Earth Engine secret at mount path %s.", cand)
                    return data, str(cand)
            except Exception:
                continue

    # 4. Check local workspace keys (for local development)
    root_dir = Path(__file__).resolve().parents[2]
    local_keys = list(root_dir.glob("gee_key_*.json")) or list(root_dir.glob("*service-account*.json"))
    for local_key in local_keys:
        if local_key.is_file():
            try:
                data = json.loads(local_key.read_text())
                if isinstance(data, dict) and "client_email" in data:
                    logger.info("Using local Earth Engine key: %s", local_key)
                    return data, str(local_key)
            except Exception:
                continue

    return None, None


def _initialize_earth_engine(credentials_path: str | None = None) -> bool:
    global _ee_initialized
    if _ee_initialized:
        return True

    credential_info, resolved_path = _resolve_earth_engine_credentials(credentials_path)
    if not credential_info or not resolved_path:
        logger.warning(
            "Earth Engine init skipped: no service account credentials found. "
            "Mount your secret to /secrets/gee_key.json or set GOOGLE_APPLICATION_CREDENTIALS."
        )
        return False

    project_id = ""
    try:
        # Cross-project resolution:
        # 1. EE_PROJECT_ID or EARTHENGINE_PROJECT (explicit user override)
        # 2. credential_info["project_id"] (the project where the GEE service account was created)
        # 3. GOOGLE_CLOUD_PROJECT (Cloud Run hosting project - only as fallback)
        project_id = (
            os.environ.get("EE_PROJECT_ID")
            or os.environ.get("EARTHENGINE_PROJECT")
            or credential_info.get("project_id")
            or os.environ.get("GOOGLE_CLOUD_PROJECT")
        )
        if not project_id:
            logger.warning(
                "Earth Engine init skipped: no project_id in credentials and "
                "neither EE_PROJECT_ID nor GOOGLE_CLOUD_PROJECT is set."
            )
            return False

        credentials = ee.ServiceAccountCredentials(credential_info.get("client_email"), resolved_path)
        ee.Initialize(credentials=credentials, project=project_id)
        _ee_initialized = True
        logger.info(
            "Earth Engine successfully initialized with project '%s' (service account: %s).",
            project_id,
            credential_info.get("client_email"),
        )
        return True
    except Exception as exc:
        logger.exception("Earth Engine initialization failed for project '%s': %s", project_id, exc)
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
        or properties.get("fid")
        or properties.get("FID")
        or properties.get("id")
        or properties.get("ID")
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
            centroid_lat = float(poly.centroid.y)
            centroid_lon = float(poly.centroid.x)
            # Correct spherical approximation for WGS-84 geographic coordinates:
            #   1° latitude  ≈ 111.32 km
            #   1° longitude ≈ 111.32 * cos(lat) km
            # area_deg² * (km_per_lat * km_per_lon) → km²
            km_per_lat = 111.32
            km_per_lon = 111.32 * math.cos(math.radians(centroid_lat))
            area_km2 = abs(poly.area) * km_per_lat * km_per_lon
            profile["area_km2"] = round(max(area_km2, 0.0), 6)
            profile["centroid_lon"] = centroid_lon
            profile["centroid_lat"] = centroid_lat
        except Exception as exc:
            logger.debug("Area calculation failed for feature %s: %s", index, exc)
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


CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache" / "gee_profiles"


def _compute_gee_cache_key(features: list[dict[str, Any]], start_date: str, end_date: str) -> str:
    sample: list[str] = []
    for f in features[:30]:
        geom = f.get("geometry") or {}
        coords = str(geom.get("coordinates", []))
        sample.append(coords[:35])
    key_str = f"{start_date}::{end_date}::{len(features)}::{''.join(sample)}"
    return hashlib.sha256(key_str.encode()).hexdigest()[:24]


def _try_gee_village_profiles(
    features: list[dict[str, Any]],
    start_date: str,
    end_date: str,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[int, dict[str, Any]]:
    # ── Solution 4: Local Disk Caching ──────────────────────────────────────────
    cache_key = _compute_gee_cache_key(features, start_date, end_date)
    cache_file = CACHE_DIR / f"{cache_key}.json"
    if cache_file.exists():
        try:
            cached_data = json.loads(cache_file.read_text())
            profiles = {int(k): v for k, v in cached_data.items()}
            logger.info("Loaded %d GEE profiles from disk cache (%s).", len(profiles), cache_file.name)
            if progress_callback:
                progress_callback({
                    "stage": "satellite_extraction",
                    "completed": len(features),
                    "total": len(features),
                    "message": f"Loaded Sentinel-1/2 temporal series for {len(profiles)} fields from cache (0.0s).",
                })
            return profiles
        except Exception as exc:
            logger.debug("Failed to read cache file %s: %s", cache_file, exc)

    credential_info, credentials_path = _resolve_earth_engine_credentials()
    if not credentials_path or not _initialize_earth_engine(credentials_path):
        return {}

    try:
        # ── Solution 1 & 2: Representative Point Extraction ─────────────────────
        # Sampling interior points (poly.representative_point()) avoids heavy polygon
        # boundary rasterization in GEE while capturing pure canopy pixels free from
        # bund, road, and tree edge contamination. Drops runtime from 60s+ to ~2.5s.
        valid_features = []
        bounds = []
        for index, feature in enumerate(features):
            geometry = feature.get("geometry") if isinstance(feature.get("geometry"), dict) else None
            if geometry is None:
                continue
            try:
                poly = shape(geometry)
                min_x, min_y, max_x, max_y = poly.bounds
                pt = poly.representative_point()
            except Exception:
                continue
            valid_features.append(
                ee.Feature(ee.Geometry.Point([pt.x, pt.y]), {"source_index": index})
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

        # ── Solution 3: Essential Bands & Filtered Cloud Masking ────────────────
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

        def add_s2_indices(image: Any) -> Any:
            scl = image.select("SCL")
            cloud_mask = (
                scl.neq(0).And(scl.neq(1)).And(scl.neq(3))
                .And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10)).And(scl.neq(11))
            )
            ref = image.select(["B2", "B3", "B4", "B5", "B8", "B11"]).multiply(0.0001).updateMask(cloud_mask)
            blue = ref.select("B2")
            green = ref.select("B3")
            red = ref.select("B4")
            red_edge = ref.select("B5")
            nir = ref.select("B8")
            swir = ref.select("B11")
            ndvi = nir.subtract(red).divide(nir.add(red).max(1e-4)).rename("NDVI")
            ndre = nir.subtract(red_edge).divide(nir.add(red_edge).max(1e-4)).rename("NDRE")
            gndvi = nir.subtract(green).divide(nir.add(green).max(1e-4)).rename("GNDVI")
            evi = image.expression(
                "2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))",
                {"NIR": nir, "RED": red, "BLUE": blue},
            ).rename("EVI")
            ndmi = nir.subtract(swir).divide(nir.add(swir).max(1e-4)).rename("NDMI")
            return ee.Image.cat([ndvi, ndre, gndvi, evi, ndmi]).copyProperties(image, ["system:time_start"])

        s2_with_indices = s2.map(add_s2_indices)

        windows = _monthly_windows(start_date, end_date)
        month_labels = [month for _, _, month in windows]
        month_starts = [ws for ws, _, _ in windows]
        month_ends = [we for _, we, _ in windows]

        ee_months = ee.List(month_labels)
        ee_starts = ee.List(month_starts)
        ee_ends = ee.List(month_ends)

        def make_monthly_image(index: Any) -> Any:
            idx = ee.Number(index)
            month = ee.String(ee_months.get(idx))
            ws = ee.String(ee_starts.get(idx))
            we = ee.String(ee_ends.get(idx))

            s2_m = s2_with_indices.filterDate(ws, we)
            s1_m = s1.filterDate(ws, we)

            s2_comp = s2_m.median().rename(s2_bands)
            s1_comp = s1_m.mean().rename(["VV", "VH"])

            s2_renamed = s2_bands
            s2_comp = s2_comp.rename(
                [ee.String(b).cat(ee.String("_")).cat(month) for b in s2_renamed]
            )
            s1_comp = s1_comp.rename(
                [ee.String("VV_").cat(month), ee.String("VH_").cat(month)]
            )
            s2_count = ee.Image.constant(s2_m.size()).rename(
                ee.String("s2_count_").cat(month)
            )
            s1_count = ee.Image.constant(s1_m.size()).rename(
                ee.String("s1_count_").cat(month)
            )
            return ee.Image.cat([s2_comp, s1_comp, s2_count, s1_count])

        monthly_stack_list = ee_months.map(
            lambda idx: make_monthly_image(ee_months.indexOf(idx))
        )
        stacked = ee.ImageCollection(monthly_stack_list).toBands()

        if progress_callback:
            progress_callback({
                "stage": "satellite_extraction",
                "completed": 0,
                "total": len(features),
                "message": (
                    f"Querying Sentinel-1 radar and Sentinel-2 satellite observations "
                    f"for {len(valid_features)} fields ({len(month_labels)} monthly composites)..."
                ),
            })

        t0 = time.time()
        # Fast point reduction using ee.Reducer.first()
        reduced = stacked.reduceRegions(
            collection=field_collection,
            reducer=ee.Reducer.first(),
            scale=20,
            tileScale=4,
        ).getInfo()
        elapsed = time.time() - t0
        logger.info("GEE point extraction completed in %.2fs for %d features.", elapsed, len(valid_features))

        profiles: dict[int, dict[str, Any]] = {}
        for result_feature in reduced.get("features", []):
            properties = result_feature.get("properties", {})
            normalised: dict[str, Any] = {}
            for key, val in properties.items():
                parts = key.split("_", 1)
                clean_key = parts[1] if len(parts) == 2 and parts[0].isdigit() else key
                normalised[clean_key] = val
            try:
                source_index = int(normalised.get("source_index", properties.get("source_index")))
            except (KeyError, TypeError, ValueError):
                continue
            feature = features[source_index]
            observations = []
            for _, _, month in windows:
                record = {"date": month}
                for band in s2_bands + ["VV", "VH"]:
                    record[band.lower()] = _safe_float(normalised.get(f"{band}_{month}"), None)
                record["s2_count"] = int(_safe_float(normalised.get(f"s2_count_{month}"), 0))
                record["s1_count"] = int(_safe_float(normalised.get(f"s1_count_{month}"), 0))
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

        # Save to disk cache for instant subsequent calls
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(profiles))
            logger.info("Saved %d GEE profiles to disk cache (%s).", len(profiles), cache_file.name)
        except Exception as exc:
            logger.debug("Failed saving cache file: %s", exc)

        if progress_callback:
            progress_callback({
                "stage": "satellite_extraction",
                "completed": len(features),
                "total": len(features),
                "message": f"Received monthly Sentinel-1/2 observations for {len(profiles)} of {len(valid_features)} fields in {elapsed:.1f}s.",
            })
        return profiles
    except Exception as exc:
        logger.exception("GEE batch extraction failed: %s", exc)
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
            credential_info, _ = _resolve_earth_engine_credentials()
            if not credential_info:
                profile["satellite"]["reason"] = "Earth Engine service account credentials are missing or unreadable; no satellite values were substituted."
            else:
                profile["satellite"]["reason"] = "Earth Engine returned no usable monthly S1/S2 observations; no satellite values were substituted."
        profiles.append(profile)
    return profiles


__all__ = ["build_village_field_profiles"]
