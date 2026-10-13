from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.labeling.temporal_reference import canonicalize_crop_name, load_crop_signature_reference


CROP_GROUPS = {
    "kharif": {"paddy", "soybean", "cotton", "jowar", "bajra", "tur", "groundnut", "maize", "chilli"},
    "rabi": {"wheat", "gram", "mustard", "onion"},
    "perennial": {"sugarcane", "banana", "mango", "sapota", "orange", "turmeric"},
    "mixed": {"mixed_crop", "fallow", "other_crop"},
}

TEMPORAL_BANDS = ("ndvi", "ndre", "gndvi", "evi", "ndmi", "vv", "vh")
EVIDENCE_INDICES = ("ndvi", "ndre", "gndvi")
MIN_TEMPORAL_MATCH_SCORE = 0.5


def _safe_numeric(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _reference_crop_stats(reference_records: list[dict[str, Any]] | None = None) -> dict[str, dict[str, float]]:
    records = reference_records if reference_records is not None else load_crop_signature_reference()
    if not records:
        return {}

    frame = pd.DataFrame(records)
    if frame.empty:
        return {}

    frame["crop"] = frame["crop"].apply(canonicalize_crop_name)
    frame["index"] = frame["index"].fillna("").astype(str).str.lower()
    frame["index_value"] = pd.to_numeric(frame.get("index_value"), errors="coerce")
    frame = frame.dropna(subset=["crop", "index_value"]).copy()

    summary: dict[str, dict[str, float]] = {}
    for crop, crop_group in frame.groupby("crop"):
        crop_summary: dict[str, float] = {}
        for index_name, index_group in crop_group.groupby("index"):
            values = pd.to_numeric(index_group["index_value"], errors="coerce")
            crop_summary[index_name] = float(values.mean())
        summary[crop] = crop_summary
    return summary


def build_temporal_signature_features(field_df: pd.DataFrame, reference_records: list[dict[str, Any]] | None = None) -> pd.DataFrame:
    if field_df.empty:
        return field_df.copy()

    features = field_df.copy()
    ref_stats = _reference_crop_stats(reference_records)

    def _signature_for_row(row: pd.Series) -> dict[str, Any]:
        crop_label = canonicalize_crop_name(row.get("crop_label") or row.get("crop") or row.get("dominant_top_crop"))
        crop_template = ref_stats.get(crop_label, {})

        peak_ndvi = crop_template.get("ndvi", 0.5)
        peak_ndre = crop_template.get("ndre", 0.35)
        peak_gndvi = crop_template.get("gndvi", 0.45)
        peak_timing = float(row.get("signature_peak_timing", 0.0) or 0.0)

        if crop_template:
            peak_ndvi = crop_template.get("ndvi", peak_ndvi)
            peak_ndre = crop_template.get("ndre", peak_ndre)
            peak_gndvi = crop_template.get("gndvi", peak_gndvi)
            peak_timing = max(30.0, float(pd.Series([int(x) for x in row.get("crop_age", [0]) if isinstance(x, (int, float))]).mean()) if row.get("crop_age") is not None else 75.0)

        season = "unknown"
        for season_name, members in CROP_GROUPS.items():
            if crop_label in members:
                season = season_name
                break

        if crop_label == "unknown":
            peak_ndvi = 0.45
            peak_ndre = 0.28
            peak_gndvi = 0.4
            peak_timing = 75.0
            season = "mixed"

        similarity_profile = {
            "ndvi": round(peak_ndvi, 4),
            "ndre": round(peak_ndre, 4),
            "gndvi": round(peak_gndvi, 4),
            "season": season,
        }

        return {
            "crop_label_normalized": crop_label,
            "signature_peak_ndvi": round(peak_ndvi, 4),
            "signature_peak_ndre": round(peak_ndre, 4),
            "signature_peak_gndvi": round(peak_gndvi, 4),
            "signature_peak_timing": round(peak_timing, 2),
            "signature_season": season,
            "similarity_profile": similarity_profile,
        }

    signature_rows = features.apply(_signature_for_row, axis=1, result_type="expand")
    features = pd.concat([features, signature_rows], axis=1)
    return features


def match_field_to_crop_reference(field: dict[str, Any], reference_records: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    ref_stats = _reference_crop_stats(reference_records)
    if not ref_stats:
        return [{"crop": "unknown", "score": 0.0, "evidence": {}}]

    crop_name = canonicalize_crop_name(field.get("crop_label") or field.get("crop") or field.get("dominant_top_crop"))
    candidates: list[dict[str, Any]] = []

    for crop, stats in ref_stats.items():
        ndvi = _safe_numeric(stats.get("ndvi", 0.5))
        ndre = _safe_numeric(stats.get("ndre", 0.35))
        gndvi = _safe_numeric(stats.get("gndvi", 0.45))

        score = 0.2
        if crop == crop_name:
            score += 0.6
        elif crop_name in CROP_GROUPS.get("kharif", set()) and crop in CROP_GROUPS.get("kharif", set()):
            score += 0.25
        elif crop_name in CROP_GROUPS.get("rabi", set()) and crop in CROP_GROUPS.get("rabi", set()):
            score += 0.25
        elif crop_name in CROP_GROUPS.get("perennial", set()) and crop in CROP_GROUPS.get("perennial", set()):
            score += 0.25

        score += min(0.2, abs(ndvi - 0.5) * 0.35)
        score += min(0.15, abs(ndre - 0.3) * 0.5)
        score += min(0.15, abs(gndvi - 0.4) * 0.4)
        score = max(0.0, min(1.0, score))

        candidates.append({
            "crop": crop,
            "score": round(score, 4),
            "evidence": {
                "ndvi": ndvi,
                "ndre": ndre,
                "gndvi": gndvi,
            },
        })

    candidates.sort(key=lambda item: (-item["score"], item["crop"]))
    return candidates[:5]


def summarize_temporal_cluster_profiles(field_df: pd.DataFrame, cluster_col: str = "cluster") -> pd.DataFrame:
    if field_df.empty:
        return pd.DataFrame(columns=["cluster", "cluster_count", "mean_peak_ndvi", "mean_peak_ndre", "mean_peak_gndvi", "dominant_signature_crop", "dominant_signature_season", "candidate_label", "crop_candidates"])

    features = field_df.copy()
    if "crop_label_normalized" not in features.columns:
        features["crop_label_normalized"] = features["crop_label"].apply(canonicalize_crop_name)
    if "signature_peak_ndvi" not in features.columns:
        features = build_temporal_signature_features(features)

    cluster_summary = (
        features.groupby(cluster_col, dropna=False)
        .agg(
            cluster_count=("field_id", "count"),
            mean_peak_ndvi=("signature_peak_ndvi", "mean"),
            mean_peak_ndre=("signature_peak_ndre", "mean"),
            mean_peak_gndvi=("signature_peak_gndvi", "mean"),
            dominant_signature_crop=("crop_label_normalized", lambda s: s.mode().iloc[0] if not s.empty else "unknown"),
            dominant_signature_season=("signature_season", lambda s: s.mode().iloc[0] if not s.empty else "unknown"),
        )
        .reset_index()
    )

    cluster_summary["candidate_label"] = "unknown"
    cluster_summary["crop_candidates"] = [[] for _ in range(len(cluster_summary))]

    for idx, row in cluster_summary.iterrows():
        profile = {
            "cluster": row[cluster_col],
            "dominant_signature_crop": row["dominant_signature_crop"],
            "mean_peak_ndvi": row["mean_peak_ndvi"],
            "mean_peak_ndre": row["mean_peak_ndre"],
            "mean_peak_gndvi": row["mean_peak_gndvi"],
            "dominant_signature_season": row["dominant_signature_season"],
        }
        label_info = label_temporal_cluster_profile(profile)
        cluster_summary.at[idx, "candidate_label"] = label_info["label"]
        cluster_summary.at[idx, "crop_candidates"] = label_info["crop_candidates"]

    return cluster_summary


def label_temporal_cluster_profile(profile: dict[str, Any]) -> dict[str, Any]:
    dominant_crop = str(profile.get("dominant_signature_crop") or profile.get("crop_label") or "unknown").strip()
    season = str(profile.get("dominant_signature_season") or profile.get("season") or "mixed").strip().lower()
    mean_ndvi = _safe_numeric(profile.get("mean_peak_ndvi"), 0.5)
    mean_ndre = _safe_numeric(profile.get("mean_peak_ndre"), 0.35)
    mean_gndvi = _safe_numeric(profile.get("mean_peak_gndvi"), 0.45)

    field_like = {
        "crop_label": dominant_crop,
        "signature_peak_ndvi": mean_ndvi,
        "signature_peak_ndre": mean_ndre,
        "signature_peak_gndvi": mean_gndvi,
    }
    ranked = match_field_to_crop_reference(field_like)

    best_crop = ranked[0]["crop"] if ranked else "unknown"
    crop_candidates = [entry["crop"] for entry in ranked[:3]]

    if best_crop == "unknown":
        label = "mixed temporal cluster"
    elif season in {"kharif", "rabi", "perennial"}:
        label = f"{best_crop}-dominant temporal cluster"
    else:
        label = f"mixed {best_crop} temporal cluster"

    return {
        "label": label,
        "best_crop": best_crop,
        "crop_candidates": crop_candidates,
        "season": season,
        "ranked_matches": ranked,
    }


def _numeric_series_matrix(field_profiles: list[dict[str, Any]]) -> tuple[list[str], list[str], np.ndarray]:
    field_ids = []
    records_by_field = []
    all_months: set[str] = set()
    for profile in field_profiles:
        satellite = profile.get("satellite") or {}
        records = satellite.get("timeseries") or []
        if satellite.get("source") != "gee_sentinel_monthly" or not records:
            continue
        field_ids.append(str(profile.get("field_id", f"field_{len(field_ids) + 1}")))
        records_by_field.append(records)
        all_months.update(str(record.get("date")) for record in records if record.get("date"))

    months = sorted(all_months)
    if not field_ids or not months:
        return [], months, np.empty((0, 0), dtype=float)

    matrix = []
    for records in records_by_field:
        indexed = {str(record.get("date")): record for record in records}
        row = []
        for month in months:
            for band in TEMPORAL_BANDS:
                try:
                    value = float(indexed[month].get(band))
                    if not math.isfinite(value):
                        value = np.nan
                except (KeyError, TypeError, ValueError):
                    value = np.nan
                row.append(value)
        matrix.append(row)

    values = np.asarray(matrix, dtype=float)
    for column in range(values.shape[1]):
        series = pd.Series(values[:, column])
        values[:, column] = series.fillna(series.median()).fillna(0.0).to_numpy()
    return field_ids, months, values


def cluster_field_time_series(field_profiles: list[dict[str, Any]]) -> dict[str, Any]:
    field_ids, months, raw_matrix = _numeric_series_matrix(field_profiles)
    if not field_ids:
        return {"assignments": {}, "months": months, "cluster_count": 0, "feature_matrix": {}}

    temporal_features = StandardScaler().fit_transform(raw_matrix)
    profiles_by_id = {str(profile.get("field_id")): profile for profile in field_profiles}
    coordinates = np.asarray([
        [
            _safe_numeric(profiles_by_id[field_id].get("centroid_lon"), np.nan),
            _safe_numeric(profiles_by_id[field_id].get("centroid_lat"), np.nan),
        ]
        for field_id in field_ids
    ], dtype=float)
    for column in range(coordinates.shape[1]):
        series = pd.Series(coordinates[:, column])
        coordinates[:, column] = series.fillna(series.median()).fillna(0.0).to_numpy()
    spatial_features = StandardScaler().fit_transform(coordinates)
    spatial_weight = np.sqrt(temporal_features.shape[1]) * 0.35
    standardized = np.column_stack([temporal_features, spatial_features * spatial_weight])
    unique_count = len(np.unique(np.round(standardized, decimals=8), axis=0))
    if len(field_ids) < 2 or unique_count < 2:
        labels = np.zeros(len(field_ids), dtype=int)
        cluster_count = 1
    else:
        candidate_ks = range(2, min(8, len(field_ids) - 1, unique_count - 1) + 1)
        best_score = -1.0
        labels = None
        for k in candidate_ks:
            candidate = KMeans(n_clusters=k, n_init=20, random_state=42).fit_predict(standardized)
            score = silhouette_score(standardized, candidate)
            if score > best_score:
                best_score = score
                labels = candidate
        if labels is None:
            labels = KMeans(n_clusters=2, n_init=20, random_state=42).fit_predict(standardized)
        cluster_count = len(set(int(label) for label in labels))

    return {
        "assignments": {field_id: int(label) for field_id, label in zip(field_ids, labels)},
        "months": months,
        "cluster_count": cluster_count,
        "feature_matrix": {
            field_id: raw_matrix[index].tolist()
            for index, field_id in enumerate(field_ids)
        },
    }


def _crop_age_days(value: Any, unit: Any) -> float | None:
    match = re.search(r"\d+(?:\.\d+)?", str(value or ""))
    if not match:
        return None
    age = float(match.group())
    unit_text = f"{value or ''} {unit or ''}".lower()
    if "week" in unit_text:
        return age * 7
    if "month" in unit_text:
        return age * 30.44
    if "year" in unit_text:
        return age * 365.25
    return age


def rank_crops_for_temporal_series(
    timeseries: list[dict[str, Any]],
    reference_records: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    records = reference_records if reference_records is not None else load_crop_signature_reference()
    if not timeseries or not records:
        return []

    reference = pd.DataFrame(records)
    reference["crop"] = reference["crop"].apply(canonicalize_crop_name)
    reference["index"] = reference["index"].fillna("").astype(str).str.lower()
    reference["index_value"] = pd.to_numeric(reference["index_value"], errors="coerce")
    reference["age_days"] = reference.apply(lambda row: _crop_age_days(row.get("crop_age"), row.get("age_unit")), axis=1)
    reference = reference[
        reference["index"].isin(EVIDENCE_INDICES)
        & reference["age_days"].notna()
        & reference["index_value"].notna()
    ]
    if reference.empty:
        return []

    observed = []
    for record in timeseries:
        try:
            month = pd.Timestamp(record["date"])
        except (KeyError, TypeError, ValueError):
            continue
        point = {"date": month}
        for index_name in EVIDENCE_INDICES:
            try:
                point[index_name] = float(record[index_name])
            except (KeyError, TypeError, ValueError):
                point[index_name] = np.nan
        observed.append(point)
    if len(observed) < 2:
        return []

    observed_frame = pd.DataFrame(observed).sort_values("date").reset_index(drop=True)
    candidates = []
    for crop, crop_rows in reference.groupby("crop"):
        curves = {}
        for index_name, index_rows in crop_rows.groupby("index"):
            curve = index_rows.groupby("age_days")["index_value"].mean().sort_index()
            if len(curve) >= 2:
                curves[index_name] = curve
        if not curves:
            continue

        best = None
        for planting_lag in range(13):
            errors = []
            for point_index, point in observed_frame.iterrows():
                age_days = (point_index - planting_lag) * 30.44
                if age_days < 0:
                    continue
                for index_name, curve in curves.items():
                    observed_value = point[index_name]
                    if pd.isna(observed_value) or age_days < curve.index.min() or age_days > curve.index.max():
                        continue
                    observed_value = float(observed_value)
                    if not math.isfinite(observed_value):
                        continue
                    expected = float(np.interp(age_days, curve.index.to_numpy(dtype=float), curve.to_numpy(dtype=float)))
                    errors.append((observed_value - expected) ** 2)
            if len(errors) >= 2:
                rmse = math.sqrt(sum(errors) / len(errors))
                if best is None or rmse < best["rmse"]:
                    best = {"rmse": rmse, "matched_points": len(errors), "planting_lag_months": planting_lag}

        if best is not None:
            coverage = min(1.0, best["matched_points"] / 6.0)
            score = math.exp(-best["rmse"] / 0.15) * coverage
            candidates.append({
                "crop": crop,
                "score": round(score, 4),
                "evidence": {
                    "rmse": round(best["rmse"], 4),
                    "matched_points": best["matched_points"],
                    "planting_lag_months": best["planting_lag_months"],
                    "reference": "Crop_Index_Exact_Temporal_Points.csv",
                },
            })

    candidates.sort(key=lambda item: (-item["score"], item["crop"]))
    return candidates[:5]


__all__ = [
    "build_temporal_signature_features",
    "match_field_to_crop_reference",
    "summarize_temporal_cluster_profiles",
    "label_temporal_cluster_profile",
    "cluster_field_time_series",
    "rank_crops_for_temporal_series",
    "MIN_TEMPORAL_MATCH_SCORE",
]
