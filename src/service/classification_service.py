from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Callable

import pandas as pd

from src.feature_engineering.sentinel_field_extractor import build_village_field_profiles
from src.ingestion.village_geojson import normalize_village_fields
from src.labeling.temporal_clustering import (
    CROP_GROUPS,
    MIN_TEMPORAL_MATCH_SCORE,
    build_temporal_signature_features,
    cluster_field_time_series,
    label_temporal_cluster_profile,
    match_field_to_crop_reference,
    rank_crops_for_temporal_series,
)
from src.labeling.temporal_reference import canonicalize_crop_name, load_crop_signature_reference
from src.rules.field_candidate_generator import classify_field_profile
from src.rules.village_classifier import classify_village

REQUEST_VERSION = "0.7.0"


def _season_from_profile(profile: dict[str, Any]) -> str:
    kharif = float(profile.get("avg_kharif_share", 0.0) or 0.0)
    rabi = float(profile.get("avg_rabi_share", 0.0) or 0.0)
    other = float(profile.get("avg_other_share", 0.0) or 0.0)
    if kharif >= rabi and kharif >= other:
        return "kharif"
    if rabi >= kharif and rabi >= other:
        return "rabi"
    if other >= kharif and other >= rabi:
        return "perennial"
    return "mixed"


def _enrich_profile_with_temporal_evidence(profile: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(profile)
    crop_label = (
        enriched.get("dominant_top_crop")
        or enriched.get("top_crop")
        or enriched.get("crop_label")
        or "unknown"
    )
    normalized_crop = canonicalize_crop_name(crop_label)

    base_row = {
        "field_id": enriched.get("field_id", "field_1"),
        "crop_label": crop_label,
        "dominant_top_crop": crop_label,
    }

    temporal_frame = pd.DataFrame([base_row])
    temporal_frame = build_temporal_signature_features(temporal_frame)
    row = temporal_frame.iloc[0].to_dict()

    ranked = match_field_to_crop_reference({
        "crop_label": normalized_crop,
        "signature_peak_ndvi": row.get("signature_peak_ndvi", 0.5),
        "signature_peak_ndre": row.get("signature_peak_ndre", 0.3),
        "signature_peak_gndvi": row.get("signature_peak_gndvi", 0.45),
    })

    label_info = label_temporal_cluster_profile({
        "dominant_signature_crop": row.get("crop_label_normalized", normalized_crop),
        "dominant_signature_season": row.get("signature_season", _season_from_profile(enriched)),
        "mean_peak_ndvi": row.get("signature_peak_ndvi", 0.5),
        "mean_peak_ndre": row.get("signature_peak_ndre", 0.3),
        "mean_peak_gndvi": row.get("signature_peak_gndvi", 0.45),
    })

    enriched["temporal_signature"] = row.get("similarity_profile", {})
    enriched["temporal_cluster_label"] = label_info.get("label", "mixed temporal cluster")
    enriched["temporal_ranked_candidates"] = ranked
    enriched["temporal_best_crop"] = label_info.get("best_crop", ranked[0]["crop"] if ranked else "unknown")
    return enriched


def _build_field_prediction(profile: dict[str, Any]) -> dict[str, Any]:
    profile = _enrich_profile_with_temporal_evidence(profile)
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
        "temporal_signature": profile.get("temporal_signature", {}),
        "temporal_cluster_label": profile.get("temporal_cluster_label"),
        "temporal_ranked_candidates": profile.get("temporal_ranked_candidates", []),
    }


def classify_village_request(
    payload: dict[str, Any],
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    village_id = payload.get("village_id", "unknown")
    fields = payload.get("fields", [])
    end_date = payload.get("end_date", date.today().isoformat())
    start_date = payload.get("start_date", (date.today() - timedelta(days=365)).isoformat())

    if isinstance(payload, dict) and "fields" not in payload and "type" in payload and payload.get("type") == "FeatureCollection":
        fields = payload

    field_profiles = build_village_field_profiles(
        fields,
        start_date=start_date,
        end_date=end_date,
        progress_callback=progress_callback,
    )
    if not field_profiles:
        field_profiles = normalize_village_fields(fields)

    field_predictions = [_build_field_prediction(profile) for profile in field_profiles]

    if progress_callback:
        progress_callback({"stage": "temporal_clustering", "completed": 0, "total": 1, "message": "Clustering fields by their monthly Sentinel trajectories."})
    temporal_clusters = cluster_field_time_series(field_profiles)
    satellite_profiles = {
        str(profile.get("field_id")): profile
        for profile in field_profiles
        if (profile.get("satellite") or {}).get("source") == "gee_sentinel_monthly"
    }
    complete_sensor_field_count = 0
    for profile in satellite_profiles.values():
        satellite = profile["satellite"]
        has_s2 = satellite.get("s2_observation_count", 0) > 0 or any(
            any(observation.get(band) is not None for band in ("ndvi", "ndre", "gndvi", "evi", "ndmi"))
            for observation in satellite.get("timeseries", [])
        )
        has_s1 = any(
            observation.get("vv") is not None or observation.get("vh") is not None
            for observation in satellite.get("timeseries", [])
        )
        complete_sensor_field_count += int(has_s1 and has_s2)
    for prediction in field_predictions:
        if str(prediction.get("field_id")) not in satellite_profiles:
            prediction["best_crop"] = "unknown"
            prediction["best_score"] = 0.0
            prediction["top_3_candidates"] = []
            prediction["temporal_cluster_label"] = "satellite data unavailable"
            prediction["satellite_source"] = "unavailable"
    cluster_members: dict[int, list[dict[str, Any]]] = {}
    for field_id, cluster_id in temporal_clusters["assignments"].items():
        profile = satellite_profiles.get(field_id)
        if profile is not None:
            cluster_members.setdefault(cluster_id, []).append(profile)

    evidence_records = load_crop_signature_reference() if satellite_profiles else []
    cluster_summary = []
    cluster_labels: dict[int, str] = {}
    if progress_callback:
        progress_callback({"stage": "evidence_labeling", "completed": 0, "total": max(1, len(cluster_members)), "message": "Matching temporal clusters to crop-age index evidence."})
    for cluster_id, members in sorted(cluster_members.items()):
        monthly: dict[str, dict[str, list[float]]] = {}
        for member in members:
            for observation in member["satellite"]["timeseries"]:
                month = str(observation.get("date"))
                monthly.setdefault(month, {})
                for band in ("ndvi", "ndre", "gndvi", "evi", "ndmi", "vv", "vh"):
                    try:
                        value = float(observation.get(band))
                    except (TypeError, ValueError):
                        continue
                    monthly[month].setdefault(band, []).append(value)
        cluster_series = []
        for month, bands in sorted(monthly.items()):
            record = {"date": month}
            for band, values in bands.items():
                record[band] = sum(values) / len(values)
            cluster_series.append(record)
        crop_candidates = rank_crops_for_temporal_series(cluster_series, evidence_records)
        best = crop_candidates[0] if crop_candidates else None
        has_confident_match = bool(best and best["score"] >= MIN_TEMPORAL_MATCH_SCORE)
        label = f"{best['crop']}-like temporal cluster" if has_confident_match else "unmatched temporal cluster"
        cluster_labels[cluster_id] = label
        cluster_summary.append({
            "cluster": cluster_id,
            "cluster_id": cluster_id,
            "candidate_label": label,
            "label": label,
            "cluster_count": len(members),
            "crop_candidates": crop_candidates[:3],
            "match_status": "matched" if has_confident_match else "insufficient_evidence",
            "evidence_basis": "monthly Sentinel-1/2 trajectory matched to crop-age index observations",
            "temporal_observations": cluster_series,
        })
        if progress_callback:
            progress_callback({"stage": "evidence_labeling", "completed": len(cluster_summary), "total": max(1, len(cluster_members)), "message": f"Matched evidence for cluster {cluster_id}."})

    if progress_callback:
        progress_callback({"stage": "temporal_clustering", "completed": 1, "total": 1, "message": f"Created {temporal_clusters['cluster_count']} temporal clusters."})

    for prediction in field_predictions:
        field_id = str(prediction.get("field_id"))
        profile = satellite_profiles.get(field_id)
        cluster_id = temporal_clusters["assignments"].get(field_id)
        if profile is None or cluster_id is None:
            continue
        series = profile["satellite"].get("timeseries", [])
        crop_candidates = rank_crops_for_temporal_series(series, evidence_records)
        label = cluster_labels.get(cluster_id, "unmatched temporal cluster")
        prediction["cluster_id"] = cluster_id
        prediction["cluster_label"] = label
        prediction["temporal_signature"] = series
        prediction["temporal_ranked_candidates"] = crop_candidates
        prediction["temporal_cluster_label"] = label
        prediction["satellite_source"] = profile["satellite"]["source"]
        has_confident_match = bool(crop_candidates and crop_candidates[0]["score"] >= MIN_TEMPORAL_MATCH_SCORE)
        if has_confident_match:
            prediction["best_crop"] = crop_candidates[0]["crop"]
            prediction["best_score"] = crop_candidates[0]["score"]
        else:
            prediction["best_crop"] = "unknown"
            prediction["best_score"] = 0.0
        prediction["top_3_candidates"] = [
            {
                "rank": rank,
                "crop": candidate["crop"],
                "score": candidate["score"],
                "season": next((season for season, crops in CROP_GROUPS.items() if candidate["crop"] in crops), "mixed"),
                "evidence": candidate["evidence"],
            }
            for rank, candidate in enumerate(crop_candidates[:3], start=1)
        ]

    satellite_field_count = len(satellite_profiles)
    if complete_sensor_field_count == len(field_profiles) and complete_sensor_field_count:
        satellite_status = "complete"
    elif satellite_field_count:
        satellite_status = "partial"
    else:
        satellite_status = "unavailable"

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
        "cluster_summary": cluster_summary if satellite_field_count else [],
        "field_predictions": field_predictions,
        "satellite_status": satellite_status,
        "temporal_period": {"start_date": start_date, "end_date": end_date},
        "satellite_field_count": satellite_field_count,
        "complete_sensor_field_count": complete_sensor_field_count,
    }
    if satellite_field_count and cluster_summary:
        dominant_cluster = max(cluster_summary, key=lambda item: item["cluster_count"])
        best_cluster_candidate = dominant_cluster["crop_candidates"][0] if dominant_cluster["crop_candidates"] else None
        response["dominant_crop"] = (
            best_cluster_candidate["crop"]
            if best_cluster_candidate and best_cluster_candidate["score"] >= MIN_TEMPORAL_MATCH_SCORE
            else dominant_cluster["label"]
        )
        response["dominant_share"] = dominant_cluster["cluster_count"] / max(len(field_profiles), 1)
        matched_counts: dict[str, list[float]] = {}
        for prediction in field_predictions:
            crop = prediction.get("best_crop")
            if not crop or crop == "unknown":
                continue
            matched_counts.setdefault(crop, []).append(float(prediction.get("best_score", 0.0)))
        response["crop_mix"] = [
            {
                "crop": crop,
                "count": len(scores),
                "share": round(len(scores) / max(len(field_predictions), 1), 4),
                "average_score": round(sum(scores) / len(scores), 4),
            }
            for crop, scores in sorted(matched_counts.items(), key=lambda item: (-len(item[1]), item[0]))
        ]
        response["top_candidates"] = response["crop_mix"][:5]
        response["summary"] = {
            "reason": "Village crop evidence is aggregated from monthly Sentinel time-series matches to crop-age reference curves.",
            "dominant_crop": response["dominant_crop"],
            "field_count": len(field_predictions),
        }
    elif not satellite_field_count:
        response["dominant_crop"] = "satellite data unavailable"
        response["dominant_share"] = 0.0
        response["crop_mix"] = []
        response["top_candidates"] = []
        response["summary"] = {
            "reason": "No Sentinel time series was available; crop labels were not inferred from polygon metadata.",
            "dominant_crop": "satellite data unavailable",
            "field_count": len(field_predictions),
        }
    if progress_callback:
        progress_callback({"stage": "complete", "completed": 1, "total": 1, "message": "Classification and cluster evidence are ready."})
    return response
