from __future__ import annotations

from collections import Counter
from typing import Any

from src.rules.field_candidate_generator import classify_field_profile
from src.rules.rule_engine import load_crop_rules, score_cluster_profile


def _normalize_crop(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    return text.replace("-", " ").replace("/", " ").replace("(", " ").replace(")", " ").strip()


def summarize_village_predictions(field_profiles: list[dict]) -> dict:
    field_count = max(len(field_profiles), 1)
    crop_counts: Counter[str] = Counter()
    crop_scores: dict[str, float] = {}

    for profile in field_profiles:
        result = classify_field_profile(profile)
        crop = _normalize_crop(result["best_crop"])
        crop_counts[crop] += 1
        crop_scores[crop] = crop_scores.get(crop, 0.0) + float(result["best_score"])

    if not crop_counts:
        return {
            "dominant_crop": "unknown",
            "dominant_share": 0.0,
            "crop_mix": [],
            "field_count": 0,
        }

    total_votes = sum(crop_counts.values())
    crop_mix = []
    for crop, count in sorted(crop_counts.items(), key=lambda item: (-item[1], item[0])):
        crop_mix.append({
            "crop": crop,
            "count": count,
            "share": round(count / total_votes, 4),
            "average_score": round(crop_scores.get(crop, 0.0) / max(count, 1), 4),
        })

    dominant_crop = crop_mix[0]["crop"]
    dominant_share = crop_mix[0]["share"]

    return {
        "dominant_crop": dominant_crop,
        "dominant_share": dominant_share,
        "crop_mix": crop_mix,
        "field_count": field_count,
    }


def classify_village(village: dict) -> dict:
    field_profiles = village.get("fields", [])
    summary = summarize_village_predictions(field_profiles)

    dominant = summary["dominant_crop"]
    reason = (
        f"Village-level classification is dominated by {dominant}; "
        f"this crop accounts for about {summary['dominant_share'] * 100:.1f}% of the field-level best matches."
    )

    top_candidates = sorted(summary["crop_mix"], key=lambda x: (-x["share"], -x["average_score"], x["crop"]))[:5]
    if len(top_candidates) < 2:
        aggregated = {
            "dominant_top_crop": dominant,
            "avg_kharif_share": sum(float(p.get("avg_kharif_share", 0.0) or 0.0) for p in field_profiles) / max(len(field_profiles), 1),
            "avg_rabi_share": sum(float(p.get("avg_rabi_share", 0.0) or 0.0) for p in field_profiles) / max(len(field_profiles), 1),
            "avg_zaid_share": sum(float(p.get("avg_zaid_share", 0.0) or 0.0) for p in field_profiles) / max(len(field_profiles), 1),
            "avg_other_share": sum(float(p.get("avg_other_share", 0.0) or 0.0) for p in field_profiles) / max(len(field_profiles), 1),
            "top_crop_share": max(float(p.get("top_crop_share", 0.0) or 0.0) for p in field_profiles) if field_profiles else 0.0,
        }
        fallback = []
        rules = load_crop_rules()
        for crop_name in rules:
            score = score_cluster_profile(aggregated, crop_name)
            if crop_name == dominant:
                continue
            fallback.append({
                "crop": crop_name,
                "share": round(score, 4),
                "average_score": round(score, 4),
            })
        fallback.sort(key=lambda x: (-x["share"], x["crop"]))
        top_candidates = (top_candidates + fallback[: max(1, 5 - len(top_candidates))])[:5]

    return {
        "village_id": village.get("village_id"),
        "field_count": summary["field_count"],
        "dominant_crop": dominant,
        "dominant_share": summary["dominant_share"],
        "summary": {
            "reason": reason,
            "dominant_crop": dominant,
            "field_count": summary["field_count"],
        },
        "top_candidates": top_candidates,
        "crop_mix": summary["crop_mix"],
    }
