from __future__ import annotations

from typing import Any

from src.rules.rule_engine import load_crop_rules, score_cluster_profile


def _normalize_top_crop(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    return text.replace("-", " ").replace("/", " ").replace("(", " ").replace(")", " ")


def generate_crop_candidates(profile: dict) -> list[dict]:
    rules = load_crop_rules()
    candidates = []

    for crop_name, rule in rules.items():
        score = score_cluster_profile(profile, crop_name)
        dominant_crop = _normalize_top_crop(profile.get("dominant_top_crop") or profile.get("top_crop"))
        crop_aliases = [
            alias.lower().replace("-", " ").replace("/", " ").replace("(", " ").replace(")", " ")
            for alias in rule.get("aliases", [])
        ]

        if dominant_crop and dominant_crop in crop_aliases:
            score = min(1.0, score + 0.15)

        # extra signal from top crop share and season alignment
        top_share = float(profile.get("top_crop_share", 0.0) or 0.0)
        if top_share > 0.6 and dominant_crop in crop_aliases:
            score = min(1.0, score + 0.1)

        season = rule.get("season")
        kharif = float(profile.get("avg_kharif_share", 0.0) or 0.0)
        rabi = float(profile.get("avg_rabi_share", 0.0) or 0.0)
        other = float(profile.get("avg_other_share", 0.0) or 0.0)

        if season == "kharif" and kharif > 0.55:
            score = min(1.0, score + 0.05)
        if season == "rabi" and rabi > 0.45:
            score = min(1.0, score + 0.05)
        if season == "perennial" and other > 0.2:
            score = min(1.0, score + 0.05)

        candidates.append({
            "crop": crop_name,
            "score": round(score, 4),
            "season": season,
            "notes": rule.get("notes", ""),
        })

    candidates.sort(key=lambda x: (-x["score"], x["crop"]))
    return candidates


def classify_field_profile(profile: dict) -> dict:
    candidates = generate_crop_candidates(profile)
    best = candidates[0]

    dominant_crop = _normalize_top_crop(profile.get("dominant_top_crop") or profile.get("top_crop"))
    season_label = "mixed"
    kharif = float(profile.get("avg_kharif_share", 0.0) or 0.0)
    rabi = float(profile.get("avg_rabi_share", 0.0) or 0.0)
    other = float(profile.get("avg_other_share", 0.0) or 0.0)

    if kharif >= rabi and kharif >= other:
        season_label = "kharif"
    elif rabi >= kharif and rabi >= other:
        season_label = "rabi"
    elif other >= kharif and other >= rabi:
        season_label = "commercial_or_perennial"

    reason = (
        f"Dominant seasonal pattern is {season_label}; "
        f"top crop signal is {dominant_crop or 'not provided'}; "
        f"the highest-scoring crop candidate is {best['crop']}."
    )

    return {
        "best_crop": best["crop"],
        "best_score": best["score"],
        "season_label": season_label,
        "evidence_summary": {
            "field_id": profile.get("field_id"),
            "dominant_top_crop": profile.get("dominant_top_crop") or profile.get("top_crop"),
            "kharif_share": kharif,
            "rabi_share": rabi,
            "zaid_share": float(profile.get("avg_zaid_share", 0.0) or 0.0),
            "other_share": other,
            "reason": reason,
        },
        "ranked_candidates": candidates[:5],
    }
