from __future__ import annotations

import re
from pathlib import Path

import yaml

from src.config.settings import ROOT_DIR


from functools import lru_cache

DEFAULT_RULES_PATH = ROOT_DIR / "src" / "config" / "crop_rules.yaml"


def _normalize_crop_name(value: str | None) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    text = text.replace("&", " and ")
    text = text.replace("/", " ")
    text = text.replace("-", " ")
    text = text.replace("(", " ").replace(")", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _rule_aliases(rule: dict) -> list[str]:
    aliases = rule.get("aliases", []) or []
    return [_normalize_crop_name(alias) for alias in aliases]


@lru_cache(maxsize=4)
def load_crop_rules(path: str | Path | None = None) -> dict:
    if path is None:
        path = DEFAULT_RULES_PATH

    with open(path, "r", encoding="utf-8") as fh:
        rules = yaml.safe_load(fh) or {}

    normalized = {}
    for crop, config in rules.items():
        normalized[_normalize_crop_name(crop)] = config
    return normalized


def score_cluster_profile(profile: dict, crop_name: str) -> float:
    rules = load_crop_rules()
    crop_key = _normalize_crop_name(crop_name)
    if crop_key not in rules:
        return 0.0

    rule = rules[crop_key]
    kharif = float(profile.get("avg_kharif_share", 0.0) or 0.0)
    rabi = float(profile.get("avg_rabi_share", 0.0) or 0.0)
    zaid = float(profile.get("avg_zaid_share", 0.0) or 0.0)
    other = float(profile.get("avg_other_share", 0.0) or 0.0)
    dominant_crop = str(profile.get("dominant_top_crop", "")).strip()
    dominant_norm = _normalize_crop_name(dominant_crop)

    score = 0.0

    aliases = _rule_aliases(rule)

    if rule["season"] == "kharif":
        score += min(kharif, 1.0) * float(rule.get("season_weight", 0.35))
        if dominant_norm in aliases:
            score += 0.45
        if kharif >= rule.get("min_kharif_share", 0.0):
            score += 0.25
    elif rule["season"] == "rabi":
        score += min(rabi, 1.0) * float(rule.get("season_weight", 0.35))
        if dominant_norm in aliases:
            score += 0.45
        if rabi >= rule.get("min_rabi_share", 0.0):
            score += 0.25
    elif rule["season"] == "perennial":
        score += min(other, 1.0) * 0.35
        if dominant_norm in aliases:
            score += 0.45
        if other >= rule.get("min_other_share", 0.15):
            score += 0.25

    if dominant_norm in aliases:
        score += 0.15

    if "kharif" in rule["season"]:
        score = min(score, 0.99)
    elif "rabi" in rule["season"]:
        score = min(score, 0.99)
    elif rule["season"] == "perennial":
        score = min(score, 0.99)

    return round(max(0.0, min(1.0, score)), 4)


def classify_cluster_profile(profile: dict) -> dict:
    rules = load_crop_rules()
    ranked = []

    for crop_name, rule in rules.items():
        score = score_cluster_profile(profile, crop_name)
        ranked.append({
            "crop": crop_name,
            "season": rule.get("season"),
            "score": score,
            "notes": rule.get("notes", ""),
        })

    ranked.sort(key=lambda item: (-item["score"], item["crop"]))
    best_crop = ranked[0]["crop"] if ranked else None
    best_score = ranked[0]["score"] if ranked else 0.0

    return {
        "best_crop": best_crop,
        "best_score": best_score,
        "ranked_candidates": ranked[:5],
        "evidence_summary": {
            "kharif_share": float(profile.get("avg_kharif_share", 0.0) or 0.0),
            "rabi_share": float(profile.get("avg_rabi_share", 0.0) or 0.0),
            "zaid_share": float(profile.get("avg_zaid_share", 0.0) or 0.0),
            "other_share": float(profile.get("avg_other_share", 0.0) or 0.0),
            "dominant_top_crop": str(profile.get("dominant_top_crop", "")).strip(),
        },
    }
