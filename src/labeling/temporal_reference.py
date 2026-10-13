from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
EXACT_SERIES_PATH = ROOT_DIR / "Crop_Index_Exact_Temporal_Points.csv"
RANGE_SERIES_PATH = ROOT_DIR / "Crop_Index_Range_Anchors_Not_Point_Series.csv"

CROP_CANONICAL_MAP = {
    "paddy rice": "paddy",
    "paddy / rice": "paddy",
    "rice": "paddy",
    "paddy": "paddy",
    "jowar / sorghum": "jowar",
    "jowar": "jowar",
    "sorghum": "jowar",
    "gram / chickpea": "gram",
    "gram": "gram",
    "chickpea": "gram",
    "rapeseed & mustard": "mustard",
    "rapeseed and mustard": "mustard",
    "mustard": "mustard",
    "tur / arhar / pigeon pea": "tur",
    "tur": "tur",
    "arhar": "tur",
    "pigeon pea": "tur",
    "soybean": "soybean",
    "soyabean": "soybean",
    "maize": "maize",
    "corn": "maize",
    "cotton": "cotton",
    "sugarcane": "sugarcane",
    "wheat": "wheat",
    "onion": "onion",
    "chilli": "chilli",
    "chili": "chilli",
    "groundnut": "groundnut",
    "peanut": "groundnut",
    "turmeric": "turmeric",
    "banana": "banana",
    "mango": "mango",
    "sapota": "sapota",
    "orange": "orange",
    "bajra": "bajra",
    "pearl millet": "bajra",
    "mung": "mung",
    "urad": "urad",
    "black gram": "urad",
    "blackgram": "urad",
    "mixed crop": "mixed_crop",
    "mixed_crop": "mixed_crop",
    "fallow": "fallow",
    "other crop": "other_crop",
    "other_crop": "other_crop",
}


def canonicalize_crop_name(value: Any) -> str:
    if value is None or pd.isna(value):
        return "unknown"

    text = str(value).strip().lower()
    text = text.replace("&", " and ")
    text = text.replace("/", " ")
    text = text.replace("-", " ")
    text = text.replace("(", " ").replace(")", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return "unknown"

    if text in CROP_CANONICAL_MAP:
        return CROP_CANONICAL_MAP[text]

    for alias, canonical in CROP_CANONICAL_MAP.items():
        if alias in text:
            return canonical

    return text


def load_crop_signature_reference(csv_path: str | Path | None = None) -> list[dict[str, Any]]:
    path = Path(csv_path) if csv_path is not None else EXACT_SERIES_PATH

    if not path.exists():
        raise FileNotFoundError(f"Temporal reference file not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        return []

    required = {"crop", "index", "crop_age", "age_unit", "index_value", "observation_date"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Temporal reference file is missing required columns: {missing}")

    normalized = df.copy()
    normalized["crop"] = normalized["crop"].apply(canonicalize_crop_name)
    normalized["index"] = normalized["index"].fillna("").astype(str).str.strip().str.lower()
    normalized["index_value"] = pd.to_numeric(normalized["index_value"], errors="coerce")
    normalized = normalized.dropna(subset=["crop", "index_value"]).copy()
    normalized = normalized[normalized["crop"] != "unknown"].copy()
    return normalized[
        [
            "crop",
            "study",
            "study_location",
            "study_year",
            "measurement_method",
            "index",
            "crop_age",
            "age_unit",
            "observation_date",
            "index_value",
            "standard_deviation",
            "source_url",
            "notes",
        ]
    ].to_dict(orient="records")


def build_crop_signature_summary(reference_records: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    records = reference_records if reference_records is not None else load_crop_signature_reference()
    if not records:
        return []

    frame = pd.DataFrame(records)
    if frame.empty:
        return []

    frame["index_value"] = pd.to_numeric(frame.get("index_value"), errors="coerce")
    frame = frame.dropna(subset=["index_value"]).copy()

    summary = (
        frame.groupby(["crop", "index"], as_index=False)
        .agg(
            mean_value=("index_value", "mean"),
            min_value=("index_value", "min"),
            max_value=("index_value", "max"),
            sample_count=("index_value", "count"),
            avg_crop_age=("crop_age", lambda s: pd.to_numeric(s, errors="coerce").mean()),
        )
        .sort_values(["crop", "index"])
        .reset_index(drop=True)
    )
    return summary.to_dict(orient="records")


def load_crop_evidence_index() -> dict[str, dict[str, Any]]:
    summary = build_crop_signature_summary()
    by_crop: dict[str, dict[str, Any]] = {}
    for row in summary:
        crop = str(row["crop"]).strip()
        by_crop.setdefault(crop, {"indices": []})
        by_crop[crop]["indices"].append({
            "index": row["index"],
            "mean_value": float(row["mean_value"]),
            "min_value": float(row["min_value"]),
            "max_value": float(row["max_value"]),
            "sample_count": int(row["sample_count"]),
        })
    return by_crop


__all__ = [
    "ROOT_DIR",
    "EXACT_SERIES_PATH",
    "RANGE_SERIES_PATH",
    "canonicalize_crop_name",
    "load_crop_signature_reference",
    "build_crop_signature_summary",
    "load_crop_evidence_index",
]
