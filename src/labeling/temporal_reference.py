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


MASTER_FILE = ROOT_DIR / "MH_Crop_Master_2015_2026.xlsx"


def _parse_range(val: Any) -> float | None:
    if val is None or pd.isna(val):
        return None
    s = str(val).strip().replace("−", "-")
    parts = re.split(r"\s+to\s+|–", s)
    if len(parts) == 1:
        m = re.findall(r"[-+]?\d*\.?\d+", s)
        if len(m) >= 2 and "-" in s[1:]:
            parts = m[:2]
        else:
            parts = m
    try:
        nums = [float(p.strip().replace("DAT", "").replace("+", "")) for p in parts if p.strip()]
        if len(nums) >= 2:
            return (nums[0] + nums[1]) / 2.0
        elif len(nums) == 1:
            return nums[0]
    except Exception:
        pass
    return None


from functools import lru_cache


@lru_cache(maxsize=1)
def _load_master_crop_signatures() -> tuple[dict[str, Any], ...]:
    if not MASTER_FILE.exists():
        return ()
    try:
        df = pd.read_excel(MASTER_FILE, sheet_name="Crop_Index_Signatures", header=3)
        records: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            crop_raw = str(row.get("Crop", "")).strip()
            crop = canonicalize_crop_name(crop_raw)
            if not crop or crop == "unknown":
                continue
            das = _parse_range(row.get("DAS"))
            if das is None:
                continue
            for idx_col in ["NDVI", "EVI", "NDRE", "S1_VH_dB", "S1_VV_dB"]:
                val = _parse_range(row.get(idx_col))
                if val is not None:
                    clean_idx = idx_col.lower().replace("s1_", "").replace("_db", "")
                    records.append({
                        "crop": crop,
                        "study": "MH_Crop_Master",
                        "study_location": "Maharashtra",
                        "study_year": "2015-2026",
                        "measurement_method": "satellite_composite_signature",
                        "index": clean_idx,
                        "crop_age": das,
                        "age_unit": "days",
                        "observation_date": "2024-01-01",
                        "index_value": val,
                        "standard_deviation": None,
                        "source_url": "MH_Crop_Master_2015_2026.xlsx",
                        "notes": f"Stage: {row.get('Stage')}; Key_Cue: {row.get('Key_Cue')}",
                    })
        return tuple(records)
    except Exception:
        return ()


@lru_cache(maxsize=4)
def _load_crop_signature_reference_cached(csv_path_str: str | None = None) -> tuple[dict[str, Any], ...]:
    path = Path(csv_path_str) if csv_path_str is not None else EXACT_SERIES_PATH

    if not path.exists():
        raise FileNotFoundError(f"Temporal reference file not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        return ()

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
    base_records = normalized[
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

    # Exclude proximal leaf nitrogen trial observations (Raja et al. 2026, NDRE 0.03-0.13)
    # which severely distort canopy-level Sentinel satellite temporal matching.
    base_records = [
        r for r in base_records
        if not (r["crop"] == "cotton" and "Raja et al" in str(r.get("study", "")))
    ]

    master_records = _load_master_crop_signatures() if csv_path_str is None else ()
    return tuple(base_records) + master_records


def load_crop_signature_reference(csv_path: str | Path | None = None) -> list[dict[str, Any]]:
    path_key = str(csv_path) if csv_path is not None else None
    return list(_load_crop_signature_reference_cached(path_key))



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
