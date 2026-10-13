from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from src.config.settings import LABELS_DIR, PROCESSED_DIR


REFERENCE_FILES = {
    "exact": "Crop_Index_Exact_Temporal_Points.csv",
    "range": "Crop_Index_Range_Anchors_Not_Point_Series.csv",
}


def _normalize_crop_name(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    text = text.replace("&", " and ")
    text = text.replace("/", " ")
    text = text.replace("-", " ")
    text = re.sub(r"\s+", " ", text)
    replacements = {
        "rice": "paddy",
        "paddy rice": "paddy",
        "paddy": "paddy",
        "jowar sorghum": "jowar",
        "sorghum": "jowar",
        "gram chickpea": "gram",
        "chickpea": "gram",
        "rapeseed mustard": "mustard",
        "musard": "mustard",
        "maharashtra": "maharashtra",
    }
    for old, new in replacements.items():
        if text == old:
            return new
    return text


def _safe_float(value: object) -> float | None:
    if pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clean_reference_table(df: pd.DataFrame, source_type: str) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(col).strip() for col in out.columns]
    if source_type == "exact":
        out["crop_normalized"] = out.get("crop", "").apply(_normalize_crop_name)
        out["index"] = out.get("index", "")
        out["crop_age"] = out.get("crop_age", "")
        out["age_unit"] = out.get("age_unit", "")
        out["value"] = out.get("index_value").apply(_safe_float)
        out["source_type"] = "exact"
    else:
        out["crop_normalized"] = out.get("crop", "").apply(_normalize_crop_name)
        out["index"] = out.get("index", "")
        out["crop_age"] = out.get("crop_age", "")
        out["age_unit"] = ""
        out["value"] = out.get("reported_value_or_range").apply(_safe_float)
        out["source_type"] = "range"

    out["evidence_type"] = out.get("evidence_type", out.get("value_type", ""))
    out["source_url"] = out.get("source_url", "")
    out["notes"] = out.get("notes", "")
    out["crop"] = out.get("crop", "")
    out["index"] = out["index"].fillna("")
    out["crop_age"] = out["crop_age"].fillna("")
    out["evidence_type"] = out["evidence_type"].fillna("")
    out["source_url"] = out["source_url"].fillna("")
    out["notes"] = out["notes"].fillna("")
    return out[["crop", "crop_normalized", "index", "crop_age", "age_unit", "value", "evidence_type", "source_url", "notes", "source_type"]]


def load_phenology_reference(base_dir: str | Path | None = None) -> pd.DataFrame:
    if base_dir is None:
        base_dir = Path(__file__).resolve().parents[2] / "data" / "raw" / "reference"

    frames = []
    for source_type, filename in REFERENCE_FILES.items():
        path = Path(base_dir) / filename
        if not path.exists():
            continue
        df = pd.read_csv(path)
        frames.append(_clean_reference_table(df, source_type))

    if not frames:
        raise FileNotFoundError("No phenology reference CSV files found under the expected raw/reference location.")

    result = pd.concat(frames, ignore_index=True)
    result = result[result["crop_normalized"].astype(str).str.len() > 0].copy()
    return result.reset_index(drop=True)


def summarize_crop_evidence(reference_df: pd.DataFrame | None = None) -> pd.DataFrame:
    if reference_df is None:
        reference_df = load_phenology_reference()

    summary = (
        reference_df.groupby("crop_normalized", as_index=False)
        .agg(
            evidence_count=("value", "count"),
            index_types=("index", lambda s: ", ".join(sorted(set(str(v) for v in s if str(v).strip())))),
            typical_ages=("crop_age", lambda s: ", ".join(sorted(set(str(v) for v in s if str(v).strip()))[:5])),
            avg_value=("value", "mean"),
            min_value=("value", "min"),
            max_value=("value", "max"),
        )
        .sort_values("evidence_count", ascending=False)
    )
    return summary


def _cluster_label_for_row(row: pd.Series) -> tuple[str, str, str]:
    top_crop = str(row.get("dominant_top_crop", "")).strip()
    kharif = float(row.get("avg_kharif_share", 0.0) or 0.0)
    rabi = float(row.get("avg_rabi_share", 0.0) or 0.0)
    other = float(row.get("avg_other_share", 0.0) or 0.0)
    zaid = float(row.get("avg_zaid_share", 0.0) or 0.0)

    top_crop_norm = _normalize_crop_name(top_crop)

    if "paddy" in top_crop_norm or "rice" in top_crop_norm:
        label = "paddy-dominant kharif regime"
        rationale = "High kharif dominance and paddy as the main crop indicate a monsoon rice belt; this matches the literature pattern of rice-dominant seasonal systems."
    elif "cotton" in top_crop_norm:
        label = "cotton-dominant kharif regime"
        rationale = "The cluster is strongly kharif-dominant with cotton as the main crop, consistent with a cotton-heavy rainfed regime."
    elif "sugarcane" in top_crop_norm:
        label = "sugarcane-heavy commercial belt"
        rationale = "Sugarcane dominance with elevated non-rainy acreage suggests a perennial/commercial agricultural regime, especially where sugarcane is the stand-out crop."
    elif "jowar" in top_crop_norm:
        label = "mixed cereal dryland regime"
        rationale = "Jowar dominance and relatively balanced seasonality suggest a dryland cereal belt with important kharif and rabi activity."
    elif "gram" in top_crop_norm or "mustard" in top_crop_norm:
        label = "rabi pulse-oilseed regime"
        rationale = "The cluster is tilted toward rabi production with pulse/oilseed dominance, consistent with winter-sown crop regimes."
    elif kharif > 0.7:
        label = "kharif-dominant seasonal regime"
        rationale = "A heavily kharif-weighted cluster indicates a rainfall-driven seasonal system without a single dominant crop label yet."
    elif rabi > 0.45:
        label = "rabi-dominant seasonal regime"
        rationale = "The cluster’s winter-season share is dominant, indicating a winter-sown regime with likely wheat, gram, or mustard emphasis."
    elif other > 0.2:
        label = "mixed commercial or fallow regime"
        rationale = "The cluster shows notable non-seasonal acreage and mixed patterns, which usually reflects a commercial or fallow-influenced agricultural regime."
    elif zaid > 0.1:
        label = "summer or mixed seasonal regime"
        rationale = "The cluster retains meaningful summer-season activity, suggesting an irrigated or summer crop contribution."
    else:
        label = "mixed seasonal regime"
        rationale = "This cluster shows a mixed seasonal pattern without a single crop category dominating the historical profile."

    evidence_crops = ", ".join(sorted({top_crop_norm, "paddy", "cotton", "sugarcane", "jowar", "gram"} - {""}))
    return label, rationale, evidence_crops


def build_cluster_label_map(summary_df: pd.DataFrame, reference_df: pd.DataFrame | None = None) -> pd.DataFrame:
    if reference_df is None:
        reference_df = load_phenology_reference()

    cluster_df = summary_df.copy()
    rows = []
    for _, row in cluster_df.iterrows():
        label, rationale, evidence_crops = _cluster_label_for_row(row)
        phenology_match = reference_df[reference_df["crop_normalized"].isin([_normalize_crop_name(row.get("dominant_top_crop", ""))])] if str(row.get("dominant_top_crop", "")).strip() else pd.DataFrame()
        rows.append(
            {
                "cluster": int(row["cluster"]),
                "dominant_top_crop": row.get("dominant_top_crop", ""),
                "avg_kharif_share": row.get("avg_kharif_share", 0.0),
                "avg_rabi_share": row.get("avg_rabi_share", 0.0),
                "avg_zaid_share": row.get("avg_zaid_share", 0.0),
                "avg_other_share": row.get("avg_other_share", 0.0),
                "label": label,
                "rationale": rationale,
                "evidence_crops": evidence_crops,
                "phenology_evidence_count": int(len(phenology_match)) if isinstance(phenology_match, pd.DataFrame) else 0,
            }
        )

    result = pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)
    return result


def save_cluster_label_artifacts(summary_path: str | Path | None = None, output_dir: str | Path | None = None) -> pd.DataFrame:
    if summary_path is None:
        summary_path = PROCESSED_DIR / "clusters" / "cluster_summary.csv"
    if output_dir is None:
        output_dir = LABELS_DIR

    summary_path = Path(summary_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.read_csv(summary_path)
    labels_df = build_cluster_label_map(summary_df)

    labels_path = output_dir / "cluster_labels.csv"
    metadata_path = output_dir / "cluster_labels_metadata.json"

    labels_df.to_csv(labels_path, index=False)
    metadata = {
        "source_summary": str(summary_path),
        "generated_rows": int(len(labels_df)),
        "labels": labels_df[["cluster", "label"]].to_dict(orient="records"),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2))
    return labels_df


if __name__ == "__main__":
    labels_df = save_cluster_label_artifacts()
    print(labels_df.to_string(index=False))
