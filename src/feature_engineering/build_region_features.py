from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config.settings import RAW_HISTORICAL_FILE, RAW_TRAINING_FILE, PROCESSED_DIR


PERENNIAL_CROPS = {
    "orange",
    "sapota",
    "orchard",
    "plantation",
    "banana",
}


def _normalize_crop_name(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().lower()
    return text.replace("-", " ")


def _season_key(value: object) -> str:
    if pd.isna(value):
        return "unknown"
    season = str(value).strip().lower()
    if "kharif" in season:
        return "kharif"
    if "rabi" in season:
        return "rabi"
    if "zaid" in season or "summer" in season:
        return "zaid"
    return "other"


def _safe_float(value: object) -> float:
    if pd.isna(value):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _compute_entropy(series: pd.Series) -> float:
    values = series[series > 0]
    if values.empty:
        return 0.0
    p = values / values.sum()
    return float(-(p * np.log(p)).sum())


def load_district_history(path: str | Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="History_1997_2019", header=3)
    expected_columns = [
        "Crop_Year",
        "Year_Start",
        "District",
        "Source_District",
        "Crop",
        "Season",
        "Crop_Season",
        "Area_ha",
        "Production",
        "Production_Unit",
        "Yield",
        "Source_Crop",
    ]
    available = [col for col in expected_columns if col in df.columns]
    if not available:
        raise ValueError(f"No expected historical crop columns found in {path}")
    df = df.loc[:, available].copy()
    df["District"] = df["District"].fillna("").astype(str).str.strip()
    df["Crop"] = df["Crop"].fillna("").astype(str).str.strip()
    df["Season"] = df["Season"].fillna("").astype(str).str.strip()
    df["Area_ha"] = df["Area_ha"].apply(_safe_float)
    df["Crop_Year"] = df["Crop_Year"].fillna("").astype(str).str.strip()
    df["season_key"] = df["Season"].apply(_season_key)
    return df


def load_village_data(path: str | Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="Villages")
    df = df.copy()
    df["District"] = df["District"].fillna("").astype(str).str.strip()
    df["Area_ha"] = df["Area_ha"].apply(_safe_float)
    return df


def load_district_summary(path: str | Path) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name="District_Summary")
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    for col in ["District", "Talukas", "Villages", "Geographic_Area_ha", "Avg_Village_Area_ha"]:
        if col in df.columns:
            df[col] = df[col].apply(_safe_float if col != "District" else lambda x: str(x).strip())
    return df


def aggregate_recent_crop_profile(history_df: pd.DataFrame) -> pd.DataFrame:
    recent_years = [
        year for year in sorted(history_df["Crop_Year"].dropna().unique(), key=str)
        if "2017" in str(year) or "2018" in str(year) or "2019" in str(year)
    ]
    recent_df = history_df[history_df["Crop_Year"].isin(recent_years)].copy()

    district_crop = (
        recent_df.groupby(["District", "Crop"], as_index=False)["Area_ha"].sum()
        .rename(columns={"Area_ha": "crop_area_ha"})
    )

    district_season = (
        recent_df.groupby(["District", "season_key"], as_index=False)["Area_ha"].sum()
        .rename(columns={"Area_ha": "season_area_ha"})
    )

    district_total = (
        recent_df.groupby("District", as_index=False)["Area_ha"].sum()
        .rename(columns={"Area_ha": "total_area_ha"})
    )

    crop_share = district_crop.merge(district_total, on="District", how="left")
    crop_share["crop_share"] = crop_share["crop_area_ha"] / crop_share["total_area_ha"].replace(0, np.nan)

    top_crop = (
        crop_share.sort_values(["District", "crop_area_ha"], ascending=[True, False])
        .drop_duplicates("District")
        .rename(columns={"Crop": "top_crop", "crop_area_ha": "top_crop_area_ha", "crop_share": "top_crop_share"})
        [["District", "top_crop", "top_crop_area_ha", "top_crop_share"]]
    )

    profile = district_total.merge(
        district_season.pivot(index="District", columns="season_key", values="season_area_ha").fillna(0),
        on="District",
        how="left",
    )
    profile["kharif_share"] = profile.get("kharif", 0.0) / profile["total_area_ha"].replace(0, np.nan)
    profile["rabi_share"] = profile.get("rabi", 0.0) / profile["total_area_ha"].replace(0, np.nan)
    profile["zaid_share"] = profile.get("zaid", 0.0) / profile["total_area_ha"].replace(0, np.nan)
    profile["other_share"] = profile.get("other", 0.0) / profile["total_area_ha"].replace(0, np.nan)

    crop_group = recent_df.copy()
    crop_group["crop_norm"] = crop_group["Crop"].apply(_normalize_crop_name)
    crop_group["is_perennial"] = crop_group["crop_norm"].isin(PERENNIAL_CROPS)
    perennial_area = (
        crop_group.groupby("District")["Area_ha"].apply(lambda s: s[crop_group.loc[s.index, "is_perennial"]].sum())
        if False else None
    )

    perennial_by_district = (
        crop_group[crop_group["is_perennial"]]
        .groupby("District", as_index=False)["Area_ha"].sum()
        .rename(columns={"Area_ha": "perennial_area_ha"})
    )
    profile = profile.merge(perennial_by_district, on="District", how="left")
    profile["perennial_share"] = profile["perennial_area_ha"].fillna(0) / profile["total_area_ha"].replace(0, np.nan)

    crop_count = (
        recent_df.groupby("District", as_index=False)["Crop"].nunique()
        .rename(columns={"Crop": "crop_count"})
    )
    profile = profile.merge(crop_count, on="District", how="left")

    season_entropy = (
        recent_df.groupby(["District", "season_key"], as_index=False)["Area_ha"].sum()
        .rename(columns={"Area_ha": "season_area_ha"})
        .groupby("District")["season_area_ha"]
        .apply(_compute_entropy)
        .rename("season_entropy")
        .reset_index()
    )
    profile = profile.merge(season_entropy, on="District", how="left")

    top_crop = top_crop.merge(profile[["District", "total_area_ha"]], on="District", how="left")
    profile = profile.merge(top_crop[["District", "top_crop", "top_crop_area_ha", "top_crop_share"]], on="District", how="left")

    return profile.fillna(0)


def build_region_feature_table() -> pd.DataFrame:
    history_df = load_district_history(RAW_HISTORICAL_FILE)
    villages_df = load_village_data(RAW_TRAINING_FILE)
    district_summary = load_district_summary(RAW_TRAINING_FILE)

    district_profile = aggregate_recent_crop_profile(history_df)

    village_by_district = (
        villages_df.groupby("District", as_index=False)["Area_ha"].sum()
        .rename(columns={"Area_ha": "village_area_ha_total"})
    )
    village_counts = (
        villages_df.groupby("District", as_index=False)["Village_UID"].count()
        .rename(columns={"Village_UID": "village_count"})
    )

    feature_table = district_profile.merge(village_by_district, on="District", how="left")
    feature_table = feature_table.merge(village_counts, on="District", how="left")
    feature_table = feature_table.merge(district_summary[["District", "Talukas", "Villages", "Geographic_Area_ha", "Avg_Village_Area_ha"]], on="District", how="left")

    feature_table["area_gap_ha"] = feature_table["Geographic_Area_ha"].fillna(0) - feature_table["village_area_ha_total"].fillna(0)
    feature_table["avg_village_area_ha_actual"] = feature_table["village_area_ha_total"].fillna(0) / feature_table["village_count"].replace(0, np.nan)
    feature_table["seasonal_balance"] = (
        feature_table.get("kharif_share", 0.0).fillna(0) - feature_table.get("rabi_share", 0.0).fillna(0)
    ).abs()
    feature_table["dominant_season"] = feature_table[["kharif_share", "rabi_share", "zaid_share"]].idxmax(axis=1).str.replace("_share", "")

    return feature_table.sort_values("District").reset_index(drop=True)


def save_region_feature_table(out_path: str | Path) -> pd.DataFrame:
    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table = build_region_feature_table()
    table.to_csv(output_path, index=False)
    return table


if __name__ == "__main__":
    output_file = PROCESSED_DIR / "region_feature_matrix.csv"
    result = save_region_feature_table(output_file)
    print(f"Saved {len(result)} district-level feature rows to {output_file}")
    print(result.head(5).to_string(index=False))
