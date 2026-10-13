"""
district_priors.py
~~~~~~~~~~~~~~~~~~
Loads the District_Crop_Priors sheet from MH_Crop_Master_2015_2026.xlsx
and exposes district-level Bayesian prior weights per crop.

These priors down-weight satellite scores for implausible crops (e.g. paddy
in Amravati, < 1 % of kharif area) and up-weight dominant crops (e.g. soybean
at 44 %).
"""
from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]
MASTER_FILE = ROOT_DIR / "MH_Crop_Master_2015_2026.xlsx"

# ---------------------------------------------------------------------------
# master-file class name → canonical crop name
# ---------------------------------------------------------------------------
_MASTER_TO_CANONICAL: dict[str, str] = {
    "soybean": "soybean",
    "cotton": "cotton",
    "tur (pigeonpea)": "tur",
    "paddy (kharif)": "paddy",
    "paddy (summer)": "paddy",
    "maize (kharif)": "maize",
    "maize (rabi)": "maize",
    "jowar (kharif)": "jowar",
    "jowar (rabi)": "jowar",
    "bajra": "bajra",
    "gram (chickpea)": "gram",
    "wheat": "wheat",
    "moong": "mung",
    "urad": "urad",
    "groundnut (kharif)": "groundnut",
    "groundnut (summer)": "groundnut",
    "sugarcane": "sugarcane",
    "turmeric": "turmeric",
    "banana": "banana",
    "orange / mosambi": "orange",
    "mango / cashew": "mango",
    "onion (rabi)": "onion",
    "chilli": "chilli",
    "safflower": "other_crop",
    "moong / maize (summer)": "maize",
    "nagli / ragi and small millets": "other_crop",
    "nagli / ragi & small millets": "other_crop",
    "rabi pulses (other)": "gram",
    "other kharif pulses": "other_crop",
    "other cereals": "other_crop",
    "other summer pulses": "other_crop",
    "pomegranate": "other_crop",
    "grapes": "other_crop",
    "rapeseed and mustard": "mustard",
    "rapeseed & mustard": "mustard",
    "fallow / bare": "fallow",
    "grass / scrub": "fallow",
    "water": "fallow",
    "built-up": "fallow",
    "tree / forest": "fallow",
    # Intercrop classes
    "soybean + tur": "soybean_tur",
    "cotton + tur": "cotton_tur",
    "cotton + soybean": "cotton_soybean",
    "cotton + moong/urad": "cotton",
    "bajra + tur": "tur",
    "jowar + tur": "jowar",
    "turmeric + soybean": "turmeric",
}

# Intercrop keys -> component single crops they also boost (via max prior)
_INTERCROP_COMPONENTS: dict[str, list[str]] = {
    "soybean_tur": ["soybean", "tur"],
    "cotton_tur": ["cotton", "tur"],
    "cotton_soybean": ["cotton", "soybean"],
}


def _normalise_class_name(name: str) -> str:
    text = str(name).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


@lru_cache(maxsize=1)
def _load_prior_table() -> pd.DataFrame:
    if not MASTER_FILE.exists():
        logger.warning("Master prior file not found: %s", MASTER_FILE)
        return pd.DataFrame()
    try:
        df = pd.read_excel(MASTER_FILE, sheet_name="District_Crop_Priors", header=3)
        df.columns = [str(c).strip() for c in df.columns]
        df["District"] = df["District"].fillna("").astype(str).str.strip()
        df["Class"] = df["Class"].fillna("").astype(str).str.strip()
        df["Plausibility"] = df["Plausibility"].fillna("").astype(str).str.strip()
        df["Prior_Weight"] = pd.to_numeric(df["Prior_Weight"], errors="coerce").fillna(0.002)
        df["Season"] = df["Season"].fillna("").astype(str).str.strip()
        df["_canonical"] = df["Class"].apply(
            lambda x: _MASTER_TO_CANONICAL.get(_normalise_class_name(x), "unknown")
        )
        logger.info("Loaded %d district crop prior rows from %s.", len(df), MASTER_FILE.name)
        return df
    except Exception as exc:
        logger.exception("Failed to load district crop priors: %s", exc)
        return pd.DataFrame()


def _match_district(df: pd.DataFrame, district_name: str) -> pd.DataFrame:
    query = district_name.strip().lower()
    mask = df["District"].str.lower().str.strip() == query
    subset = df[mask]
    if not subset.empty:
        return subset
    mask2 = df["District"].str.lower().str.contains(re.escape(query), na=False)
    return df[mask2]


def get_district_priors(district_name: str) -> dict[str, float]:
    """
    Return {canonical_crop: prior_weight} for the given district.

    Intercrop entries boost both their component crops.
    Returns {} if district is unknown (= no prior filtering).
    """
    if not district_name:
        return {}
    df = _load_prior_table()
    if df.empty:
        return {}
    district_df = _match_district(df, district_name)
    if district_df.empty:
        logger.warning("District '%s' not found in prior table.", district_name)
        return {}

    priors: dict[str, float] = {}
    for _, row in district_df.iterrows():
        canonical = str(row["_canonical"]).strip()
        weight = float(row["Prior_Weight"])
        if not canonical or canonical == "unknown":
            continue
        priors[canonical] = max(priors.get(canonical, 0.0), weight)
        for component in _INTERCROP_COMPONENTS.get(canonical, []):
            priors[component] = max(priors.get(component, 0.0), weight)
    return priors


def get_plausible_crops(district_name: str) -> set[str]:
    """
    Return canonical crop names that are NOT 'Implausible' or 'Not expected'.
    Returns empty set when district unknown (= no hard filtering applied).
    """
    if not district_name:
        return set()
    df = _load_prior_table()
    if df.empty:
        return set()
    district_df = _match_district(df, district_name)
    if district_df.empty:
        return set()
    plausible = district_df[
        ~district_df["Plausibility"].str.lower().str.contains(
            "implausible|not expected", na=False
        )
    ]
    return set(plausible["_canonical"].dropna().unique()) - {"unknown"}


__all__ = ["get_district_priors", "get_plausible_crops"]
