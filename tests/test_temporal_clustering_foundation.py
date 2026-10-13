import pandas as pd

from src.labeling.temporal_clustering import (
    build_temporal_signature_features,
    label_temporal_cluster_profile,
    match_field_to_crop_reference,
    rank_crops_for_temporal_series,
    summarize_temporal_cluster_profiles,
)


def test_build_temporal_signature_features_adds_reference_metrics():
    df = pd.DataFrame(
        [
            {
                "field_id": "F1",
                "crop_label": "rice",
                "sowing_date": "2024-06-15",
                "area_hectares": 1.2,
            },
            {
                "field_id": "F2",
                "crop_label": "soybean",
                "sowing_date": "2024-06-25",
                "area_hectares": 1.5,
            },
        ]
    )

    out = build_temporal_signature_features(df)

    assert {"field_id", "crop_label", "signature_peak_ndvi", "signature_peak_timing", "similarity_profile"}.issubset(out.columns)
    assert len(out) == 2
    assert out["signature_peak_ndvi"].notna().all()


def test_match_field_to_crop_reference_prefers_known_crop_match():
    field = {
        "field_id": "F1",
        "crop_label": "cotton",
        "sowing_date": "2024-06-20",
        "area_hectares": 2.0,
    }

    ranked = match_field_to_crop_reference(field)
    assert ranked
    assert ranked[0]["crop"] in {"cotton", "soybean", "paddy", "sugarcane", "wheat"}
    assert ranked[0]["score"] >= 0.0


def test_summarize_temporal_cluster_profiles_and_label_them():
    fields = pd.DataFrame(
        [
            {"field_id": "F1", "crop_label": "rice", "cluster": 0, "signature_peak_ndvi": 0.82, "signature_peak_ndre": 0.35, "signature_peak_gndvi": 0.5, "signature_season": "kharif"},
            {"field_id": "F2", "crop_label": "rice", "cluster": 0, "signature_peak_ndvi": 0.78, "signature_peak_ndre": 0.33, "signature_peak_gndvi": 0.46, "signature_season": "kharif"},
            {"field_id": "F3", "crop_label": "soybean", "cluster": 1, "signature_peak_ndvi": 0.76, "signature_peak_ndre": 0.32, "signature_peak_gndvi": 0.44, "signature_season": "kharif"},
        ]
    )

    cluster_summary = summarize_temporal_cluster_profiles(fields)
    assert {"cluster", "mean_peak_ndvi", "mean_peak_ndre", "dominant_signature_crop", "candidate_label"}.issubset(cluster_summary.columns)

    label = label_temporal_cluster_profile(cluster_summary.iloc[0].to_dict())
    assert label["label"]
    assert label["crop_candidates"]


def test_rank_crops_for_temporal_series_uses_crop_age_evidence():
    reference = []
    curves = {
        "soybean": {"ndvi": [0.2, 0.6], "ndre": [0.1, 0.3], "gndvi": [0.25, 0.55]},
        "cotton": {"ndvi": [0.8, 0.3], "ndre": [0.4, 0.2], "gndvi": [0.6, 0.25]},
    }
    for crop, indices in curves.items():
        for index, values in indices.items():
            for age, value in zip((30, 60), values):
                reference.append({
                    "crop": crop,
                    "index": index,
                    "crop_age": f"{age} days after emergence",
                    "age_unit": "days",
                    "index_value": value,
                })

    timeseries = [
        {"date": "2024-01", "ndvi": 0.3, "ndre": 0.2, "gndvi": 0.4},
        {"date": "2024-02", "ndvi": 0.2, "ndre": 0.1, "gndvi": 0.25},
        {"date": "2024-03", "ndvi": 0.6, "ndre": 0.3, "gndvi": 0.55},
    ]

    ranked = rank_crops_for_temporal_series(timeseries, reference)

    assert ranked[0]["crop"] == "soybean"
    assert ranked[0]["evidence"]["matched_points"] >= 2
    assert ranked[0]["evidence"]["reference"] == "Crop_Index_Exact_Temporal_Points.csv"
