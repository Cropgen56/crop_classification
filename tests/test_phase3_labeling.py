import pandas as pd

from src.labeling.cluster_labeling import build_cluster_label_map, load_phenology_reference


def test_load_phenology_reference_has_expected_rows():
    df = load_phenology_reference()
    assert not df.empty
    assert {"crop", "index", "crop_age", "evidence_type"}.issubset(set(df.columns))
    assert "cotton" in set(df["crop_normalized"].str.lower())


def test_build_cluster_label_map_generates_labels_for_each_cluster():
    summary = pd.read_csv("data/processed/clusters/cluster_summary.csv")
    result = build_cluster_label_map(summary)
    assert {"cluster", "label", "rationale", "evidence_crops"}.issubset(set(result.columns))
    assert len(result) == summary["cluster"].nunique()
    assert result["label"].notna().all()
