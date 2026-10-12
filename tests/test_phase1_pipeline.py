from pathlib import Path

from src.feature_engineering.build_region_features import build_region_feature_table


def test_region_feature_table_has_expected_columns():
    table = build_region_feature_table()
    expected = {
        "District",
        "total_area_ha",
        "kharif_share",
        "rabi_share",
        "zaid_share",
        "crop_count",
        "season_entropy",
        "top_crop",
        "village_count",
    }
    assert expected.issubset(set(table.columns))
    assert not table.empty


def test_region_feature_table_has_reasonable_scores():
    table = build_region_feature_table()
    assert (table["kharif_share"].fillna(0) >= 0).all()
    assert (table["rabi_share"].fillna(0) >= 0).all()
    assert (table["zaid_share"].fillna(0) >= 0).all()
    assert (table["season_entropy"].fillna(0) >= 0).all()
