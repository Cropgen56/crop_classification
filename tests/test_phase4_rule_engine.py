import pandas as pd

from src.rules.rule_engine import classify_cluster_profile, load_crop_rules, score_cluster_profile


def test_load_crop_rules_includes_priority_maharashtra_crops():
    rules = load_crop_rules()
    for crop in ["paddy", "cotton", "jowar", "sugarcane", "gram"]:
        assert crop in rules
    assert rules["paddy"]["season"] == "kharif"


def test_score_cluster_profile_favors_paddy_for_kharif_rice_cluster():
    profile = {
        "avg_kharif_share": 0.88,
        "avg_rabi_share": 0.10,
        "avg_zaid_share": 0.02,
        "avg_other_share": 0.00,
        "dominant_top_crop": "Paddy (Rice)",
    }

    score = score_cluster_profile(profile, "paddy")
    assert score > 0.7


def test_classify_cluster_profile_returns_best_candidate():
    row = {
        "cluster": 0,
        "dominant_top_crop": "Paddy (Rice)",
        "avg_kharif_share": 0.88,
        "avg_rabi_share": 0.10,
        "avg_zaid_share": 0.02,
        "avg_other_share": 0.00,
    }

    result = classify_cluster_profile(row)
    assert result["best_crop"] == "paddy"
    assert result["best_score"] >= 0.7
    assert len(result["ranked_candidates"]) >= 3
