from src.rules.field_candidate_generator import classify_field_profile, generate_crop_candidates


def test_generate_crop_candidates_ranks_paddy_high_for_rice_profile():
    profile = {
        "dominant_top_crop": "Paddy (Rice)",
        "avg_kharif_share": 0.88,
        "avg_rabi_share": 0.10,
        "avg_zaid_share": 0.02,
        "avg_other_share": 0.00,
        "top_crop_share": 0.72,
    }

    candidates = generate_crop_candidates(profile)
    assert candidates[0]["crop"] == "paddy"
    assert candidates[0]["score"] >= 0.8
    assert len(candidates) >= 3


def test_classify_field_profile_returns_best_candidate_and_reasons():
    profile = {
        "field_id": "field_101",
        "dominant_top_crop": "Cotton",
        "avg_kharif_share": 0.77,
        "avg_rabi_share": 0.21,
        "avg_zaid_share": 0.01,
        "avg_other_share": 0.01,
        "top_crop_share": 0.59,
    }

    result = classify_field_profile(profile)
    assert result["best_crop"] == "cotton"
    assert result["best_score"] >= 0.7
    assert "reason" in result["evidence_summary"]
    assert len(result["ranked_candidates"]) >= 3
