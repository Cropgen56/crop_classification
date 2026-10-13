from src.rules.village_classifier import classify_village, summarize_village_predictions


def test_summarize_village_predictions_finds_paddy_dominant_mix():
    field_profiles = [
        {
            "field_id": "f1",
            "dominant_top_crop": "Paddy (Rice)",
            "avg_kharif_share": 0.88,
            "avg_rabi_share": 0.10,
            "avg_zaid_share": 0.02,
            "avg_other_share": 0.00,
            "top_crop_share": 0.72,
        },
        {
            "field_id": "f2",
            "dominant_top_crop": "Paddy (Rice)",
            "avg_kharif_share": 0.82,
            "avg_rabi_share": 0.12,
            "avg_zaid_share": 0.03,
            "avg_other_share": 0.03,
            "top_crop_share": 0.70,
        },
        {
            "field_id": "f3",
            "dominant_top_crop": "Cotton",
            "avg_kharif_share": 0.74,
            "avg_rabi_share": 0.18,
            "avg_zaid_share": 0.04,
            "avg_other_share": 0.04,
            "top_crop_share": 0.58,
        },
    ]

    summary = summarize_village_predictions(field_profiles)
    assert summary["dominant_crop"] == "paddy"
    assert summary["dominant_share"] >= 0.55
    assert len(summary["crop_mix"]) >= 2


def test_classify_village_returns_village_summary_and_top_candidates():
    village = {
        "village_id": "v001",
        "fields": [
            {
                "field_id": "f1",
                "dominant_top_crop": "Paddy (Rice)",
                "avg_kharif_share": 0.88,
                "avg_rabi_share": 0.10,
                "avg_zaid_share": 0.02,
                "avg_other_share": 0.00,
                "top_crop_share": 0.72,
            },
            {
                "field_id": "f2",
                "dominant_top_crop": "Paddy (Rice)",
                "avg_kharif_share": 0.83,
                "avg_rabi_share": 0.11,
                "avg_zaid_share": 0.02,
                "avg_other_share": 0.04,
                "top_crop_share": 0.70,
            },
        ],
    }

    result = classify_village(village)
    assert result["dominant_crop"] == "paddy"
    assert result["field_count"] == 2
    assert len(result["top_candidates"]) >= 2
    assert "reason" in result["summary"]
