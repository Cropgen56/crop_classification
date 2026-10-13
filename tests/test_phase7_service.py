from src.service.classification_service import classify_village_request


def test_classify_village_request_formats_structured_response():
    payload = {
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

    result = classify_village_request(payload)
    assert result["village_id"] == "v001"
    assert result["dominant_crop"] == "paddy"
    assert result["request_version"] == "0.7.0"
    assert "top_candidates" in result
    assert len(result["top_candidates"]) >= 2


def test_classify_village_request_returns_per_field_top3_predictions():
    payload = {
        "village_id": "v002",
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
                "dominant_top_crop": "Cotton",
                "avg_kharif_share": 0.61,
                "avg_rabi_share": 0.18,
                "avg_zaid_share": 0.06,
                "avg_other_share": 0.15,
                "top_crop_share": 0.58,
            },
        ],
    }

    result = classify_village_request(payload)
    assert "field_predictions" in result
    assert len(result["field_predictions"]) == 2
    assert len(result["field_predictions"][0]["top_3_candidates"]) == 3
