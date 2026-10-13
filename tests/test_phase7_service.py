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
