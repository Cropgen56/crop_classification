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
    assert result["dominant_crop"] == "satellite data unavailable"
    assert result["request_version"] == "0.7.0"
    assert "top_candidates" in result
    assert result["top_candidates"] == []


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
    assert result["field_predictions"][0]["best_crop"] == "unknown"
    assert result["field_predictions"][0]["top_3_candidates"] == []


def test_classify_village_request_includes_temporal_cluster_evidence():
    payload = {
        "village_id": "v003",
        "fields": [
            {
                "field_id": "f1",
                "dominant_top_crop": "Soybean",
                "avg_kharif_share": 0.75,
                "avg_rabi_share": 0.12,
                "avg_zaid_share": 0.05,
                "avg_other_share": 0.08,
                "top_crop_share": 0.63,
            }
        ],
    }

    result = classify_village_request(payload)
    pred = result["field_predictions"][0]
    assert "temporal_signature" in pred
    assert "temporal_cluster_label" in pred
    assert pred["temporal_cluster_label"] == "satellite data unavailable"


def test_classify_village_request_includes_cluster_summary():
    payload = {
        "village_id": "v004",
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
    assert "cluster_summary" in result
    assert isinstance(result["cluster_summary"], list)
    assert result["satellite_status"] == "unavailable"
    assert result["dominant_crop"] == "satellite data unavailable"
    assert result["cluster_summary"] == []


def test_temporal_clusters_use_satellite_series_not_crop_metadata(monkeypatch):
    from src.service import classification_service

    def make_profile(field_id, values):
        return {
            "field_id": field_id,
            "dominant_top_crop": "Soybean",
            "avg_kharif_share": 0.7,
            "avg_rabi_share": 0.2,
            "avg_zaid_share": 0.05,
            "avg_other_share": 0.05,
            "satellite": {
                "source": "gee_sentinel_monthly",
                "timeseries": [
                    {"date": f"2024-{month:02d}", "ndvi": value, "ndre": value - 0.1, "gndvi": value - 0.05, "evi": value, "ndmi": value - 0.2, "vv": -12 + value, "vh": -19 + value}
                    for month, value in enumerate(values, start=6)
                ],
            },
        }

    profiles = [make_profile("field_a", [0.2, 0.3, 0.4]), make_profile("field_b", [0.7, 0.8, 0.9])]
    monkeypatch.setattr(classification_service, "build_village_field_profiles", lambda *_args, **_kwargs: profiles)

    result = classification_service.classify_village_request({"village_id": "test", "fields": []})

    assert result["satellite_status"] == "complete"
    assert result["satellite_field_count"] == 2
    assert len(result["cluster_summary"]) == 2
    assert result["field_predictions"][0]["cluster_id"] != result["field_predictions"][1]["cluster_id"]
