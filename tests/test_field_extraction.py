from src.feature_engineering import sentinel_field_extractor
from src.feature_engineering.sentinel_field_extractor import _monthly_windows, build_village_field_profiles


FEATURE_COLLECTION = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "id": "field_1",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[73.1, 19.1], [73.2, 19.1], [73.2, 19.2], [73.1, 19.2], [73.1, 19.1]]],
            },
            "properties": {
                "dominant_top_crop": "Paddy (Rice)",
                "avg_kharif_share": 0.88,
                "avg_rabi_share": 0.10,
                "avg_zaid_share": 0.02,
                "avg_other_share": 0.00,
                "top_crop_share": 0.72,
            },
        },
        {
            "type": "Feature",
            "id": "field_2",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[[73.2, 19.1], [73.3, 19.1], [73.3, 19.2], [73.2, 19.2], [73.2, 19.1]]],
            },
            "properties": {
                "dominant_top_crop": "Cotton",
                "avg_kharif_share": 0.70,
                "avg_rabi_share": 0.18,
                "avg_zaid_share": 0.02,
                "avg_other_share": 0.10,
                "top_crop_share": 0.61,
            },
        },
    ],
}


def test_build_village_field_profiles(monkeypatch):
    monkeypatch.setattr(sentinel_field_extractor, "_initialize_earth_engine", lambda _path: False)
    profiles = build_village_field_profiles(FEATURE_COLLECTION)
    assert len(profiles) == 2
    first = profiles[0]
    assert first["field_id"] == "field_1"
    assert first["dominant_top_crop"] == "Paddy (Rice)"
    assert 0.0 <= first["avg_kharif_share"] <= 1.0
    assert "satellite" in first
    assert "ndvi_mean" in first["satellite"]
    assert first["satellite"]["source"] == "unavailable"
    assert first["satellite"]["timeseries"] == []


def test_monthly_windows_cover_partial_start_and_end_months():
    assert _monthly_windows("2024-06-15", "2024-08-02") == [
        ("2024-06-15", "2024-07-01", "2024-06"),
        ("2024-07-01", "2024-08-01", "2024-07"),
        ("2024-08-01", "2024-08-03", "2024-08"),
    ]


def test_build_village_field_profiles_uses_one_village_batch(monkeypatch):
    calls = []

    def fake_batch(features, start_date, end_date, progress_callback=None):
        calls.append((len(features), start_date, end_date))
        return {}

    monkeypatch.setattr(sentinel_field_extractor, "_try_gee_village_profiles", fake_batch)

    profiles = build_village_field_profiles(FEATURE_COLLECTION)

    assert len(profiles) == len(FEATURE_COLLECTION["features"])
    assert calls == [(2, "2024-01-01", "2024-12-31")]


def test_build_village_field_profiles_uses_crop_evidence_when_gee_is_missing(monkeypatch):
    monkeypatch.setattr(sentinel_field_extractor, "_initialize_earth_engine", lambda _path: False)
    payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": "field_3",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[73.4, 19.1], [73.5, 19.1], [73.5, 19.2], [73.4, 19.2], [73.4, 19.1]]],
                },
                "properties": {
                    "dominant_top_crop": "Soybean",
                    "avg_kharif_share": 0.76,
                    "avg_rabi_share": 0.10,
                    "avg_zaid_share": 0.05,
                    "avg_other_share": 0.09,
                    "top_crop_share": 0.63,
                },
            }
        ],
    }

    profiles = build_village_field_profiles(payload)
    assert len(profiles) == 1
    satellite = profiles[0]["satellite"]
    assert satellite["source"] == "unavailable"
    assert satellite["timeseries"] == []
    assert "substituted" in satellite["reason"]
