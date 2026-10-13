from src.feature_engineering.sentinel_field_extractor import build_village_field_profiles


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


def test_build_village_field_profiles():
    profiles = build_village_field_profiles(FEATURE_COLLECTION)
    assert len(profiles) == 2
    first = profiles[0]
    assert first["field_id"] == "field_1"
    assert first["dominant_top_crop"] == "Paddy (Rice)"
    assert 0.0 <= first["avg_kharif_share"] <= 1.0
    assert "satellite" in first
    assert "ndvi_mean" in first["satellite"]
