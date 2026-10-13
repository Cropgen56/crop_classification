from fastapi.testclient import TestClient

from src.api.app import app


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_classify_village_endpoint():
    payload = {
        "village_id": "v001",
        "fields": {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "id": "f1",
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
                    "id": "f2",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[73.2, 19.1], [73.3, 19.1], [73.3, 19.2], [73.2, 19.2], [73.2, 19.1]]],
                    },
                    "properties": {
                        "dominant_top_crop": "Paddy (Rice)",
                        "avg_kharif_share": 0.83,
                        "avg_rabi_share": 0.11,
                        "avg_zaid_share": 0.02,
                        "avg_other_share": 0.04,
                        "top_crop_share": 0.70,
                    },
                },
            ],
        },
    }

    response = client.post("/classify-village", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["village_id"] == "v001"
    assert body["dominant_crop"] == "paddy"
    assert "top_candidates" in body
