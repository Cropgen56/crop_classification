import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.feature_engineering.sentinel_field_extractor import build_village_field_profiles


if __name__ == "__main__":
    sample_payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "id": "field_1",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [73.1, 19.1],
                            [73.2, 19.1],
                            [73.2, 19.2],
                            [73.1, 19.2],
                            [73.1, 19.1],
                        ]
                    ],
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
                    "coordinates": [
                        [
                            [73.2, 19.1],
                            [73.3, 19.1],
                            [73.3, 19.2],
                            [73.2, 19.2],
                            [73.2, 19.1],
                        ]
                    ],
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

    profiles = build_village_field_profiles(sample_payload)
    print(json.dumps(profiles, indent=2))
