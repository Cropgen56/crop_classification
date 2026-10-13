import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.service.classification_service import classify_village_request


if __name__ == "__main__":
    sample = {
        "village_id": "example_village_01",
        "fields": [
            {
                "field_id": "field_1",
                "dominant_top_crop": "Paddy (Rice)",
                "avg_kharif_share": 0.88,
                "avg_rabi_share": 0.10,
                "avg_zaid_share": 0.02,
                "avg_other_share": 0.00,
                "top_crop_share": 0.72,
            },
            {
                "field_id": "field_2",
                "dominant_top_crop": "Cotton",
                "avg_kharif_share": 0.77,
                "avg_rabi_share": 0.18,
                "avg_zaid_share": 0.02,
                "avg_other_share": 0.03,
                "top_crop_share": 0.60,
            },
        ],
    }

    response = classify_village_request(sample)
    print(json.dumps(response, indent=2))
