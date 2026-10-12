import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.feature_engineering.build_region_features import save_region_feature_table
from src.config.settings import PROCESSED_DIR


if __name__ == "__main__":
    output_path = PROCESSED_DIR / "region_feature_matrix.csv"
    table = save_region_feature_table(output_path)
    print(f"Feature matrix created with {len(table)} rows.")
    print(table.head(3).to_string(index=False))
