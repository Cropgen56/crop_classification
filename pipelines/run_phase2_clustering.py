import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from src.clustering.cluster_regions import save_cluster_artifacts


if __name__ == "__main__":
    input_path = ROOT / "data/processed/region_feature_matrix.csv"
    output_dir = ROOT / "data/processed/clusters"

    df = pd.read_csv(input_path)
    assignments, summary, metadata = save_cluster_artifacts(df, output_dir)

    print(f"Selected cluster count: {metadata['selected_k']}")
    print("Assignments:")
    print(assignments.head().to_string(index=False))
    print("\nSummary:")
    print(summary.to_string(index=False))
