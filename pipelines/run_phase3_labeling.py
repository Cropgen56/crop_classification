import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.labeling.cluster_labeling import save_cluster_label_artifacts


if __name__ == "__main__":
    labels_df = save_cluster_label_artifacts(ROOT / "data/processed/clusters/cluster_summary.csv", ROOT / "data/labels")
    print(f"Saved cluster labels for {len(labels_df)} clusters.")
    print(labels_df.to_string(index=False))
