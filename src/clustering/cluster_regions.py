from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "kharif_share",
    "rabi_share",
    "zaid_share",
    "other_share",
    "perennial_share",
    "crop_count",
    "season_entropy",
    "seasonal_balance",
    "top_crop_share",
]


def _select_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required cluster columns: {missing}")
    return df[["District", *FEATURE_COLUMNS]].copy()


def get_best_k(df: pd.DataFrame, k_values: list[int] | None = None) -> int:
    if k_values is None:
        k_values = [2, 3, 4, 5, 6]

    feature_df = _select_feature_frame(df)
    matrix = StandardScaler().fit_transform(feature_df[FEATURE_COLUMNS])

    best_k = 2
    best_score = float("-inf")

    for k in k_values:
        if k >= len(feature_df):
            continue
        labels = KMeans(n_clusters=k, n_init=25, random_state=42).fit_predict(matrix)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(matrix, labels)
        if score > best_score:
            best_score = score
            best_k = k

    return best_k


def cluster_feature_matrix(df: pd.DataFrame, k: int | None = None) -> pd.DataFrame:
    feature_df = _select_feature_frame(df)
    matrix = StandardScaler().fit_transform(feature_df[FEATURE_COLUMNS])

    if k is None:
        k = get_best_k(feature_df)

    model = KMeans(n_clusters=min(k, len(feature_df)), n_init=25, random_state=42)
    labels = model.fit_predict(matrix)
    result = feature_df[["District"]].copy()
    result["cluster"] = labels
    return result


def generate_cluster_summary(df: pd.DataFrame, cluster_df: pd.DataFrame | None = None) -> pd.DataFrame:
    if cluster_df is None:
        cluster_df = cluster_feature_matrix(df)

    merged = df[["District", *FEATURE_COLUMNS, "top_crop"]].merge(cluster_df, on="District", how="left")
    summary = (
        merged.groupby("cluster", as_index=False)
        .agg(
            district_count=("District", "count"),
            avg_kharif_share=("kharif_share", "mean"),
            avg_rabi_share=("rabi_share", "mean"),
            avg_zaid_share=("zaid_share", "mean"),
            avg_other_share=("other_share", "mean"),
            avg_perennial_share=("perennial_share", "mean"),
            avg_crop_count=("crop_count", "mean"),
            avg_season_entropy=("season_entropy", "mean"),
            avg_seasonal_balance=("seasonal_balance", "mean"),
            dominant_top_crop=("top_crop", lambda s: s.mode().iloc[0] if not s.empty else "unknown")
        )
        .sort_values("cluster")
    )
    return summary


def save_cluster_artifacts(df: pd.DataFrame, output_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cluster_df = cluster_feature_matrix(df)
    summary_df = generate_cluster_summary(df, cluster_df)

    assignment_path = output_dir / "cluster_assignments.csv"
    summary_path = output_dir / "cluster_summary.csv"
    cluster_df.to_csv(assignment_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    metadata = {
        "selected_k": int(cluster_df["cluster"].nunique()),
        "silhouette_k": get_best_k(df),
        "feature_columns": FEATURE_COLUMNS,
        "district_count": int(len(df)),
    }

    metadata_path = output_dir / "cluster_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    return cluster_df, summary_df, metadata


if __name__ == "__main__":
    import sys
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    input_df = pd.read_csv(root / "data/processed/region_feature_matrix.csv")
    cluster_df, summary_df, metadata = save_cluster_artifacts(input_df, root / "data/processed/clusters")
    print(f"Clusters: {metadata['selected_k']}")
    print(cluster_df.head().to_string(index=False))
    print("\nSummary:")
    print(summary_df.to_string(index=False))
