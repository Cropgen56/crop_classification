import pandas as pd

from src.clustering.cluster_regions import cluster_feature_matrix, get_best_k


def test_get_best_k_returns_reasonable_value():
    df = pd.read_csv("data/processed/region_feature_matrix.csv")
    k = get_best_k(df)
    assert 2 <= k <= 6


def test_cluster_feature_matrix_returns_assignments():
    df = pd.read_csv("data/processed/region_feature_matrix.csv")
    result = cluster_feature_matrix(df)
    assert {"District", "cluster"}.issubset(set(result.columns))
    assert not result.empty
    assert result["cluster"].nunique() >= 2
