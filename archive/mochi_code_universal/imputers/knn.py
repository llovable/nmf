"""
KNN imputation (scikit-learn 필요).
"""
from __future__ import annotations

import pandas as pd


def impute_knn(df: pd.DataFrame, n_neighbors: int = 5, weights: str = "uniform") -> pd.DataFrame:
    try:
        from sklearn.impute import KNNImputer
    except Exception as e:
        raise ImportError("scikit-learn이 필요합니다. `pip install scikit-learn` 후 사용하세요.") from e

    imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights)
    arr = imputer.fit_transform(df.values)
    return pd.DataFrame(arr, index=df.index, columns=df.columns)
