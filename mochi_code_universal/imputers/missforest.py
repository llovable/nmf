"""
MissForest imputation (missingpy 필요).
"""
from __future__ import annotations

import pandas as pd


def impute_missforest(df: pd.DataFrame, n_estimators: int = 100, random_state: int = 0) -> pd.DataFrame:
    try:
        from missingpy import MissForest
    except Exception as e:
        raise ImportError("missingpy가 필요합니다. `pip install missingpy` 후 사용하세요.") from e

    model = MissForest(n_estimators=n_estimators, random_state=random_state)
    arr = model.fit_transform(df.values)
    return pd.DataFrame(arr, index=df.index, columns=df.columns)
