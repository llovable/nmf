"""
SoftImpute (Matrix Completion).
fancyimpute가 설치되어 있으면 사용, 없으면 안내 메시지 제공.
"""
from __future__ import annotations

import pandas as pd


def impute_softimpute(df: pd.DataFrame, max_iters: int = 100, shrinkage_value: float | None = None) -> pd.DataFrame:
    try:
        from fancyimpute import SoftImpute
    except Exception as e:
        raise ImportError("fancyimpute가 필요합니다. `pip install fancyimpute` 후 사용하세요.") from e

    model = SoftImpute(max_iters=max_iters, shrinkage_value=shrinkage_value)
    arr = model.fit_transform(df.values)
    return pd.DataFrame(arr, index=df.index, columns=df.columns)
