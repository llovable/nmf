"""
Mean/Median imputation (특징별).
"""
from __future__ import annotations

import pandas as pd


def impute_mean(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(lambda col: col.fillna(col.mean()), axis=0)


def impute_median(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(lambda col: col.fillna(col.median()), axis=0)
