"""
통일된 인터페이스로 결측 보간 모델 호출.
"""
from __future__ import annotations

from typing import Any, Dict

import pandas as pd

from imputers.mean_median import impute_mean, impute_median
from imputers.knn import impute_knn
from imputers.softimpute import impute_softimpute
from imputers.missforest import impute_missforest


SUPPORTED_MODELS = {
    "mean": impute_mean,
    "median": impute_median,
    "knn": impute_knn,
    "softimpute": impute_softimpute,
    "missforest": impute_missforest,
}


def impute(df: pd.DataFrame, model: str, **kwargs: Any) -> pd.DataFrame:
    """
    공통 인터페이스:
      impute(df, model="knn", n_neighbors=5)
    """
    model = model.lower().strip()
    if model not in SUPPORTED_MODELS:
        raise ValueError(f"지원하지 않는 모델: {model}. 지원 목록: {list(SUPPORTED_MODELS.keys())}")
    return SUPPORTED_MODELS[model](df, **kwargs)


def list_models() -> Dict[str, str]:
    return {
        "mean": "특징별 평균으로 결측치 채움",
        "median": "특징별 중앙값으로 결측치 채움",
        "knn": "KNN 기반 보간 (scikit-learn 필요)",
        "softimpute": "저랭크 행렬 보간 (fancyimpute 필요)",
        "missforest": "랜덤포레스트 기반 보간 (missingpy 필요)",
    }
