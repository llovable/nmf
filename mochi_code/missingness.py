#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MCAR / MNAR 칸 마스크. MNAR는 MIMIR `_make_mask_low_vals`(순위 역가중)와 같다."""

import numpy as np


def mcar_mask(obs, rate, rng):
    """관측 칸 중 rate 비율을 균등하게 고른다."""
    cand = np.flatnonzero(np.asarray(obs, dtype=bool).ravel())
    n = int(len(cand) * rate)
    mask = np.zeros(obs.shape, dtype=bool)
    if n <= 0 or len(cand) == 0:
        return mask
    chosen = rng.choice(cand, size=n, replace=False)
    mask.ravel()[chosen] = True
    return mask


def mnar_mask(X, obs, rate, rng, alpha=1.0, eps=1e-12):
    """낮은 값일수록 가릴 확률이 높다 (MIMIR rank-inverse, 비복원 K개)."""
    eligible = np.asarray(obs, dtype=bool) & np.isfinite(X)
    idx = np.flatnonzero(eligible.ravel())
    mask = np.zeros(X.shape, dtype=bool)
    if idx.size == 0:
        return mask
    k = int(np.floor(rate * idx.size))
    if k <= 0:
        return mask
    v = np.asarray(X, dtype=np.float64).ravel()[idx]
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(order.size)
    low_score = (order.size - 1 - ranks).astype(np.float64)
    low_score = low_score / max(1.0, (order.size - 1))
    w = (low_score + eps) ** alpha
    w = w / w.sum()
    chosen = rng.choice(idx, size=k, replace=False, p=w)
    mask.ravel()[chosen] = True
    return mask


def block_mask(obs):
    """해당 모달리티의 관측 칸 전부."""
    return np.asarray(obs, dtype=bool).copy()
