#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""타깃 오믹스의 남은 관측 칸을 쓰는 대치 베이스라인.

왜 필요한가
-----------
현재 칸 결측 비교에서 MOCHI(ω=10), MIMIR, PIMMS-DAE는 타깃 오믹스의 남은
관측 칸을 본다. 반면 Ridge / TOBMI / OmicsNMF / OmiTrans는 소스 오믹스만
받는다 (eval_mcar_mnar.py의 pred_ridge, pred_tobmi, predict_z 경로).
그래서 칸 결측 표의 이득 중 얼마가 '교차 오믹스 보완' 덕이고 얼마가
'타깃 자기 정보 접근' 덕인지 구분되지 않는다.

여기 있는 네 가지는 모두 타깃 자기 정보를 쓴다. 이들을 이기면 교차 오믹스
보완이 실제로 기여한다는 근거가 된다.

  knn              : 표본 유사도 기반 (오믹스별)
  softimpute       : 저랭크 행렬 완성 (오믹스별)  <- 저랭크 NMF 잔차의 직접 대조군
  softimpute-concat: 저랭크 행렬 완성 (세 오믹스 이어붙임) <- 블록 결측에서도 동작
  ridge-self       : 세 오믹스 전부를 입력으로 받는 능형회귀 (마스킹 증강 학습)

블록 결측에서 오믹스별 방법(knn, softimpute)은 타깃 전체가 비므로 열 평균,
즉 z 공간의 0으로 퇴화한다. 이건 버그가 아니라 예상된 결과이고, 논문에서
'블록 결측은 교차 오믹스 없이는 풀 수 없다'는 근거로 그대로 보고하면 된다.

입력 규약: {modality: [n_samples, n_features] float array}, 결측은 np.nan.
출력 규약: 같은 모양의 채워진 배열. 지표는 기존 masked_metrics로 잰다.
"""

from __future__ import annotations

import numpy as np

MODS = ("protein", "rna", "methyl")


# ---------------------------------------------------------------------------
# 1) k-최근접 이웃 (표본 유사도)
# ---------------------------------------------------------------------------

def knn_impute(nan_tabs, n_neighbors=10):
    from sklearn.impute import KNNImputer
    out = {}
    for m in MODS:
        X = np.asarray(nan_tabs[m], dtype=np.float64)
        k = int(min(n_neighbors, max(2, X.shape[0] - 1)))
        imp = KNNImputer(n_neighbors=k, weights="distance")
        filled = imp.fit_transform(X)
        # 전 열이 NaN이면 sklearn이 열을 지운다. 모양을 되돌려 놓는다.
        if filled.shape[1] != X.shape[1]:
            full = np.zeros_like(X)
            keep = ~np.all(np.isnan(X), axis=0)
            full[:, keep] = filled
            filled = full
        out[m] = filled.astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# 2) softImpute (반복 특이값 축소 행렬 완성)
# ---------------------------------------------------------------------------

def _soft_impute_matrix(X, rank, n_iter=100, tol=1e-4, shrink=0.0):
    """관측 칸은 고정하고 결측 칸을 저랭크 근사로 반복 갱신한다.

    Mazumder, Hastie & Tibshirani (2010)의 softImpute. rank로 절단하고,
    shrink>0이면 특이값을 추가로 연축한다. n << p 이므로 economy SVD로 충분하다.
    """
    X = np.asarray(X, dtype=np.float64)
    obs = np.isfinite(X)
    if not obs.any():
        return np.zeros_like(X)
    Z = np.where(obs, X, 0.0)
    r = int(max(1, min(rank, min(X.shape) - 1)))
    prev = None
    Zhat = np.zeros_like(Z)
    for _ in range(n_iter):
        U, S, Vt = np.linalg.svd(Z, full_matrices=False)
        S = np.maximum(S[:r] - shrink, 0.0)
        Zhat = (U[:, :r] * S) @ Vt[:r]
        Z = np.where(obs, X, Zhat)
        cur = float(np.linalg.norm(Zhat))
        if prev is not None and abs(cur - prev) <= tol * max(prev, 1e-8):
            break
        prev = cur
    return Zhat


def softimpute_per_omics(nan_tabs, rank=20, **kw):
    return {m: _soft_impute_matrix(nan_tabs[m], rank, **kw).astype(np.float32) for m in MODS}


def softimpute_concat(nan_tabs, rank=20, anchor_tabs=None, **kw):
    """세 오믹스를 이어붙여 한 행렬로 완성한다.

    anchor_tabs(보통 train)를 주면 그 행들을 위에 쌓아 함께 분해한 뒤 버린다.
    **블록 결측에서는 이게 필수다**: 한 오믹스의 열이 통째로 비면 그 열은 Z의
    영공간에 남아 Zhat에서도 정확히 0이 되고, 반복해도 0에서 못 벗어난다
    (0으로 초기화한 hard-impute의 고정점). 해당 오믹스가 관측된 train 행을
    같이 넣어야 저랭크 구조가 그 열로 전달된다. tests/test_patches.py에서
    두 경우를 모두 확인한다.
    """
    dims = [np.asarray(nan_tabs[m]).shape[1] for m in MODS]
    X = np.concatenate([np.asarray(nan_tabs[m], dtype=np.float64) for m in MODS], axis=1)
    n_eval = X.shape[0]
    if anchor_tabs is not None:
        A = np.concatenate([np.asarray(anchor_tabs[m], dtype=np.float64) for m in MODS], axis=1)
        if A.shape[1] != X.shape[1]:
            raise ValueError(f"anchor 차원 불일치: {A.shape[1]} != {X.shape[1]}")
        X = np.concatenate([A, X], axis=0)
    Z = _soft_impute_matrix(X, rank, **kw)[-n_eval:]
    out, i = {}, 0
    for m, d in zip(MODS, dims):
        out[m] = Z[:, i:i + d].astype(np.float32)
        i += d
    return out


def choose_rank_on_val(val_tabs, val_obs, ranks=(5, 10, 20, 40, 80), rate=0.3,
                       seed=0, concat=True, anchor_tabs=None):
    """val에서 인위 마스크를 씌워 rank를 고른다. test는 건드리지 않는다.

    반환: (best_rank, {rank: 평균 z-RMSE}) — 논문 부록에 그대로 실을 수 있다.
    """
    rng = np.random.default_rng(seed)
    masks, corrupted = {}, {}
    for m in MODS:
        Y = np.asarray(val_tabs[m], dtype=np.float64)
        o = np.asarray(val_obs[m], dtype=bool)
        mk = o & (rng.random(Y.shape) < rate)
        masks[m] = mk
        Xc = Y.copy()
        Xc[~o] = np.nan
        Xc[mk] = np.nan
        corrupted[m] = Xc
    scores = {}
    for r in ranks:
        hat = softimpute_concat(corrupted, rank=r, anchor_tabs=anchor_tabs) if concat \
            else softimpute_per_omics(corrupted, rank=r)
        errs = []
        for m in MODS:
            if masks[m].any():
                d = np.asarray(val_tabs[m])[masks[m]] - hat[m][masks[m]]
                errs.append(float(np.sqrt(np.mean(d ** 2))))
        scores[r] = float(np.mean(errs)) if errs else float("nan")
    best = min(scores, key=lambda r: scores[r])
    return best, scores


# ---------------------------------------------------------------------------
# 3) 세 오믹스 전부를 입력으로 받는 능형회귀
# ---------------------------------------------------------------------------

class RidgeSelf:
    """입력 = [protein | rna | methyl] 전부 (타깃 포함, 가린 칸은 0).

    기존 Ridge 2→1은 타깃을 입력에서 뺀다. 이 판은 타깃의 남은 관측 칸도 본다.
    MOCHI가 ω=10으로 쓰는 정보와 같은 정보다.

    학습 때 같은 방식의 마스킹 증강을 넣는다. 증강 없이 깨끗한 train으로만
    적합하면 추론 시 0-채움 입력과 분포가 어긋나 불리해진다 — MOCHI는
    mask_p=0.15로 학습하므로 그쪽에만 유리해지는 것을 막는다.
    """

    def __init__(self, alphas=None, cv=3, n_aug=4, rates=(0.1, 0.3, 0.5), seed=0):
        self.alphas = np.logspace(-2, 3, 8) if alphas is None else alphas
        self.cv = cv
        self.n_aug = int(n_aug)
        self.rates = tuple(rates)
        self.seed = int(seed)
        self.models = {}
        self.dims = None

    @staticmethod
    def _pack(tabs):
        return np.concatenate([np.asarray(tabs[m], dtype=np.float32) for m in MODS], axis=1)

    def fit(self, train_tabs, train_obs=None):
        from sklearn.linear_model import RidgeCV
        rng = np.random.default_rng(self.seed)
        self.dims = {m: np.asarray(train_tabs[m]).shape[1] for m in MODS}
        Xs, Ys = [], {m: [] for m in MODS}
        for a in range(max(1, self.n_aug)):
            rate = self.rates[a % len(self.rates)] if self.n_aug > 1 else 0.3
            aug = {}
            for m in MODS:
                Y = np.asarray(train_tabs[m], dtype=np.float32)
                o = np.ones(Y.shape, dtype=bool) if train_obs is None \
                    else np.asarray(train_obs[m], dtype=bool)
                mk = o & (rng.random(Y.shape) < rate)
                aug[m] = np.where(mk, 0.0, Y).astype(np.float32)
            Xs.append(self._pack(aug))
            for m in MODS:
                Ys[m].append(np.asarray(train_tabs[m], dtype=np.float32))
        X = np.concatenate(Xs, axis=0)
        for m in MODS:
            self.models[m] = RidgeCV(alphas=self.alphas, cv=self.cv).fit(
                X, np.concatenate(Ys[m], axis=0))
        return self

    def predict(self, filled_tabs):
        X = self._pack(filled_tabs)
        return {m: self.models[m].predict(X).astype(np.float32) for m in MODS}


# ---------------------------------------------------------------------------
# 편의 함수: eval 스크립트에서 한 줄로 부르기
# ---------------------------------------------------------------------------

def all_self_predictions(nan_tabs, ridge_self=None, filled_tabs=None,
                         knn_k=10, si_rank=20, anchor_tabs=None):
    """이름 -> 예측 dict. eval 루프에서 그대로 score_method에 넘긴다.

    anchor_tabs에는 train의 z-점수 행렬을 넘긴다 (블록 결측에서 필수).
    """
    out = {
        "KNN": knn_impute(nan_tabs, n_neighbors=knn_k),
        "softImpute": softimpute_per_omics(nan_tabs, rank=si_rank),
        "softImpute-concat": softimpute_concat(nan_tabs, rank=si_rank,
                                               anchor_tabs=anchor_tabs),
    }
    if ridge_self is not None and filled_tabs is not None:
        out["Ridge-self"] = ridge_self.predict(filled_tabs)
    return out
