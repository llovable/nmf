#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""패치 A/B/C 검증. 데이터 없이 도는 자족 테스트.

  python tests/test_patches.py

A) gamma가 전용 param group에서 실제로 움직이는가, gamma_nonneg가 음수를 막는가,
   녹아웃이 set_lowrank(False)로 정확히 0이 되는가
B) nmf_recon_loss의 계수 W가 FrozenNMF.encode와 같은 정의(비음수)인가
C) 자기정보 베이스라인이 타깃의 남은 관측 칸을 실제로 쓰는가
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.optim import Adam

from baselines_self import MODS, knn_impute, softimpute_concat, softimpute_per_omics
from models_nmf_tf import FrozenNMF, NMFTransformerMOCHI
from train_gate import build_nmf_basis, nmf_coefficients, nmf_recon_loss

FAILS = []


def check(name, ok, detail=""):
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}" + (f"  — {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def make_model(dims, k, gamma_init=0.3, gamma_nonneg=False, device="cpu"):
    rng = np.random.default_rng(0)
    toks = {}
    for m, d in dims.items():
        X = np.abs(rng.normal(size=(60, d))) + 0.1
        b = build_nmf_basis(X, k=k, device=device)
        toks[m] = FrozenNMF(b["H"], b["shift"], b["HHt_inv"], b["w_mean"])
    return NMFTransformerMOCHI(dims, toks, k=k, d_model=32, n_layers=1,
                               gamma_init=gamma_init, gamma_nonneg=gamma_nonneg).to(device)


def test_a():
    print("A) gamma")
    dims = {"protein": 20, "rna": 40, "methyl": 30}
    k = 6

    # A1. 전용 param group이면 gamma가 초기값에서 실제로 벗어난다.
    m = make_model(dims, k, gamma_init=0.3)
    gamma_ids = {id(m.gamma)}
    body = [p for p in m.parameters() if id(p) not in gamma_ids]
    opt = Adam([{"params": body, "lr": 3e-4, "weight_decay": 1e-5},
                {"params": [m.gamma], "lr": 1e-2, "weight_decay": 0.0}])
    xs = {mod: torch.randn(16, d) for mod, d in dims.items()}
    present = {mod: torch.ones(16, dtype=torch.bool) for mod in dims}
    start = m.gamma.detach().clone()
    for _ in range(40):
        loss = sum(m.loo_reconstruct(xs, present, t).pow(2).mean() for t in MODS)
        opt.zero_grad(); loss.backward(); opt.step()
    moved = float((m.gamma.detach() - start).abs().max())
    check("전용 param group에서 gamma가 움직인다", moved > 0.05, f"|Δγ|max = {moved:.4f}")

    # A2. 같은 스텝 수를 본체 lr로 돌리면 거의 안 움직인다 (기존 동작 재현).
    m2 = make_model(dims, k, gamma_init=0.3)
    opt2 = Adam(m2.parameters(), lr=3e-4, weight_decay=1e-5)
    s2 = m2.gamma.detach().clone()
    for _ in range(40):
        loss = sum(m2.loo_reconstruct(xs, present, t).pow(2).mean() for t in MODS)
        opt2.zero_grad(); loss.backward(); opt2.step()
    moved2 = float((m2.gamma.detach() - s2).abs().max())
    check("기존 설정에서는 gamma가 사실상 고정", moved2 < 0.02, f"|Δγ|max = {moved2:.4f}")

    # A3. gamma_nonneg면 실효 게이트가 절대 음수가 아니다.
    m3 = make_model(dims, k, gamma_init=0.3, gamma_nonneg=True)
    with torch.no_grad():
        m3.gamma.fill_(-5.0)
    eff = m3.effective_gamma()
    check("gamma_nonneg는 음수 게이트를 막는다", bool((eff >= 0).all()),
          f"raw=-5 -> eff={[round(float(v), 5) for v in eff]}")
    init_eff = make_model(dims, k, gamma_init=0.3, gamma_nonneg=True).effective_gamma()
    check("gamma_nonneg 초기 실효값이 gamma_init과 일치",
          bool(torch.allclose(init_eff, torch.full_like(init_eff, 0.3), atol=1e-4)),
          f"eff(init) = {[round(float(v), 4) for v in init_eff]}")

    # A4. 녹아웃은 gamma.zero_()가 아니라 set_lowrank(False)여야 한다.
    m4 = make_model(dims, k, gamma_init=0.3, gamma_nonneg=True)
    with torch.no_grad():
        m4.gamma.zero_()
    bad = float(m4.effective_gamma().max())
    check("gamma.zero_()는 nonneg 모델에서 녹아웃이 아니다", bad > 0.5,
          f"softplus(0) = {bad:.4f} — 게이트가 오히려 커진다")
    # w_head는 weight=0, bias=w_mean으로 초기화되므로 학습 전에는 W == w_mean이고
    # decode_dev(W) = (W − w_mean)H = 0 이다. 설계상 의도된 동작이지만, 그래서
    # 초기 모델로는 녹아웃 테스트를 할 수 없다. 먼저 그 사실 자체를 확인한다.
    m5 = make_model(dims, k, gamma_init=0.3, gamma_nonneg=True)
    h0, W0 = m5.loo_parts(xs, present, "rna")
    gap0 = float((m5.decode_target("rna", h0, W0) - m5.decoders["rna"](h0)).abs().max())
    check("학습 전 저랭크 항은 정확히 0 (w_head.weight=0, bias=w_mean)",
          gap0 < 1e-6, f"|저랭크 기여| = {gap0:.2e}")

    # 학습된 상태를 흉내내려면 w_head를 흔들어야 한다.
    with torch.no_grad():
        m5.w_head["rna"].weight.normal_(0, 0.1)
    h, W = m5.loo_parts(xs, present, "rna")
    with_lr = m5.decode_target("rna", h, W)
    lin = m5.decoders["rna"](h)
    m5.set_lowrank(False)
    without = m5.decode_target("rna", h, W)
    check("set_lowrank(False)가 저랭크 항을 정확히 제거",
          bool(torch.allclose(without, lin, atol=1e-6))
          and not torch.allclose(with_lr, lin, atol=1e-6),
          f"|with − linear| = {float((with_lr - lin).abs().max()):.4f}, "
          f"|without − linear| = {float((without - lin).abs().max()):.2e}")


def test_b():
    print("B) NMF 정규화 손실")
    rng = np.random.default_rng(1)
    X = np.abs(rng.normal(size=(80, 50))) + 0.1
    basis = build_nmf_basis(X, k=6, device="cpu")
    tok = FrozenNMF(basis["H"], basis["shift"], basis["HHt_inv"], basis["w_mean"])
    Y = torch.from_numpy(rng.normal(size=(32, 50)).astype(np.float32))

    W_old, _ = nmf_coefficients(Y, basis, nonneg=False)
    W_new, _ = nmf_coefficients(Y, basis, nonneg=True)
    W_enc = tok.encode(Y)
    neg_old = float((W_old < 0).float().mean()) * 100
    check("예전 정의는 음수 계수를 낸다 (= NMF가 아니라 선형 사영)", neg_old > 1.0,
          f"음수 비율 {neg_old:.1f}%")
    check("새 정의는 비음수", bool((W_new >= 0).all()), "음수 비율 0.0%")
    check("새 정의가 FrozenNMF.encode와 완전히 동일",
          bool(torch.allclose(W_new, W_enc, atol=1e-6)),
          f"최대 차이 {float((W_new - W_enc).abs().max()):.2e}")
    check("두 손실은 서로 다른 값 (기본값 변경이 실제로 학습을 바꾼다)",
          abs(float(nmf_recon_loss(Y, basis, nonneg=True))
              - float(nmf_recon_loss(Y, basis, nonneg=False))) > 1e-6,
          f"nonneg={float(nmf_recon_loss(Y, basis, True)):.4f} vs "
          f"old={float(nmf_recon_loss(Y, basis, False)):.4f}")


def test_c():
    print("C) 자기정보 베이스라인")
    rng = np.random.default_rng(2)
    n, K = 40, 4
    Z = rng.normal(size=(n, K))
    dims = {"protein": 12, "rna": 25, "methyl": 18}
    true = {m: Z @ rng.normal(size=(K, d)) + 0.2 * rng.normal(size=(n, d))
            for m, d in dims.items()}
    nan_tabs, masks = {}, {}
    for m in MODS:
        mk = rng.random(true[m].shape) < 0.3
        masks[m] = mk
        X = true[m].copy()
        X[mk] = np.nan
        nan_tabs[m] = X

    def zrmse(hat):
        return float(np.mean([np.sqrt(np.mean((true[m][masks[m]] - hat[m][masks[m]]) ** 2))
                              for m in MODS]))

    mean_score = float(np.mean([np.sqrt(np.mean(true[m][masks[m]] ** 2)) for m in MODS]))
    for nm, fn in (("KNN", lambda: knn_impute(nan_tabs, n_neighbors=5)),
                   ("softImpute", lambda: softimpute_per_omics(nan_tabs, rank=K)),
                   ("softImpute-concat", lambda: softimpute_concat(nan_tabs, rank=K))):
        s = zrmse(fn())
        check(f"{nm}가 평균 대치보다 낫다", s < mean_score,
              f"{s:.4f} vs mean {mean_score:.4f}")

    # 블록 결측. 타깃 오믹스 열이 통째로 비면 그 열은 Z의 영공간에 남아
    # Zhat에서도 정확히 0이 되고 반복해도 0에서 못 벗어난다. anchor(train) 행을
    # 같이 넣어야 저랭크 구조가 그 열로 전달된다.
    blk = {m: (true[m].copy() if m != "rna" else np.full_like(true[m], np.nan)) for m in MODS}
    per = softimpute_per_omics(blk, rank=K)["rna"]
    con_noanchor = softimpute_concat(blk, rank=K)["rna"]
    check("블록에서 오믹스별 softImpute는 퇴화한다 (예상된 동작)",
          np.allclose(per, 0.0, atol=1e-6), "전부 0 = 열 평균")
    check("anchor 없는 concat도 퇴화한다 (0으로 초기화한 hard-impute의 고정점)",
          np.allclose(con_noanchor, 0.0, atol=1e-6),
          "빈 열은 Z의 영공간에 남는다 — anchor_tabs가 필요한 이유")

    # 같은 잠재구조를 공유하는 anchor 행을 넣으면 복원된다.
    shared_loads = {m: rng.normal(size=(K, dims[m])) for m in MODS}
    Zt, Za = rng.normal(size=(n, K)), rng.normal(size=(60, K))
    truth = {m: Zt @ shared_loads[m] for m in MODS}
    anchor = {m: Za @ shared_loads[m] for m in MODS}
    blk2 = {m: (truth[m].copy() if m != "rna" else np.full_like(truth[m], np.nan))
            for m in MODS}
    con = softimpute_concat(blk2, rank=K, anchor_tabs=anchor)["rna"]
    e_con = float(np.sqrt(np.mean((truth["rna"] - con) ** 2)))
    e_mean = float(np.sqrt(np.mean(truth["rna"] ** 2)))
    check("anchor(train)를 넣으면 concat softImpute가 블록을 복원한다", e_con < 0.5 * e_mean,
          f"anchor 사용 {e_con:.4f} vs 평균 대치 {e_mean:.4f}")


if __name__ == "__main__":
    torch.manual_seed(0)
    test_a(); test_b(); test_c()
    print()
    if FAILS:
        print(f"실패 {len(FAILS)}건: " + "; ".join(FAILS))
        sys.exit(1)
    print("모든 검증 통과")
