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


def test_d():
    """레거시 비교군의 NMF 손실 정의가 조용히 바뀌지 않는지 지킨다.

    nmf_recon_loss의 기본값이 nonneg=True로 바뀌었으므로, 논문 표의 MOCHI-v5
    수치를 만든 train_gate.py / compare_gate.py 경로는 nonneg를 **명시적으로**
    넘겨야 한다. 명시하지 않으면 v5를 재학습할 때 손실이 달라져 표가 재현되지
    않는다. 소스를 AST로 훑어 모든 호출이 nonneg를 명시하는지 확인한다.
    """
    import ast
    print("D) 레거시 비교군 손실 정의 고정")
    root = Path(__file__).resolve().parent.parent
    for fname in ("train_gate.py", "compare_gate.py"):
        tree = ast.parse((root / fname).read_text())
        calls = [n for n in ast.walk(tree)
                 if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == "nmf_recon_loss"]
        check(f"{fname}에 nmf_recon_loss 호출이 존재", len(calls) > 0, f"{len(calls)}건")
        named = [c for c in calls if any(kw.arg == "nonneg" for kw in c.keywords)]
        check(f"{fname}의 모든 호출이 nonneg를 명시", len(named) == len(calls),
              f"{len(named)}/{len(calls)}건 명시")
        pinned = [c for c in named
                  if any(kw.arg == "nonneg" and isinstance(kw.value, ast.Constant)
                         and kw.value.value is False for kw in c.keywords)]
        check(f"{fname}의 모든 호출이 nonneg=False로 고정", len(pinned) == len(calls),
              f"{len(pinned)}/{len(calls)}건 고정 — 표의 MOCHI-v5 정의 보존")

    # 보고 모형은 반대로 플래그를 따라가야 한다.
    tree = ast.parse((root / "train_nmf_tf.py").read_text())
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == "nmf_recon_loss"]
    via_flag = [c for c in calls
                if any(kw.arg == "nonneg" and isinstance(kw.value, ast.Name)
                       and kw.value.id == "nmf_nonneg" for kw in c.keywords)]
    check("train_nmf_tf.py는 --nmf_nonneg 플래그를 따른다",
          len(calls) > 0 and len(via_flag) == len(calls),
          f"{len(via_flag)}/{len(calls)}건")


def test_e():
    """그림 1 파이프라인: eval 출력 -> CSV -> 그림이 코드로 이어지는지.

    합성 biology.tsv / eval_summary.tsv를 만들어 make_source_data.py를 돌리고,
    나온 CSV로 plot_pathway_knockout.py가 실제로 렌더되는지 확인한다.
    수치가 없을 때 조용히 넘어가지 않고 멈추는지도 함께 본다.
    """
    import subprocess, sys, tempfile
    import pandas as pd
    print("E) 그림 1 원자료 생성 파이프라인")
    figdir = Path(__file__).resolve().parent.parent / "paper" / "figures"
    cohorts = ["BRCA", "LUAD", "KIRC"]

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        bio_args, abl_args = [], []
        for ci, c in enumerate(cohorts):
            rows = []
            for meth, sd, r in (("mean", 0.0, 0.0), ("Ridge-2to1", 0.75, 0.72),
                                ("MIMIR", 0.88, 0.80), ("MOCHI", 0.97 - 0.05 * ci, 0.66),
                                ("MOCHI-knockout", 0.55 - 0.05 * ci, 0.71)):
                rows.append({"modality": "rna", "method": meth,
                             "pathway_sd_ratio": sd, "pathway_r": r})
            bp = td / f"biology_{c}.tsv"
            pd.DataFrame(rows).to_csv(bp, sep="\t", index=False)
            bio_args.append(f"{c}={bp}")

            arows = []
            for om, rep, kno in (("rna", 0.80, 0.81), ("methyl", 0.62, 0.63),
                                 ("protein", 0.43, 0.41)):
                for meth, setting, z in (("MOCHI", "self-w10", rep),
                                         ("MOCHI-knockout", "ablation", kno)):
                    arows.append({"method": meth, "setting": setting,
                                  "mechanism": "block", "split": "test", "rate": 1.0,
                                  "missing": om, "modality": om, "z_rmse": z})
            ap_ = td / f"ablate_{c}.tsv"
            pd.DataFrame(arows).to_csv(ap_, sep="\t", index=False)
            abl_args.append(f"{c}={ap_}")

        out_csv = td / "source_pathway_knockout.csv"
        res = subprocess.run(
            [sys.executable, str(figdir / "make_source_data.py"),
             "--biology", *bio_args, "--ablation", *abl_args, "--out", str(out_csv)],
            capture_output=True, text=True)
        check("make_source_data.py가 정상 종료", res.returncode == 0,
              res.stderr.strip().splitlines()[-1] if res.returncode else "")
        if res.returncode:
            return

        df = pd.read_csv(out_csv)
        check("Panel A에 두 지표가 모두 있다",
              set(df[df.panel == "A"].metric) == {"pathway_sd_ratio", "pathway_r"},
              f"{sorted(set(df[df.panel == 'A'].metric))}")
        check("Panel B에 코호트 3 × 지표 2 × 조건 2 = 12행",
              len(df[df.panel == "B"]) == 12, f"{len(df[df.panel == 'B'])}행")
        c_prot = df[(df.panel == "C") & (df.omics == "protein")].value
        check("Panel C의 delta가 knockout − MOCHI로 계산된다",
              bool((c_prot < 0).all()),
              f"protein delta = {[round(v, 4) for v in c_prot]} (0.41 − 0.43 = −0.02)")

        # 스크립트를 실제 파일로 복사해 그대로 실행한다 (HERE = 스크립트 위치).
        import shutil
        sandbox = td / "figs"
        sandbox.mkdir()
        shutil.copy(figdir / "plot_pathway_knockout.py", sandbox)
        shutil.copy(out_csv, sandbox / "source_pathway_knockout.csv")
        res3 = subprocess.run([sys.executable, str(sandbox / "plot_pathway_knockout.py")],
                              capture_output=True, text=True)
        png = sandbox / "fig1_pathway_knockout.png"
        check("생성된 CSV로 그림이 렌더된다", res3.returncode == 0 and png.exists(),
              (res3.stderr.strip().splitlines()[-1] if res3.returncode
               else f"{png.stat().st_size} bytes"))

        # 없는 행이 있으면 조용히 그려지지 않고 멈춰야 한다.
        df[df.metric != "pathway_r"].to_csv(sandbox / "source_pathway_knockout.csv",
                                            index=False)
        res2 = subprocess.run([sys.executable, str(sandbox / "plot_pathway_knockout.py")],
                              capture_output=True, text=True)
        out = res2.stdout + res2.stderr
        check("r 행이 빠지면 그림이 조용히 그려지지 않고 멈춘다",
              res2.returncode != 0 and "pathway_r" in out and "make_source_data" in out,
              "재생성 안내와 함께 종료")


if __name__ == "__main__":
    torch.manual_seed(0)
    test_a(); test_b(); test_c(); test_d(); test_e()
    print()
    if FAILS:
        print(f"실패 {len(FAILS)}건: " + "; ".join(FAILS))
        sys.exit(1)
    print("모든 검증 통과")
