#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NMF-Transformer 학습.

1) 오믹스별 마스크 AE
2) 고정 NMF 토큰 + Transformer LOO + 자기 AE 재구성 + 약한 조건부 WGAN
조기 종료는 val 블록 z-RMSE.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

from models import ConditionalCritic
from models_nmf_tf import (
    D_MODEL, K_DEFAULT, MODS, FrozenNMF, NMFTransformerMOCHI, predict_nmf_tf,
)
from models_shared import contrastive_loss, drop_modalities, mask_cells, mse_valid
from train_gate import TripleSplitDataset, build_nmf_basis, nmf_recon_loss
from train_shared import pretrain_aes


def _batch_tensors(batch, device):
    xs = {
        "protein": batch["x_prot"].to(device),
        "rna": batch["x_rna"].to(device),
        "methyl": batch["x_methy"].to(device),
    }
    obs = {
        "protein": (1.0 - batch["m_prot"].to(device)) > 0.5,
        "rna": (1.0 - batch["m_rna"].to(device)) > 0.5,
        "methyl": (1.0 - batch["m_methy"].to(device)) > 0.5,
    }
    return xs, obs


def _gp(D, real, fake, cond, device):
    alpha = torch.rand(real.size(0), 1, device=device).expand_as(real)
    inter = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    out = D(inter, cond)
    grad = torch.autograd.grad(
        out, inter, grad_outputs=torch.ones_like(out),
        create_graph=True, retain_graph=True, only_inputs=True)[0]
    return ((grad.norm(2, dim=1) - 1) ** 2).mean()


def _conds(xs, tgt):
    others = [m for m in MODS if m != tgt]
    return torch.cat([xs[m] for m in others], dim=1)


def train_epoch(model, critics, bases, loader, opt_g, opt_d, device,
                mask_p=0.15, drop_p=0.4, lambda_loo=1.5, lambda_con=0.2,
                lambda_nmf=0.1, lambda_w=0.3, alpha=0.5, gan_to_mse=0.1,
                n_critic=1, lambda_gp=10.0, nmf_nonneg=True):
    model.train()
    for D in critics.values():
        D.train()
    sums = {"total": 0.0, "recon": 0.0, "loo": 0.0, "con": 0.0, "nmf": 0.0,
            "w": 0.0, "wgan": 0.0, "D": 0.0}
    n = 0
    for batch in loader:
        xs, obs = _batch_tensors(batch, device)
        b = xs["rna"].size(0)
        present = {m: torch.ones(b, dtype=torch.bool, device=device) for m in MODS}
        present = drop_modalities(present, drop_p)
        xs_in, extra = {}, {}
        for m in MODS:
            xin, ex = mask_cells(xs[m], obs[m], mask_p)
            xin = torch.where(present[m].view(-1, 1), xin, torch.zeros_like(xin))
            xs_in[m], extra[m] = xin, ex
        present_xs = {m: xs_in[m] for m in MODS if present[m].any()}

        use_gan = gan_to_mse > 0 and b >= 8
        if use_gan:
            for _ in range(n_critic):
                opt_d.zero_grad()
                loss_d = xs["rna"].new_zeros(())
                for tgt in MODS:
                    others = [m for m in MODS if m != tgt]
                    can = present[tgt] & torch.stack([present[m] for m in others], 0).any(0)
                    if can.sum() < 2:
                        continue
                    with torch.no_grad():
                        fake = model.loo_reconstruct(present_xs, present, tgt)[can]
                    real = xs[tgt][can]
                    cond = _conds({m: xs_in[m] for m in MODS}, tgt)[can]
                    D = critics[tgt]
                    loss_d = loss_d + (
                        -(D(real, cond).mean() - D(fake, cond).mean())
                        + lambda_gp * _gp(D, real, fake, cond, device)
                    )
                if loss_d.requires_grad:
                    loss_d.backward()
                    opt_d.step()
                    sums["D"] += float(loss_d.item())

        own = model.reconstruct_own(present_xs)
        rloss, n_r = 0.0, 0
        for m in MODS:
            if m not in own:
                continue
            keep = present[m].view(-1, 1) & obs[m]
            masked = present[m].view(-1, 1) & extra[m]
            overall = mse_valid(own[m], xs[m], keep)
            masked_mse = mse_valid(own[m], xs[m], masked) if masked.any() else overall
            rloss = rloss + alpha * masked_mse + (1.0 - alpha) * overall
            n_r += 1
        rloss = rloss / max(n_r, 1)

        lloss, n_l = 0.0, 0
        wloss, n_w = rloss.new_zeros(()), 0
        hats = {}
        for tgt in MODS:
            others = [m for m in MODS if m != tgt]
            can = present[tgt] & torch.stack([present[m] for m in others], 0).any(0)
            if can.sum() < 1:
                continue
            h_t, W_t = model.loo_parts(present_xs, present, tgt)
            hat = model.decode_target(tgt, h_t, W_t)
            hats[tgt] = (hat, can)
            lloss = lloss + mse_valid(hat[can], xs[tgt][can], obs[tgt][can])
            n_l += 1
            if W_t is not None:
                # 융합 좌표가 타깃의 성분 구성을 맞히도록 감독한다.
                with torch.no_grad():
                    W_ref = model.tokenizers[tgt].encode(xs[tgt])
                wloss = wloss + F.mse_loss(W_t[can], W_ref[can])
                n_w += 1
        lloss = lloss / max(n_l, 1) if n_l else rloss.new_zeros(())
        wloss = wloss / max(n_w, 1) if n_w else wloss

        nloss = rloss.new_zeros(())
        n_n = 0
        for m, pred in own.items():
            keep = present[m]
            if keep.any():
                nloss = nloss + nmf_recon_loss(pred[keep], bases[m], nonneg=nmf_nonneg)
                n_n += 1
        for tgt, (hat, can) in hats.items():
            if can.any():
                nloss = nloss + nmf_recon_loss(hat[can], bases[tgt], nonneg=nmf_nonneg)
                n_n += 1
        nloss = nloss / max(n_n, 1) if n_n else nloss

        hs = model.encode_h(present_xs)
        closs = contrastive_loss({m: model.fuse.proj_h[m](h) for m, h in hs.items()}, present)

        g_wgan = rloss.new_zeros(())
        if use_gan and hats:
            for tgt, (hat, can) in hats.items():
                if can.sum() < 2:
                    continue
                cond = _conds({m: xs_in[m] for m in MODS}, tgt)[can]
                g_wgan = g_wgan - critics[tgt](hat[can], cond).mean()
            if g_wgan.abs() > 0:
                target = gan_to_mse * (rloss + lloss).detach().abs()
                g_wgan = (target / g_wgan.detach().abs().clamp_min(1e-8)) * g_wgan

        loss = (rloss + lambda_loo * lloss + lambda_con * closs
                + lambda_nmf * nloss + lambda_w * wloss + g_wgan)
        opt_g.zero_grad()
        loss.backward()
        opt_g.step()
        sums["total"] += float(loss.item())
        sums["recon"] += float(rloss.item())
        sums["loo"] += float(lloss.item())
        sums["con"] += float(closs.item())
        sums["nmf"] += float(nloss.item())
        sums["w"] += float(wloss.item())
        sums["wgan"] += float(g_wgan.item()) if torch.is_tensor(g_wgan) else float(g_wgan)
        n += 1
    return {k: v / max(n, 1) for k, v in sums.items()}


@torch.no_grad()
def block_zrmse(model, ds, device):
    """블록 결측 val 지표. 원래 결측(NaN→0으로 채운 칸)은 채점에서 뺀다.

    예전 판은 전체 칸을 평균해서, NaN을 0으로 채운 자리까지 '맞춘 것'으로
    세었다. 조기 종료 기준이 결측률에 따라 낙관적으로 치우친다.
    """
    tabs = {m: getattr(ds, {"protein": "prot_f", "rna": "rna_f", "methyl": "methy_f"}[m]) for m in MODS}
    obs = {m: getattr(ds, {"protein": "m_prot", "rna": "m_rna", "methyl": "m_methy"}[m]) < 0.5
           for m in MODS}
    per = {}
    for tgt in MODS:
        hat = predict_nmf_tf(model, tabs, device, missing=tgt)
        sel = obs[tgt]
        per[tgt] = float(np.sqrt(np.mean((tabs[tgt][sel] - hat[tgt][sel]) ** 2))) if sel.any() \
            else float("nan")
    per["avg"] = float(np.nanmean([per[t] for t in MODS]))
    return per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/home/dyan/nmf/mochi_code/processed_data/gate_brca")
    ap.add_argument("--save_dir", default="/home/dyan/nmf/mochi_code/results/current/gate_nmf_tf_h")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--k", type=int, default=K_DEFAULT)
    ap.add_argument("--d_model", type=int, default=D_MODEL)
    ap.add_argument("--n_layers", type=int, default=2)
    ap.add_argument("--epochs1", type=int, default=70)
    ap.add_argument("--epochs2", type=int, default=150)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--gan_to_mse", type=float, default=0.1)
    ap.add_argument("--no_nmf_tokens", action="store_true")
    ap.add_argument("--no_transformer", action="store_true")
    ap.add_argument("--no_lowrank", action="store_true",
                    help="저랭크 NMF 잔차 경로와 계수 보조 손실을 끈다")
    ap.add_argument("--gamma_init", type=float, default=0.3)
    ap.add_argument("--gamma_lr", type=float, default=1e-2,
                    help="저랭크 게이트 gamma 전용 학습률. 본체 lr(3e-4)로는 스칼라 3개가 "
                         "학습 중에 초기값에서 거의 움직이지 못한다")
    ap.add_argument("--gamma_nonneg", action="store_true",
                    help="gamma = softplus(raw)로 두어 음수 게이트를 막는다. "
                         "비음수 성분 해석을 유지하려면 켠다")
    ap.add_argument("--nmf_nonneg", dest="nmf_nonneg", action="store_true", default=True,
                    help="NMF 정규화 손실의 계수 W에 ReLU를 적용한다 (FrozenNMF.encode와 동일 정의)")
    ap.add_argument("--no_nmf_nonneg", dest="nmf_nonneg", action="store_false",
                    help="예전 동작: W에 ReLU를 안 쓴다. 이때 이 항은 NMF가 아니라 "
                         "span(H)로의 선형 사영이다")
    ap.add_argument("--lambda_w", type=float, default=0.3)
    ap.add_argument("--n_train", type=int, default=0,
                    help="0보다 크면 train을 그만큼만 부분표집한다 (소표본 실험용)")
    ap.add_argument("--seed", type=int, default=0,
                    help="초기화·배치 순서 시드. 제거 실험의 재현 변동을 재는 데 쓴다")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"device={device} seed={args.seed}")
    train_ds = TripleSplitDataset(args.data_dir, "train")
    if 0 < args.n_train < len(train_ds):
        sub = np.random.default_rng(42).choice(len(train_ds), size=args.n_train, replace=False)
        for attr in ("rna_f", "prot_f", "methy_f", "m_rna", "m_prot", "m_methy"):
            setattr(train_ds, attr, getattr(train_ds, attr)[sub])
        print(f"subsampled train to n={len(train_ds)}")
    val_ds = TripleSplitDataset(args.data_dir, "val", stats=train_ds.stats)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    dims = {"rna": train_ds.rna_f.shape[1], "protein": train_ds.prot_f.shape[1],
            "methyl": train_ds.methy_f.shape[1]}
    print(f"n={len(train_ds)}/{len(val_ds)} dims={dims} k={args.k}")

    print("=== fitting train NMF dictionaries ===")
    arrays = {"rna": train_ds.rna_f, "protein": train_ds.prot_f, "methyl": train_ds.methy_f}
    bases = {m: build_nmf_basis(arrays[m], k=args.k, device=device) for m in MODS}
    tokenizers = {
        m: FrozenNMF(bases[m]["H"], bases[m]["shift"], bases[m]["HHt_inv"],
                     bases[m]["w_mean"]) for m in MODS
    }

    print("=== phase1 modality AEs ===")
    encs, decs = pretrain_aes(train_loader, val_loader, dims, device, epochs=args.epochs1)

    model = NMFTransformerMOCHI(
        dims, tokenizers, k=args.k, d_model=args.d_model, n_layers=args.n_layers,
        use_nmf_tokens=not args.no_nmf_tokens,
        use_transformer=not args.no_transformer,
        use_lowrank=not args.no_lowrank,
        gamma_init=args.gamma_init,
        gamma_nonneg=args.gamma_nonneg,
    ).to(device)
    for m in MODS:
        model.encoders[m].load_state_dict(encs[m].state_dict())
        model.decoders[m].load_state_dict(decs[m].state_dict())

    critics = {
        "protein": ConditionalCritic(dims["protein"], dims["rna"] + dims["methyl"]).to(device),
        "rna": ConditionalCritic(dims["rna"], dims["protein"] + dims["methyl"]).to(device),
        "methyl": ConditionalCritic(dims["methyl"], dims["rna"] + dims["protein"]).to(device),
    }
    # gamma는 스칼라 3개다. 109만 개 파라미터와 같은 param group에 lr=3e-4로 두면
    # Adam의 스텝당 이동 상한이 대략 lr이라 학습 내내 초기값 근처에 머문다
    # (실측: 40 epoch에서 |Δgamma| < 0.03, gamma_init을 바꾸면 결과가 따라 바뀜).
    # 전용 param group으로 분리해 실제로 학습되게 한다. weight_decay는 걸지 않는다.
    gamma_params = [model.gamma]
    gamma_ids = {id(p) for p in gamma_params}
    body_params = [p for p in model.parameters() if id(p) not in gamma_ids]
    opt_g = Adam(
        [
            {"params": body_params, "lr": 3e-4, "weight_decay": 1e-5},
            {"params": gamma_params, "lr": args.gamma_lr, "weight_decay": 0.0},
        ]
    )
    opt_d = Adam([p for D in critics.values() for p in D.parameters()], lr=1e-4, betas=(0.5, 0.9))

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    print("=== phase2 AE-hidden LOO + NMF tokens + WGAN ===")
    print(f"gamma: init={args.gamma_init} lr={args.gamma_lr} nonneg={args.gamma_nonneg}  "
          f"nmf_loss_nonneg={args.nmf_nonneg}")
    best, pat = float("inf"), 0
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params G={n_params:,}")
    for ep in range(1, args.epochs2 + 1):
        tr = train_epoch(model, critics, bases, train_loader, opt_g, opt_d, device,
                         gan_to_mse=args.gan_to_mse, lambda_w=args.lambda_w,
                         nmf_nonneg=args.nmf_nonneg)
        met = block_zrmse(model, val_ds, device)
        mark = ""
        if met["avg"] < best:
            best, pat, mark = met["avg"], 0, " *"
            torch.save({
                "model": model.state_dict(), "dims": dims, "k": args.k,
                "d_model": args.d_model, "n_heads": 4, "n_layers": args.n_layers,
                "use_nmf_tokens": not args.no_nmf_tokens,
                "use_transformer": not args.no_transformer,
                "use_lowrank": not args.no_lowrank,
                # 재현에 필요한 설정을 전부 남긴다. 논문 표에 그대로 옮길 수 있어야 한다.
                "gamma_nonneg": args.gamma_nonneg,
                "gamma_init": args.gamma_init,
                "gamma_lr": args.gamma_lr,
                "nmf_nonneg": args.nmf_nonneg,
                "seed": args.seed,
                "n_train": len(train_ds),
                "epoch": ep, "val": met,
            }, save_dir / "nmf_tf_best.ckpt")
        else:
            pat += 1
        gam = ",".join(f"{float(g):.3f}" for g in model.effective_gamma().cpu())
        print(f"ep {ep:03d} tot={tr['total']:.4f} recon={tr['recon']:.4f} "
              f"loo={tr['loo']:.4f} con={tr['con']:.4f} nmf={tr['nmf']:.4f} "
              f"w={tr['w']:.4f} wgan={tr['wgan']:.4f} gamma={gam}  "
              f"val zRMSE avg={met['avg']:.4f} "
              f"P={met['protein']:.4f} R={met['rna']:.4f} M={met['methyl']:.4f}{mark}")
        if ep > 15 and pat >= args.patience:
            print(f"early stop at {ep}")
            break
    print(f"best val block zRMSE={best:.4f}  saved {save_dir / 'nmf_tf_best.ckpt'}")


if __name__ == "__main__":
    main()
