#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
게이트 학습 v5: v4 구조 + WGAN 스케일 균형.

- 구조는 v4와 같음: 조건부 critic, 어텐션, 선형 잔차, 고정 NMF, GAN은 1 에폭부터.
- critic에 spectral norm (Miyato et al.)으로 점수를 O(1) 근처로 묶음.
- 배치마다 |L_GAN| ≈ gan_to_mse * |λ_mse L_MSE| 가 되게 WGAN 항을 스케일.
  v4에서 WGAN이 125까지 커져 MSE가 꺼진 것을 고침.
- 지표: z-RMSE와 특징 평균/분산 거리.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from sklearn.decomposition import NMF

from models import Generator, ConditionalCritic, count_parameters


def build_nmf_basis(X, k=20, ridge=1e-3, device="cpu"):
    """z-score 행렬 [n, p]를 비음수화한 뒤 고정 NMF 사전 H를 만든다."""
    X = np.nan_to_num(np.asarray(X, dtype=np.float64), nan=0.0)
    mins = X.min(axis=0, keepdims=True)
    shift = np.where(mins < 0, -mins, 0.0)
    Xnn = np.clip(X + shift + 1e-8, 1e-8, None)
    k = int(max(2, min(k, Xnn.shape[0] - 1, Xnn.shape[1])))
    model = NMF(n_components=k, init="nndsvda", random_state=1111, max_iter=500)
    model.fit(Xnn)
    H = model.components_  # [k, p]
    Ht = torch.tensor(H, dtype=torch.float32, device=device)
    HHt_inv = torch.linalg.inv(Ht @ Ht.T + ridge * torch.eye(k, device=device))
    shift_t = torch.tensor(shift.squeeze(0), dtype=torch.float32, device=device)
    Xnn_t = torch.tensor(Xnn, dtype=torch.float32, device=device)
    w_mean = torch.relu(Xnn_t @ Ht.T @ HHt_inv).mean(0)
    return {"H": Ht, "HHt_inv": HHt_inv, "shift": shift_t, "w_mean": w_mean}


def nmf_coefficients(Y, basis, nonneg=True):
    """예측 Y의 NMF 계수 W. FrozenNMF.encode와 **같은** 정의를 쓴다.

    nonneg=False면 능형 최소제곱 해를 그대로 쓰므로 W에 음수가 섞이고,
    이 경우 아래 손실은 NMF가 아니라 span(H)로의 선형 사영(=저랭크 부분공간
    정규화)이 된다. 비음수성이 NMF 해석의 근거이므로 기본값은 True다.
    """
    Ys = torch.clamp(Y + basis["shift"], min=1e-8)
    W = (Ys @ basis["H"].T) @ basis["HHt_inv"]
    if nonneg:
        W = torch.nn.functional.relu(W)
    return W, Ys


def nmf_recon_loss(Y, basis, nonneg=True):
    W, Ys = nmf_coefficients(Y, basis, nonneg=nonneg)
    rec = W @ basis["H"]
    return torch.nn.functional.mse_loss(rec, Ys)


def _zscore(df, mu, sd):
    z = (df - mu) / sd
    return z


class TripleSplitDataset(Dataset):
    def __init__(self, data_dir, split, stats=None):
        data_dir = Path(data_dir)
        rna = pd.read_csv(data_dir / f"rna.{split}.tsv", sep="\t", index_col=0).T.astype(np.float32)
        prot = pd.read_csv(data_dir / f"protein.{split}.tsv", sep="\t", index_col=0).T.astype(np.float32)
        methy = pd.read_csv(data_dir / f"methy.{split}.tsv", sep="\t", index_col=0).T.astype(np.float32)
        idx = rna.index.intersection(prot.index).intersection(methy.index)
        rna, prot, methy = rna.loc[idx], prot.loc[idx], methy.loc[idx]

        if stats is None:
            def _mu_sd(df):
                mu = df.mean(axis=0)
                sd = df.std(axis=0).replace(0.0, 1.0)
                sd = sd.fillna(1.0)
                return mu, sd
            stats = {
                "rna": _mu_sd(rna),
                "prot": _mu_sd(prot),
                "methy": _mu_sd(methy),
            }
        self.stats = stats
        rna = _zscore(rna, *stats["rna"])
        prot = _zscore(prot, *stats["prot"])
        methy = _zscore(methy, *stats["methy"])

        self.m_rna = rna.isna().to_numpy(np.float32)
        self.m_prot = prot.isna().to_numpy(np.float32)
        self.m_methy = methy.isna().to_numpy(np.float32)
        self.rna_f = rna.fillna(0.0).to_numpy(np.float32)
        self.prot_f = prot.fillna(0.0).to_numpy(np.float32)
        self.methy_f = methy.fillna(0.0).to_numpy(np.float32)

    def __len__(self):
        return self.rna_f.shape[0]

    def __getitem__(self, i):
        return {
            "x_rna": torch.from_numpy(self.rna_f[i]),
            "x_prot": torch.from_numpy(self.prot_f[i]),
            "x_methy": torch.from_numpy(self.methy_f[i]),
            "m_rna": torch.from_numpy(self.m_rna[i]),
            "m_prot": torch.from_numpy(self.m_prot[i]),
            "m_methy": torch.from_numpy(self.m_methy[i]),
        }


class WGAN_GP_Trainer_Tri:
    def __init__(self, Gp, Gr, Gm, Dp, Dr, Dm, device,
                 lr_g=1e-4, lr_d=2e-4, lambda_gp=10.0, n_critic=5,
                 mse_weight=0.1, gan_weight=1.0, nmf_weight=0.1, nmf_basis=None,
                 gan_to_mse=0.3):
        self.Gp, self.Gr, self.Gm = Gp.to(device), Gr.to(device), Gm.to(device)
        self.Dp, self.Dr, self.Dm = Dp.to(device), Dr.to(device), Dm.to(device)
        self.device = device
        self.lambda_gp, self.n_critic = lambda_gp, n_critic
        self.mse_weight = mse_weight
        self.gan_weight = gan_weight
        self.nmf_weight = nmf_weight
        self.gan_to_mse = gan_to_mse
        self.nmf_basis = nmf_basis or {}
        self.optimizer_g = optim.Adam(
            list(Gp.parameters()) + list(Gr.parameters()) + list(Gm.parameters()),
            lr=lr_g, betas=(0.5, 0.9))
        self.optimizer_d = optim.Adam(
            list(Dp.parameters()) + list(Dr.parameters()) + list(Dm.parameters()),
            lr=lr_d, betas=(0.5, 0.9))

    def _gp(self, D, real, fake, cond):
        alpha = torch.rand(real.size(0), 1, device=self.device).expand_as(real)
        inter = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
        out = D(inter, cond)
        grad = torch.autograd.grad(
            out, inter, grad_outputs=torch.ones_like(out),
            create_graph=True, retain_graph=True, only_inputs=True)[0]
        return ((grad.norm(2, dim=1) - 1) ** 2).mean()

    def _conds(self, rna, prot, methy):
        return torch.cat([rna, methy], dim=1), torch.cat([prot, methy], dim=1), torch.cat([rna, prot], dim=1)

    def _forward(self, rna, prot, methy):
        c_p, c_r, c_m = self._conds(rna, prot, methy)
        return self.Gp(c_p, src=None), self.Gr(c_r, src=None), self.Gm(c_m, src=None)

    def train_step(self, batch, use_gan=True):
        rna = batch["x_rna"].to(self.device)
        prot = batch["x_prot"].to(self.device)
        methy = batch["x_methy"].to(self.device)
        obs_p = 1.0 - batch["m_prot"].to(self.device)
        obs_r = 1.0 - batch["m_rna"].to(self.device)
        obs_m = 1.0 - batch["m_methy"].to(self.device)
        c_p, c_r, c_m = self._conds(rna, prot, methy)

        loss_d_val = 0.0
        n_crit = self.n_critic if use_gan else 0
        for _ in range(n_crit):
            self.optimizer_d.zero_grad()
            y_p, y_r, y_m = [t.detach() for t in self._forward(rna, prot, methy)]
            loss_d = (
                -(self.Dp(prot, c_p).mean() - self.Dp(y_p, c_p).mean()) + self.lambda_gp * self._gp(self.Dp, prot, y_p, c_p)
                + -(self.Dr(rna, c_r).mean() - self.Dr(y_r, c_r).mean()) + self.lambda_gp * self._gp(self.Dr, rna, y_r, c_r)
                + -(self.Dm(methy, c_m).mean() - self.Dm(y_m, c_m).mean()) + self.lambda_gp * self._gp(self.Dm, methy, y_m, c_m)
            )
            loss_d.backward()
            self.optimizer_d.step()
            loss_d_val = float(loss_d.item())

        self.optimizer_g.zero_grad()
        y_p, y_r, y_m = self._forward(rna, prot, methy)
        mse = (
            ((y_p - prot) * obs_p).pow(2).sum() / obs_p.sum().clamp_min(1.0)
            + ((y_r - rna) * obs_r).pow(2).sum() / obs_r.sum().clamp_min(1.0)
            + ((y_m - methy) * obs_m).pow(2).sum() / obs_m.sum().clamp_min(1.0)
        )
        g_wgan = torch.tensor(0.0, device=self.device)
        if use_gan:
            g_wgan = -(self.Dp(y_p, c_p).mean() + self.Dr(y_r, c_r).mean() + self.Dm(y_m, c_m).mean())
        nmf = torch.tensor(0.0, device=self.device)
        if self.nmf_weight > 0 and self.nmf_basis:
            # nonneg=False로 못 박는다. 이 경로는 논문 표의 MOCHI-v5 비교군을
            # 만든 코드다. nmf_recon_loss의 기본값이 True로 바뀌었으므로,
            # 명시하지 않으면 v5를 재학습할 때 손실 정의가 조용히 달라져
            # 표의 수치가 더 이상 재현되지 않는다. 보고 모형(train_nmf_tf.py)만
            # 수정된 비음수 정의를 쓴다.
            nmf = (
                nmf_recon_loss(y_p, self.nmf_basis["protein"], nonneg=False)
                + nmf_recon_loss(y_r, self.nmf_basis["rna"], nonneg=False)
                + nmf_recon_loss(y_m, self.nmf_basis["methyl"], nonneg=False)
            )
        mse_term = self.mse_weight * mse
        nmf_term = self.nmf_weight * nmf
        gan_scale = torch.tensor(0.0, device=self.device)
        if use_gan and self.gan_to_mse > 0:
            target = self.gan_to_mse * mse_term.detach().abs()
            gan_scale = target / g_wgan.detach().abs().clamp_min(1e-8)
            gan_term = gan_scale * g_wgan
        else:
            gan_term = self.gan_weight * g_wgan
        g_total = gan_term + mse_term + nmf_term
        g_total.backward()
        self.optimizer_g.step()
        return {
            "D": loss_d_val, "G": float(g_total.item()),
            "mse": float(mse.item()), "nmf": float(nmf.item()),
            "wgan": float(g_wgan.item()), "gan_scale": float(gan_scale.item()),
        }

    @torch.no_grad()
    def block_metrics(self, loader):
        self.Gp.eval(); self.Gr.eval(); self.Gm.eval()
        reals = {"protein": [], "rna": [], "methyl": []}
        fakes = {"protein": [], "rna": [], "methyl": []}
        for batch in loader:
            rna = batch["x_rna"].to(self.device)
            prot = batch["x_prot"].to(self.device)
            methy = batch["x_methy"].to(self.device)
            yp, yr, ym = self._forward(rna, prot, methy)
            reals["protein"].append(prot.cpu()); fakes["protein"].append(yp.cpu())
            reals["rna"].append(rna.cpu()); fakes["rna"].append(yr.cpu())
            reals["methyl"].append(methy.cpu()); fakes["methyl"].append(ym.cpu())
        out = {}
        rmses = []
        mean_l2s, std_maes = [], []
        for k in ("protein", "rna", "methyl"):
            y = torch.cat(reals[k], dim=0)
            yhat = torch.cat(fakes[k], dim=0)
            rmse = float(torch.sqrt(((y - yhat) ** 2).mean()).item())
            mean_l2 = float(torch.sqrt(((y.mean(0) - yhat.mean(0)) ** 2).mean()).item())
            std_mae = float((y.std(0) - yhat.std(0)).abs().mean().item())
            out[k] = rmse
            out[f"{k}_mean_l2"] = mean_l2
            out[f"{k}_std_mae"] = std_mae
            rmses.append(rmse)
            mean_l2s.append(mean_l2)
            std_maes.append(std_mae)
        out["avg"] = float(np.mean(rmses))
        out["mean_l2"] = float(np.mean(mean_l2s))
        out["std_mae"] = float(np.mean(std_maes))
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/home/dyan/nmf/mochi_code/processed_data/gate_brca")
    ap.add_argument("--save_dir", default="/home/dyan/nmf/mochi_code/results/experiments/gate_tri_v5")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--warmup", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lr_d", type=float, default=2e-4)
    ap.add_argument("--mse_weight", type=float, default=0.1)
    ap.add_argument("--gan_weight", type=float, default=1.0)
    ap.add_argument("--gan_to_mse", type=float, default=0.3)
    ap.add_argument("--nmf_weight", type=float, default=0.1)
    ap.add_argument("--nmf_k", type=int, default=20)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--use_attn", action="store_true")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_ds = TripleSplitDataset(args.data_dir, "train")
    val_ds = TripleSplitDataset(args.data_dir, "val", stats=train_ds.stats)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    dim_r, dim_p, dim_m = train_ds.rna_f.shape[1], train_ds.prot_f.shape[1], train_ds.methy_f.shape[1]
    print(f"n_train={len(train_ds)} n_val={len(val_ds)} dims R/P/M={dim_r}/{dim_p}/{dim_m}")
    print("z-score mean-imputation baseline ≈ 1.0  (이기려면 < 1)")

    kw = dict(
        use_attn=args.use_attn, lightweight=True, n_heads=4, d_head=64,
        apply_output_range=False, n_mod_tokens=4, use_linear_skip=True, skip_rank=64,
    )
    Gp = Generator(input_size=dim_r + dim_m, output_size=dim_p, target_type="protein",
                   src_size=dim_r + dim_m, src_dims=(dim_r, dim_m), **kw)
    Gr = Generator(input_size=dim_p + dim_m, output_size=dim_r, target_type="rna",
                   src_size=dim_p + dim_m, src_dims=(dim_p, dim_m), **kw)
    Gm = Generator(input_size=dim_r + dim_p, output_size=dim_m, target_type="methyl",
                   src_size=dim_r + dim_p, src_dims=(dim_r, dim_p), **kw)
    Dp = ConditionalCritic(dim_p, dim_r + dim_m)
    Dr = ConditionalCritic(dim_r, dim_p + dim_m)
    Dm = ConditionalCritic(dim_m, dim_r + dim_p)
    print(f"mod-attn={Gp.use_mod_attn} skip={Gp.use_linear_skip}  params Gp/Gr/Gm = "
          f"{count_parameters(Gp):,} / {count_parameters(Gr):,} / {count_parameters(Gm):,}")
    print(f"cond-critic params Dp/Dr/Dm = "
          f"{count_parameters(Dp):,} / {count_parameters(Dr):,} / {count_parameters(Dm):,}")

    print("fitting NMF dictionaries on train (z-score, shifted nonnegative)...")
    nmf_basis = {
        "rna": build_nmf_basis(train_ds.rna_f, k=args.nmf_k, device=device),
        "protein": build_nmf_basis(train_ds.prot_f, k=args.nmf_k, device=device),
        "methyl": build_nmf_basis(train_ds.methy_f, k=args.nmf_k, device=device),
    }
    print(f"loss: |GAN|={args.gan_to_mse}*|MSE| + {args.mse_weight}*MSE + {args.nmf_weight}*NMF  "
          f"k={args.nmf_k}  warmup={args.warmup}  lr_g={args.lr} lr_d={args.lr_d}  spectral=True")

    trainer = WGAN_GP_Trainer_Tri(
        Gp, Gr, Gm, Dp, Dr, Dm, device,
        lr_g=args.lr, lr_d=args.lr_d,
        mse_weight=args.mse_weight, gan_weight=args.gan_weight,
        nmf_weight=args.nmf_weight, nmf_basis=nmf_basis,
        gan_to_mse=args.gan_to_mse,
    )
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    best, patience = float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        use_gan = epoch > args.warmup
        phase = "cWGAN+MSE+NMF" if use_gan else "MSE+NMF"
        Gp.train(); Gr.train(); Gm.train(); Dp.train(); Dr.train(); Dm.train()
        logs = [trainer.train_step(b, use_gan=use_gan) for b in train_loader]
        g = float(np.mean([x["G"] for x in logs]))
        mse = float(np.mean([x["mse"] for x in logs]))
        nmf = float(np.mean([x["nmf"] for x in logs]))
        wgan = float(np.mean([x["wgan"] for x in logs]))
        gscale = float(np.mean([x["gan_scale"] for x in logs]))
        met = trainer.block_metrics(val_loader)
        beat = " <mean" if met["avg"] < 1.0 else ""
        mark = ""
        if met["avg"] < best:
            best = met["avg"]
            patience = 0
            mark = " *"
            torch.save(
                {"Gp": Gp.state_dict(), "Gr": Gr.state_dict(), "Gm": Gm.state_dict(),
                 "epoch": epoch, "val": met, "stats_keys": list(train_ds.stats.keys())},
                save_dir / "tri_best.ckpt",
            )
        else:
            patience += 1
        torch.save(
            {"Gp": Gp.state_dict(), "Gr": Gr.state_dict(), "Gm": Gm.state_dict(),
             "epoch": epoch, "val": met},
            save_dir / "tri_last.ckpt",
        )
        print(f"epoch {epoch:3d} [{phase}] mse={mse:.4f} nmf={nmf:.4f} wgan={wgan:.3f} "
              f"gscale={gscale:.4f} G={g:.3f} "
              f"zRMSE avg={met['avg']:.4f} P={met['protein']:.4f} "
              f"R={met['rna']:.4f} M={met['methyl']:.4f} "
              f"dist μ={met['mean_l2']:.3f} σ={met['std_mae']:.3f}{beat}{mark}")
        if epoch > max(args.warmup, 10) and patience >= args.patience:
            print(f"early stop at epoch {epoch}")
            break
    print(f"best val zRMSE avg={best:.4f} (mean baseline ≈ 1.0)  saved {save_dir / 'tri_best.ckpt'}")


if __name__ == "__main__":
    main()
