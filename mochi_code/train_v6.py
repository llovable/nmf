#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v7 (train_v6 엔트리): Ŷ = Ridge + G_res(gated encoders).

- 단백질 핵은 RNA-only Ridge. 메틸 선형항은 시그모이드 게이트로 0까지 가능.
- RNA/메틸 핵은 2→1 concat Ridge (이쪽은 두 소스가 이득).
- 잔차 헤드는 0으로 시작 → 학습 전 단백질 ≈ Ridge(RNA).
- 손실은 OmiTrans식: MSE가 본문, GAN은 |GAN|≈0.05|MSE|, NMF 0.1.
- 조건부 critic D(Y | X_obs). 단백질 조건의 메틸은 g_m * meth.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from sklearn.linear_model import RidgeCV
from torch.utils.data import DataLoader

from models import ConditionalCritic, count_parameters
from models_v6 import FrozenRidge, MOCHIGated
from train_gate import TripleSplitDataset, build_nmf_basis, nmf_recon_loss

RIDGE_ALPHAS = np.logspace(-2, 3, 8)


def fit_ridge_torch(X, Y, device):
    mdl = RidgeCV(alphas=RIDGE_ALPHAS, cv=3).fit(X, Y)
    return FrozenRidge(mdl.coef_, mdl.intercept_).to(device), float(mdl.alpha_)


class TrainerV6:
    def __init__(self, net, ridges, critics, device, nmf_basis,
                 lr_g=1e-4, lr_d=2e-4, mse_weight=1.0, nmf_weight=0.1,
                 gan_to_mse=0.05, lambda_gp=10.0, n_critic=5, meth_gate_l1=0.01):
        self.net = net.to(device)
        self.ridge_p_rna, self.ridge_p_meth, self.ridge_r, self.ridge_m = ridges
        self.Dp, self.Dr, self.Dm = [c.to(device) for c in critics]
        self.device = device
        self.nmf_basis = nmf_basis
        self.mse_weight, self.nmf_weight = mse_weight, nmf_weight
        self.gan_to_mse, self.lambda_gp, self.n_critic = gan_to_mse, lambda_gp, n_critic
        self.meth_gate_l1 = meth_gate_l1
        self.opt_g = optim.Adam(net.parameters(), lr=lr_g, betas=(0.5, 0.9))
        self.opt_d = optim.Adam(
            list(self.Dp.parameters()) + list(self.Dr.parameters()) + list(self.Dm.parameters()),
            lr=lr_d, betas=(0.5, 0.9))

    def _gp(self, D, real, fake, cond):
        alpha = torch.rand(real.size(0), 1, device=self.device).expand_as(real)
        inter = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
        out = D(inter, cond)
        grad = torch.autograd.grad(
            out, inter, grad_outputs=torch.ones_like(out),
            create_graph=True, retain_graph=True, only_inputs=True)[0]
        return ((grad.norm(2, dim=1) - 1) ** 2).mean()

    def _predict(self, rna, prot, methy):
        res_p, res_r, res_m, g_m = self.net.residuals(rna, prot, methy)
        lin_p = self.ridge_p_rna(rna) + g_m * self.ridge_p_meth(methy)
        c_r, c_m = torch.cat([prot, methy], 1), torch.cat([rna, prot], 1)
        lin_r, lin_m = self.ridge_r(c_r), self.ridge_m(c_m)
        c_p = torch.cat([rna, g_m * methy], 1)
        return lin_p + res_p, lin_r + res_r, lin_m + res_m, (c_p, c_r, c_m), g_m

    def train_step(self, batch):
        rna = batch["x_rna"].to(self.device)
        prot = batch["x_prot"].to(self.device)
        methy = batch["x_methy"].to(self.device)

        loss_d = 0.0
        for _ in range(self.n_critic):
            self.opt_d.zero_grad()
            with torch.no_grad():
                yp, yr, ym, (c_p, c_r, c_m), _ = self._predict(rna, prot, methy)
            ld = (
                -(self.Dp(prot, c_p).mean() - self.Dp(yp, c_p).mean())
                + self.lambda_gp * self._gp(self.Dp, prot, yp, c_p)
                + -(self.Dr(rna, c_r).mean() - self.Dr(yr, c_r).mean())
                + self.lambda_gp * self._gp(self.Dr, rna, yr, c_r)
                + -(self.Dm(methy, c_m).mean() - self.Dm(ym, c_m).mean())
                + self.lambda_gp * self._gp(self.Dm, methy, ym, c_m)
            )
            ld.backward()
            self.opt_d.step()
            loss_d = float(ld.item())

        self.opt_g.zero_grad()
        yp, yr, ym, (c_p, c_r, c_m), g_m = self._predict(rna, prot, methy)
        mse = ((yp - prot).pow(2).mean() + (yr - rna).pow(2).mean() + (ym - methy).pow(2).mean())
        g_wgan = -(self.Dp(yp, c_p).mean() + self.Dr(yr, c_r).mean() + self.Dm(ym, c_m).mean())
        nmf = (
            nmf_recon_loss(yp, self.nmf_basis["protein"])
            + nmf_recon_loss(yr, self.nmf_basis["rna"])
            + nmf_recon_loss(ym, self.nmf_basis["methyl"])
        )
        mse_term = self.mse_weight * mse
        scale = (self.gan_to_mse * mse_term.detach().abs()) / g_wgan.detach().abs().clamp_min(1e-8)
        gate_pen = self.meth_gate_l1 * g_m.mean()
        loss_g = mse_term + self.nmf_weight * nmf + scale * g_wgan + gate_pen
        loss_g.backward()
        self.opt_g.step()
        return {"D": loss_d, "G": float(loss_g.item()), "mse": float(mse.item()),
                "nmf": float(nmf.item()), "wgan": float(g_wgan.item()),
                "gscale": float(scale.item()), "g_meth": float(g_m.mean().item())}

    @torch.no_grad()
    def metrics(self, loader):
        self.net.eval()
        ys = {"protein": [], "rna": [], "methyl": []}
        hats = {"protein": [], "rna": [], "methyl": []}
        alphas = {"protein": [], "rna": [], "methyl": []}
        gates = []
        for batch in loader:
            rna = batch["x_rna"].to(self.device)
            prot = batch["x_prot"].to(self.device)
            methy = batch["x_methy"].to(self.device)
            yp, yr, ym, _, g_m = self._predict(rna, prot, methy)
            ys["protein"].append(prot.cpu()); hats["protein"].append(yp.cpu())
            ys["rna"].append(rna.cpu()); hats["rna"].append(yr.cpu())
            ys["methyl"].append(methy.cpu()); hats["methyl"].append(ym.cpu())
            gates.append(g_m.cpu())
            for k, a in self.net.last_alpha.items():
                alphas[k].append(a.cpu())
        out = {}
        for k in ys:
            y, yhat = torch.cat(ys[k]), torch.cat(hats[k])
            out[k] = float(torch.sqrt(((y - yhat) ** 2).mean()).item())
            a = torch.cat(alphas[k], 0).mean(0)
            out[f"{k}_a0"] = float(a[0].item())
            out[f"{k}_a1"] = float(a[1].item())
        out["avg"] = float(np.mean([out["protein"], out["rna"], out["methyl"]]))
        out["g_meth"] = float(torch.cat(gates, 0).mean().item())
        self.net.train()
        return out


def pretrain_encoders(net, loader, device, epochs=5, lr=1e-3):
    """MIMIR 1단계: 각 오믹스를 자기 재구성으로만 예열."""
    d = net.enc_r.net[-1].out_features
    dec_r = torch.nn.Linear(d, net.head_r.fc2.out_features).to(device)
    dec_p = torch.nn.Linear(d, net.head_p.fc2.out_features).to(device)
    dec_m = torch.nn.Linear(d, net.head_m.fc2.out_features).to(device)
    opt = optim.Adam(
        list(net.enc_r.parameters()) + list(net.enc_p.parameters()) + list(net.enc_m.parameters())
        + list(dec_r.parameters()) + list(dec_p.parameters()) + list(dec_m.parameters()),
        lr=lr)
    net.train()
    for ep in range(1, epochs + 1):
        losses = []
        for batch in loader:
            rna = batch["x_rna"].to(device)
            prot = batch["x_prot"].to(device)
            methy = batch["x_methy"].to(device)
            opt.zero_grad()
            zr, zp, zm = net.encode(rna, prot, methy)
            loss = ((dec_r(zr) - rna).pow(2).mean()
                    + (dec_p(zp) - prot).pow(2).mean()
                    + (dec_m(zm) - methy).pow(2).mean())
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        print(f"pretrain encoder epoch {ep} recon={np.mean(losses):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/home/dyan/nmf/mochi_code/processed_data/gate_brca")
    ap.add_argument("--save_dir", default="/home/dyan/nmf/mochi_code/results/current/gate_v7")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--pretrain", type=int, default=5)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--gan_to_mse", type=float, default=0.05)
    ap.add_argument("--mse_weight", type=float, default=1.0)
    ap.add_argument("--nmf_weight", type=float, default=0.1)
    ap.add_argument("--meth_gate_l1", type=float, default=0.01)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_ds = TripleSplitDataset(args.data_dir, "train")
    val_ds = TripleSplitDataset(args.data_dir, "val", stats=train_ds.stats)
    test_ds = TripleSplitDataset(args.data_dir, "test", stats=train_ds.stats)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    dim_r, dim_p, dim_m = train_ds.rna_f.shape[1], train_ds.prot_f.shape[1], train_ds.methy_f.shape[1]
    print(f"n train/val/test={len(train_ds)}/{len(val_ds)}/{len(test_ds)}  R/P/M={dim_r}/{dim_p}/{dim_m}")

    print("fitting frozen Ridge (protein←RNA + gated meth; RNA/methyl 2→1) ...")
    ridge_p_rna, a_pr = fit_ridge_torch(train_ds.rna_f, train_ds.prot_f, device)
    ridge_p_meth, a_pm = fit_ridge_torch(train_ds.methy_f, train_ds.prot_f, device)
    ridge_r, a_r = fit_ridge_torch(
        np.concatenate([train_ds.prot_f, train_ds.methy_f], 1), train_ds.rna_f, device)
    ridge_m, a_m = fit_ridge_torch(
        np.concatenate([train_ds.rna_f, train_ds.prot_f], 1), train_ds.methy_f, device)
    print(f"Ridge alpha P←R/P←M/R/M = {a_pr:.4g}/{a_pm:.4g}/{a_r:.4g}/{a_m:.4g}")

    net = MOCHIGated(dim_r, dim_p, dim_m)
    Dp = ConditionalCritic(dim_p, dim_r + dim_m)
    Dr = ConditionalCritic(dim_r, dim_p + dim_m)
    Dm = ConditionalCritic(dim_m, dim_r + dim_p)
    print(f"params net={count_parameters(net):,}  critics={count_parameters(Dp)+count_parameters(Dr)+count_parameters(Dm):,}")

    nmf_basis = {
        "rna": build_nmf_basis(train_ds.rna_f, k=20, device=device),
        "protein": build_nmf_basis(train_ds.prot_f, k=20, device=device),
        "methyl": build_nmf_basis(train_ds.methy_f, k=20, device=device),
    }
    trainer = TrainerV6(
        net, (ridge_p_rna, ridge_p_meth, ridge_r, ridge_m), (Dp, Dr, Dm), device, nmf_basis,
        mse_weight=args.mse_weight, nmf_weight=args.nmf_weight, gan_to_mse=args.gan_to_mse,
        meth_gate_l1=args.meth_gate_l1,
    )

    # residual=0, g_m≈0 → 단백질은 Ridge(RNA), 나머지 2→1
    met0 = trainer.metrics(val_loader)
    print(f"Ridge frozen val zRMSE avg={met0['avg']:.4f} "
          f"P={met0['protein']:.4f} R={met0['rna']:.4f} M={met0['methyl']:.4f} "
          f"g_meth={met0['g_meth']:.3f}")

    if args.pretrain > 0:
        pretrain_encoders(net, train_loader, device, epochs=args.pretrain)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best, patience = float("inf"), 0
    print(f"loss: MSE + {args.nmf_weight}*NMF + |GAN|={args.gan_to_mse}*|MSE|")

    for epoch in range(1, args.epochs + 1):
        net.train(); Dp.train(); Dr.train(); Dm.train()
        logs = [trainer.train_step(b) for b in train_loader]
        mse = float(np.mean([x["mse"] for x in logs]))
        gscale = float(np.mean([x["gscale"] for x in logs]))
        gmeth_tr = float(np.mean([x["g_meth"] for x in logs]))
        met = trainer.metrics(val_loader)
        beat = " <ridge" if met["avg"] + 1e-6 < met0["avg"] else ""
        mark = ""
        if met["avg"] < best:
            best, patience, mark = met["avg"], 0, " *"
            torch.save({"net": net.state_dict(), "epoch": epoch, "val": met,
                        "ridge_p_rna": {"W": ridge_p_rna.W, "b": ridge_p_rna.b},
                        "ridge_p_meth": {"W": ridge_p_meth.W, "b": ridge_p_meth.b},
                        "ridge_r": {"W": ridge_r.W, "b": ridge_r.b},
                        "ridge_m": {"W": ridge_m.W, "b": ridge_m.b}},
                       save_dir / "tri_best.ckpt")
        else:
            patience += 1
        print(
            f"epoch {epoch:3d} mse={mse:.4f} gscale={gscale:.4f} "
            f"zRMSE avg={met['avg']:.4f} P={met['protein']:.4f} "
            f"R={met['rna']:.4f} M={met['methyl']:.4f} "
            f"g_meth={met['g_meth']:.3f} (tr {gmeth_tr:.3f}) "
            f"gate P[R,M]={met['protein_a0']:.2f}/{met['protein_a1']:.2f} "
            f"R[P,M]={met['rna_a0']:.2f}/{met['rna_a1']:.2f} "
            f"M[R,P]={met['methyl_a0']:.2f}/{met['methyl_a1']:.2f}"
            f"{beat}{mark}"
        )
        if epoch > 10 and patience >= args.patience:
            print(f"early stop at epoch {epoch}")
            break

    ckpt = torch.load(save_dir / "tri_best.ckpt", map_location=device, weights_only=False)
    net.load_state_dict(ckpt["net"])
    val_m = trainer.metrics(val_loader)
    test_m = trainer.metrics(test_loader)
    print(f"best val avg={val_m['avg']:.4f} P={val_m['protein']:.4f} R={val_m['rna']:.4f} "
          f"M={val_m['methyl']:.4f} g_meth={val_m['g_meth']:.3f}")
    print(f"test     avg={test_m['avg']:.4f} P={test_m['protein']:.4f} R={test_m['rna']:.4f} "
          f"M={test_m['methyl']:.4f} g_meth={test_m['g_meth']:.3f}")
    print(f"Ridge frozen val was {met0['avg']:.4f} (P={met0['protein']:.4f})  saved {save_dir / 'tri_best.ckpt'}")


if __name__ == "__main__":
    main()
