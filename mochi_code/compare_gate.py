#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BRCA 게이트 비교: 같은 split, 같은 z-score, block-missing.

표 A/C용
- mean / median
- Ridge 1→1, Ridge 2→1
- TOBMI (kNN) 1→1, 2→1
- OmicsNMF-reimpl 1→1 (MLP + uncond critic + NMF + 균형 GAN)
- MOCHI 1→1 (어텐션 + 조건부 critic + skip, 소스 하나)
- MOCHI 2→1 (v5 체크포인트)

1→1 주 소스는 2→1 concat의 첫 오믹스와 같게 둔다.
  protein ← RNA,  RNA ← protein,  methyl ← RNA
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.linear_model import RidgeCV
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader

from models import Generator, Critic, ConditionalCritic, count_parameters
from train_gate import (
    TripleSplitDataset, WGAN_GP_Trainer_Tri, build_nmf_basis, nmf_recon_loss,
)


PAIRS_1TO1 = (
    ("protein", "rna"),
    ("rna", "protein"),
    ("methyl", "rna"),
)
PAIRS_2TO1 = (
    ("protein", ("rna", "methyl")),
    ("rna", ("protein", "methyl")),
    ("methyl", ("rna", "protein")),
)
RIDGE_ALPHAS = np.logspace(-2, 3, 8)


def _rmse(y, yhat):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))


def _get(ds, name):
    return {"rna": ds.rna_f, "protein": ds.prot_f, "methyl": ds.methy_f}[name]


def _hstack(ds, names):
    return np.concatenate([_get(ds, n) for n in names], axis=1)


def eval_constant(train, ev, how="mean"):
    out = {}
    for tgt in ("protein", "rna", "methyl"):
        ytr, ye = _get(train, tgt), _get(ev, tgt)
        pred = (np.median(ytr, axis=0) if how == "median" else np.zeros(ytr.shape[1], np.float32))
        out[tgt] = _rmse(ye, np.broadcast_to(pred, ye.shape))
    out["avg"] = float(np.mean([out["protein"], out["rna"], out["methyl"]]))
    return out


def fit_ridge(train, src_names, tgt):
    Xtr, ytr = _hstack(train, src_names), _get(train, tgt)
    return RidgeCV(alphas=RIDGE_ALPHAS, cv=3).fit(Xtr, ytr)


def pred_rmse(mdl, ev, src_names, tgt):
    return _rmse(_get(ev, tgt), mdl.predict(_hstack(ev, src_names)))


def tobmi_rmse(train, ev, src_names, tgt, k=10):
    Xtr, Xev = _hstack(train, src_names), _hstack(ev, src_names)
    ytr, ye = _get(train, tgt), _get(ev, tgt)
    k = min(k, len(Xtr))
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(Xtr)
    idx = nn.kneighbors(Xev, return_distance=False)
    return _rmse(ye, ytr[idx].mean(axis=1))


@torch.no_grad()
def eval_mochi_tri(trainer, loader):
    return trainer.block_metrics(loader)


def make_mochi_2to1(dim_r, dim_p, dim_m, attn=True):
    kw = dict(
        use_attn=attn, lightweight=True, n_heads=4, d_head=64,
        apply_output_range=False, n_mod_tokens=4, use_linear_skip=True, skip_rank=64,
    )
    Gp = Generator(input_size=dim_r + dim_m, output_size=dim_p, target_type="protein",
                   src_size=dim_r + dim_m, src_dims=(dim_r, dim_m), **kw)
    Gr = Generator(input_size=dim_p + dim_m, output_size=dim_r, target_type="rna",
                   src_size=dim_p + dim_m, src_dims=(dim_p, dim_m), **kw)
    Gm = Generator(input_size=dim_r + dim_p, output_size=dim_m, target_type="methyl",
                   src_size=dim_r + dim_p, src_dims=(dim_r, dim_p), **kw)
    return Gp, Gr, Gm


def make_mochi_1to1(dim_r, dim_p, dim_m):
    kw = dict(
        use_attn=True, lightweight=True, n_heads=4, d_head=64,
        apply_output_range=False, n_mod_tokens=4, use_linear_skip=True, skip_rank=64,
    )
    Gp = Generator(input_size=dim_r, output_size=dim_p, target_type="protein",
                   src_size=dim_r, src_dims=(dim_r,), **kw)
    Gr = Generator(input_size=dim_p, output_size=dim_r, target_type="rna",
                   src_size=dim_p, src_dims=(dim_p,), **kw)
    Gm = Generator(input_size=dim_r, output_size=dim_m, target_type="methyl",
                   src_size=dim_r, src_dims=(dim_r,), **kw)
    Dp = ConditionalCritic(dim_p, dim_r)
    Dr = ConditionalCritic(dim_r, dim_p)
    Dm = ConditionalCritic(dim_m, dim_r)
    return Gp, Gr, Gm, Dp, Dr, Dm


class Trainer1to1(WGAN_GP_Trainer_Tri):
    """v5와 같은 손실, 소스는 각 타깃의 1→1 주 소스만."""

    def _conds(self, rna, prot, methy):
        return rna, prot, rna

    def _forward(self, rna, prot, methy):
        return self.Gp(rna, src=None), self.Gr(prot, src=None), self.Gm(rna, src=None)


class PairTrainer:
    """OmicsNMF-reimpl: 소스 하나, 무조건부 critic."""

    def __init__(self, G, D, device, nmf_basis, lr_g=1e-4, lr_d=2e-4,
                 mse_weight=0.1, nmf_weight=0.1, gan_to_mse=0.3,
                 lambda_gp=10.0, n_critic=5):
        self.G, self.D = G.to(device), D.to(device)
        self.device = device
        self.nmf_basis = nmf_basis
        self.mse_weight, self.nmf_weight = mse_weight, nmf_weight
        self.gan_to_mse, self.lambda_gp, self.n_critic = gan_to_mse, lambda_gp, n_critic
        self.opt_g = optim.Adam(G.parameters(), lr=lr_g, betas=(0.5, 0.9))
        self.opt_d = optim.Adam(D.parameters(), lr=lr_d, betas=(0.5, 0.9))

    def _gp(self, real, fake):
        alpha = torch.rand(real.size(0), 1, device=self.device).expand_as(real)
        inter = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
        out = self.D(inter)
        grad = torch.autograd.grad(
            out, inter, grad_outputs=torch.ones_like(out),
            create_graph=True, retain_graph=True, only_inputs=True)[0]
        return ((grad.norm(2, dim=1) - 1) ** 2).mean()

    def train_step(self, src, tgt):
        src, tgt = src.to(self.device), tgt.to(self.device)
        for _ in range(self.n_critic):
            self.opt_d.zero_grad()
            fake = self.G(src, src=None).detach()
            loss_d = -(self.D(tgt).mean() - self.D(fake).mean()) + self.lambda_gp * self._gp(tgt, fake)
            loss_d.backward()
            self.opt_d.step()
        self.opt_g.zero_grad()
        fake = self.G(src, src=None)
        mse = ((fake - tgt) ** 2).mean()
        g_wgan = -self.D(fake).mean()
        nmf = nmf_recon_loss(fake, self.nmf_basis)
        mse_term = self.mse_weight * mse
        target = self.gan_to_mse * mse_term.detach().abs()
        scale = target / g_wgan.detach().abs().clamp_min(1e-8)
        loss_g = scale * g_wgan + mse_term + self.nmf_weight * nmf
        loss_g.backward()
        self.opt_g.step()
        return float(mse.item())

    @torch.no_grad()
    def rmse(self, src, tgt):
        self.G.eval()
        pred = self.G(torch.from_numpy(src).to(self.device), src=None).cpu().numpy()
        self.G.train()
        return _rmse(tgt, pred)


def run_baselines(train, val, test):
    rows = []

    def add(method, setting, split, met):
        rows.append({
            "method": method, "setting": setting, "split": split,
            "protein": met["protein"], "rna": met["rna"], "methyl": met["methyl"],
            "avg": met["avg"],
        })

    print("fitting Ridge on train...")
    ridge_1 = {tgt: fit_ridge(train, (src,), tgt) for tgt, src in PAIRS_1TO1}
    ridge_2 = {tgt: fit_ridge(train, srcs, tgt) for tgt, srcs in PAIRS_2TO1}
    ridge_alt = {}
    for tgt, srcs in PAIRS_2TO1:
        a, b = srcs
        ridge_alt[(tgt, a)] = fit_ridge(train, (a,), tgt)
        ridge_alt[(tgt, b)] = fit_ridge(train, (b,), tgt)

    for split, ev in (("val", val), ("test", test)):
        add("mean", "none", split, eval_constant(train, ev, "mean"))
        add("median", "none", split, eval_constant(train, ev, "median"))

        r1 = {tgt: pred_rmse(ridge_1[tgt], ev, (src,), tgt) for tgt, src in PAIRS_1TO1}
        t1 = {tgt: tobmi_rmse(train, ev, (src,), tgt) for tgt, src in PAIRS_1TO1}
        r1["avg"] = float(np.mean([r1["protein"], r1["rna"], r1["methyl"]]))
        t1["avg"] = float(np.mean([t1["protein"], t1["rna"], t1["methyl"]]))
        add("Ridge", "1to1", split, r1)
        add("TOBMI", "1to1", split, t1)

        r2 = {tgt: pred_rmse(ridge_2[tgt], ev, srcs, tgt) for tgt, srcs in PAIRS_2TO1}
        t2 = {tgt: tobmi_rmse(train, ev, srcs, tgt) for tgt, srcs in PAIRS_2TO1}
        r2["avg"] = float(np.mean([r2["protein"], r2["rna"], r2["methyl"]]))
        t2["avg"] = float(np.mean([t2["protein"], t2["rna"], t2["methyl"]]))
        add("Ridge", "2to1", split, r2)
        add("TOBMI", "2to1", split, t2)

        extra = {f"{tgt}<-{s}": pred_rmse(ridge_alt[(tgt, s)], ev, (s,), tgt)
                 for tgt, srcs in PAIRS_2TO1 for s in srcs}
        print(f"[{split}] Ridge per-source: " + " ".join(f"{k}={v:.4f}" for k, v in extra.items()))

    return rows


def load_v7(ckpt_path, dim_r, dim_p, dim_m, device):
    from models_v6 import FrozenRidge, MOCHIGated
    from train_v6 import TrainerV6

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    net = MOCHIGated(dim_r, dim_p, dim_m)

    def _ridge(key, n_in, n_out):
        r = FrozenRidge(np.zeros((n_out, n_in), np.float32), np.zeros(n_out, np.float32)).to(device)
        r.W.copy_(ckpt[key]["W"].to(device))
        r.b.copy_(ckpt[key]["b"].to(device))
        return r

    ridges = (
        _ridge("ridge_p_rna", dim_r, dim_p),
        _ridge("ridge_p_meth", dim_m, dim_p),
        _ridge("ridge_r", dim_p + dim_m, dim_r),
        _ridge("ridge_m", dim_r + dim_p, dim_m),
    )
    Dp = ConditionalCritic(dim_p, dim_r + dim_m)
    Dr = ConditionalCritic(dim_r, dim_p + dim_m)
    Dm = ConditionalCritic(dim_m, dim_r + dim_p)
    trainer = TrainerV6(net, ridges, (Dp, Dr, Dm), device, nmf_basis={})
    net.load_state_dict(ckpt["net"])
    return trainer


def eval_v7(trainer, loader):
    return trainer.metrics(loader)


def merge_official_tsv(rows, official_dir):
    path = Path(official_dir) / "metrics.tsv"
    if not path.exists():
        return rows
    extra = pd.read_csv(path, sep="\t")
    print(f"merged official metrics from {path}")
    return rows + extra.to_dict("records")


def load_v5(ckpt_path, dim_r, dim_p, dim_m, device):
    Gp, Gr, Gm = make_mochi_2to1(dim_r, dim_p, dim_m)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    Gp.load_state_dict(ckpt["Gp"])
    Gr.load_state_dict(ckpt["Gr"])
    Gm.load_state_dict(ckpt["Gm"])
    dummy_d = ConditionalCritic(dim_p, dim_r + dim_m)
    trainer = WGAN_GP_Trainer_Tri(Gp, Gr, Gm, dummy_d, dummy_d, dummy_d, device)
    return trainer


def train_mochi_1to1(train_ds, val_ds, device, save_path, epochs=120, patience=25):
    dim_r, dim_p, dim_m = train_ds.rna_f.shape[1], train_ds.prot_f.shape[1], train_ds.methy_f.shape[1]
    Gp, Gr, Gm, Dp, Dr, Dm = make_mochi_1to1(dim_r, dim_p, dim_m)
    print(f"MOCHI 1→1 params Gp/Gr/Gm = {count_parameters(Gp):,}/{count_parameters(Gr):,}/{count_parameters(Gm):,}")
    nmf_basis = {
        "rna": build_nmf_basis(train_ds.rna_f, k=20, device=device),
        "protein": build_nmf_basis(train_ds.prot_f, k=20, device=device),
        "methyl": build_nmf_basis(train_ds.methy_f, k=20, device=device),
    }
    trainer = Trainer1to1(
        Gp, Gr, Gm, Dp, Dr, Dm, device,
        mse_weight=0.1, nmf_weight=0.1, gan_to_mse=0.3, nmf_basis=nmf_basis,
    )
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    best, pat = float("inf"), 0
    for epoch in range(1, epochs + 1):
        Gp.train(); Gr.train(); Gm.train(); Dp.train(); Dr.train(); Dm.train()
        logs = [trainer.train_step(b, use_gan=True) for b in train_loader]
        mse = float(np.mean([x["mse"] for x in logs]))
        met = trainer.block_metrics(val_loader)
        mark = ""
        if met["avg"] < best:
            best, pat, mark = met["avg"], 0, " *"
            torch.save({"Gp": Gp.state_dict(), "Gr": Gr.state_dict(), "Gm": Gm.state_dict(),
                        "epoch": epoch, "val": met}, save_path)
        else:
            pat += 1
        print(f"MOCHI1to1 epoch {epoch:3d} mse={mse:.4f} "
              f"avg={met['avg']:.4f} P={met['protein']:.4f} R={met['rna']:.4f} "
              f"M={met['methyl']:.4f}{mark}")
        if epoch > 10 and pat >= patience:
            print(f"MOCHI 1→1 early stop at {epoch}")
            break
    ckpt = torch.load(save_path, map_location=device, weights_only=False)
    Gp.load_state_dict(ckpt["Gp"]); Gr.load_state_dict(ckpt["Gr"]); Gm.load_state_dict(ckpt["Gm"])
    return trainer


def train_omicsnmf_pairs(train_ds, val_ds, device, save_dir, epochs=100, patience=20):
    save_dir = Path(save_dir)
    dim = {"rna": train_ds.rna_f.shape[1], "protein": train_ds.prot_f.shape[1],
           "methyl": train_ds.methy_f.shape[1]}
    trainers = {}
    for tgt, src in PAIRS_1TO1:
        ckpt_path = save_dir / f"omicsnmf_{src}_to_{tgt}.ckpt"
        G = Generator(
            input_size=dim[src], output_size=dim[tgt], target_type=tgt,
            lightweight=True, use_attn=False, use_linear_skip=False,
            apply_output_range=False,
        )
        D = Critic(dim[tgt], use_spectral_norm=True)
        basis = build_nmf_basis(_get(train_ds, tgt), k=20, device=device)
        tr = PairTrainer(G, D, device, basis)
        print(f"OmicsNMF {src}→{tgt} params G={count_parameters(G):,}")
        best, pat = float("inf"), 0
        Xtr, Ytr = _get(train_ds, src), _get(train_ds, tgt)
        Xva, Yva = _get(val_ds, src), _get(val_ds, tgt)
        n = len(Xtr)
        for epoch in range(1, epochs + 1):
            G.train(); D.train()
            perm = np.random.permutation(n)
            mses = []
            for i in range(0, n, 32):
                sl = perm[i:i + 32]
                mses.append(tr.train_step(
                    torch.from_numpy(Xtr[sl]), torch.from_numpy(Ytr[sl])))
            rmse = tr.rmse(Xva, Yva)
            mark = ""
            if rmse < best:
                best, pat, mark = rmse, 0, " *"
                torch.save({"G": G.state_dict(), "epoch": epoch, "val_rmse": rmse}, ckpt_path)
            else:
                pat += 1
            print(f"OmicsNMF {src}→{tgt} epoch {epoch:3d} mse={np.mean(mses):.4f} "
                  f"val={rmse:.4f}{mark}")
            if epoch > 10 and pat >= patience:
                print(f"OmicsNMF {src}→{tgt} early stop at {epoch}")
                break
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        G.load_state_dict(ckpt["G"])
        trainers[(tgt, src)] = tr
    return trainers


def eval_omicsnmf(trainers, ev):
    out = {}
    for tgt, src in PAIRS_1TO1:
        out[tgt] = trainers[(tgt, src)].rmse(_get(ev, src), _get(ev, tgt))
    out["avg"] = float(np.mean([out["protein"], out["rna"], out["methyl"]]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/home/dyan/nmf/mochi_code/processed_data/gate_brca")
    ap.add_argument("--v5_ckpt", default="/home/dyan/nmf/mochi_code/results/current/gate_tri_v5/tri_best.ckpt")
    ap.add_argument("--v7_ckpt", default="/home/dyan/nmf/mochi_code/results/current/gate_v7/tri_best.ckpt")
    ap.add_argument("--official_dir", default="/home/dyan/nmf/mochi_code/results/current/official")
    ap.add_argument("--out_dir", default="/home/dyan/nmf/mochi_code/results/current/gate_compare")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--skip_neural_train", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_ds = TripleSplitDataset(args.data_dir, "train")
    val_ds = TripleSplitDataset(args.data_dir, "val", stats=train_ds.stats)
    test_ds = TripleSplitDataset(args.data_dir, "test", stats=train_ds.stats)
    dim_r, dim_p, dim_m = train_ds.rna_f.shape[1], train_ds.prot_f.shape[1], train_ds.methy_f.shape[1]
    print(f"n train/val/test={len(train_ds)}/{len(val_ds)}/{len(test_ds)}  dims R/P/M={dim_r}/{dim_p}/{dim_m}")

    print("=== baselines (mean / median / Ridge / TOBMI) ===")
    rows = run_baselines(train_ds, val_ds, test_ds)

    print("=== MOCHI 2→1 v5 ===")
    tri = load_v5(args.v5_ckpt, dim_r, dim_p, dim_m, device)
    for split, ds in (("val", val_ds), ("test", test_ds)):
        met = eval_mochi_tri(tri, DataLoader(ds, batch_size=32, shuffle=False))
        rows.append({"method": "MOCHI", "setting": "2to1", "split": split,
                     "protein": met["protein"], "rna": met["rna"], "methyl": met["methyl"],
                     "avg": met["avg"]})
        print(f"MOCHI 2→1 {split}: avg={met['avg']:.4f} P={met['protein']:.4f} "
              f"R={met['rna']:.4f} M={met['methyl']:.4f}")

    v7_path = Path(args.v7_ckpt)
    if v7_path.exists():
        print("=== MOCHI v7 (protein meth-drop) ===")
        tr7 = load_v7(str(v7_path), dim_r, dim_p, dim_m, device)
        for split, ds in (("val", val_ds), ("test", test_ds)):
            met = eval_v7(tr7, DataLoader(ds, batch_size=32, shuffle=False))
            rows.append({"method": "MOCHI-v7", "setting": "2to1-gated", "split": split,
                         "protein": met["protein"], "rna": met["rna"], "methyl": met["methyl"],
                         "avg": met["avg"]})
            print(f"MOCHI v7 {split}: avg={met['avg']:.4f} P={met['protein']:.4f} "
                  f"R={met['rna']:.4f} M={met['methyl']:.4f} g_meth={met['g_meth']:.3f}")
    else:
        print(f"no v7 ckpt at {v7_path}")

    rows = merge_official_tsv(rows, args.official_dir)

    mochi1_path = out_dir / "mochi_1to1.ckpt"
    if not args.skip_neural_train:
        print("=== train MOCHI 1→1 ===")
        tr1 = train_mochi_1to1(train_ds, val_ds, device, mochi1_path)
        for split, ds in (("val", val_ds), ("test", test_ds)):
            met = eval_mochi_tri(tr1, DataLoader(ds, batch_size=32, shuffle=False))
            rows.append({"method": "MOCHI", "setting": "1to1", "split": split,
                         "protein": met["protein"], "rna": met["rna"], "methyl": met["methyl"],
                         "avg": met["avg"]})
            print(f"MOCHI 1→1 {split}: avg={met['avg']:.4f} P={met['protein']:.4f} "
                  f"R={met['rna']:.4f} M={met['methyl']:.4f}")

        print("=== train OmicsNMF-reimpl 1→1 ===")
        onmf = train_omicsnmf_pairs(train_ds, val_ds, device, out_dir)
        for split, ds in (("val", val_ds), ("test", test_ds)):
            met = eval_omicsnmf(onmf, ds)
            rows.append({"method": "OmicsNMF-reimpl", "setting": "1to1", "split": split,
                         "protein": met["protein"], "rna": met["rna"], "methyl": met["methyl"],
                         "avg": met["avg"]})
            print(f"OmicsNMF {split}: avg={met['avg']:.4f} P={met['protein']:.4f} "
                  f"R={met['rna']:.4f} M={met['methyl']:.4f}")
    elif mochi1_path.exists():
        print("skip train; loading existing 1→1 ckpts if present")

    df = pd.DataFrame(rows).sort_values(["split", "setting", "method"])
    df.to_csv(out_dir / "metrics.tsv", sep="\t", index=False, float_format="%.4f")
    print("\n=== summary ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"saved {out_dir / 'metrics.tsv'}")


if __name__ == "__main__":
    main()
