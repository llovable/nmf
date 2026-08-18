#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
공유 z MOCHI 학습.

1) 모달리티별 마스크 AE
2) 교차-어텐션 융합 + leave-one-out MSE + 대비 손실
조기 종료는 val 블록 z-RMSE.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from models_shared import (
    HIDDEN, MODS, SharedMOCHI, contrastive_loss, drop_modalities,
    mask_cells, mse_valid, predict_shared,
)
from train_gate import TripleSplitDataset


def _ae_step(enc, dec, xb, obs, opt, device, p=0.3, alpha=0.5):
    enc.train(); dec.train()
    xb, obs = xb.to(device), obs.to(device)
    extra = (torch.rand_like(xb) < p) & obs
    xin = xb.clone()
    xin[extra] = 0.0
    h = enc(xin)
    pred = dec(h)
    overall = mse_valid(pred, xb, obs)
    masked = mse_valid(pred, xb, extra) if extra.any() else overall
    loss = alpha * masked + (1.0 - alpha) * overall
    opt.zero_grad()
    loss.backward()
    opt.step()
    return float(loss.item())


@torch.no_grad()
def _ae_val(enc, dec, loader, key_x, key_m, device, p=0.3):
    enc.eval(); dec.eval()
    tot, n = 0.0, 0
    for batch in loader:
        xb = batch[key_x].to(device)
        obs = (1.0 - batch[key_m].to(device)) > 0.5
        extra = (torch.rand_like(xb) < p) & obs
        xin = xb.clone()
        xin[extra] = 0.0
        pred = dec(enc(xin))
        tot += float(mse_valid(pred, xb, extra if extra.any() else obs).item())
        n += 1
    return tot / max(n, 1)


def pretrain_aes(train_loader, val_loader, dims, device, epochs=70, patience=12):
    keys = {"protein": ("x_prot", "m_prot"), "rna": ("x_rna", "m_rna"), "methyl": ("x_methy", "m_methy")}
    encs, decs = {}, {}
    for m in MODS:
        enc = nn.Linear(dims[m], HIDDEN[m]).to(device)
        dec = nn.Linear(HIDDEN[m], dims[m]).to(device)
        opt = Adam(list(enc.parameters()) + list(dec.parameters()), lr=1e-3, weight_decay=1e-5)
        kx, km = keys[m]
        best, pat, state = float("inf"), 0, None
        for ep in range(1, epochs + 1):
            losses = []
            for batch in train_loader:
                xb = batch[kx]
                obs = (1.0 - batch[km]) > 0.5
                losses.append(_ae_step(enc, dec, xb, obs, opt, device))
            val = _ae_val(enc, dec, val_loader, kx, km, device)
            mark = ""
            if val < best:
                best, pat, mark = val, 0, " *"
                state = {
                    "enc": {k: v.detach().cpu().clone() for k, v in enc.state_dict().items()},
                    "dec": {k: v.detach().cpu().clone() for k, v in dec.state_dict().items()},
                }
            else:
                pat += 1
            if ep == 1 or ep % 10 == 0 or mark:
                print(f"  AE {m} ep {ep:03d} train={np.mean(losses):.4f} val={val:.4f}{mark}")
            if ep > 8 and pat >= patience:
                print(f"  AE {m} early stop at {ep}")
                break
        enc.load_state_dict(state["enc"])
        dec.load_state_dict(state["dec"])
        encs[m], decs[m] = enc, dec
    return encs, decs


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


def train_epoch(model, loader, opt, device, mask_p=0.15, drop_p=0.4,
                lambda_loo=1.0, lambda_con=1.0, alpha=0.5):
    model.train()
    sums = {"total": 0.0, "recon": 0.0, "loo": 0.0, "con": 0.0}
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
        for tgt in MODS:
            others = [m for m in MODS if m != tgt]
            can = present[tgt] & torch.stack([present[m] for m in others], 0).any(0)
            if can.sum() < 1:
                continue
            hat = model.loo_reconstruct(present_xs, present, tgt)
            lloss = lloss + mse_valid(hat[can], xs[tgt][can], obs[tgt][can])
            n_l += 1
        lloss = lloss / max(n_l, 1) if n_l else rloss.new_zeros(())

        zs = model.encode_z(present_xs)
        closs = contrastive_loss(zs, present)

        loss = rloss + lambda_loo * lloss + lambda_con * closs
        opt.zero_grad()
        loss.backward()
        opt.step()
        sums["total"] += float(loss.item())
        sums["recon"] += float(rloss.item())
        sums["loo"] += float(lloss.item())
        sums["con"] += float(closs.item())
        n += 1
    return {k: v / max(n, 1) for k, v in sums.items()}


@torch.no_grad()
def block_zrmse(model, ds, device):
    tabs = {m: getattr(ds, {"protein": "prot_f", "rna": "rna_f", "methyl": "methy_f"}[m]) for m in MODS}
    rmses = []
    per = {}
    for tgt in MODS:
        hat = predict_shared(model, tabs, device, missing=tgt)
        y, yhat = tabs[tgt], hat[tgt]
        rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))
        per[tgt] = rmse
        rmses.append(rmse)
    per["avg"] = float(np.mean(rmses))
    return per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/home/dyan/nmf/mochi_code/processed_data/gate_brca")
    ap.add_argument("--save_dir", default="/home/dyan/nmf/mochi_code/results/current/gate_shared")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--epochs1", type=int, default=70)
    ap.add_argument("--epochs2", type=int, default=150)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--fuse", default="attn", choices=["attn", "mean"])
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    train_ds = TripleSplitDataset(args.data_dir, "train")
    val_ds = TripleSplitDataset(args.data_dir, "val", stats=train_ds.stats)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    dims = {"rna": train_ds.rna_f.shape[1], "protein": train_ds.prot_f.shape[1],
            "methyl": train_ds.methy_f.shape[1]}
    print(f"n={len(train_ds)}/{len(val_ds)} dims={dims}")

    print("=== phase1 modality AEs ===")
    encs, decs = pretrain_aes(train_loader, val_loader, dims, device, epochs=args.epochs1)

    model = SharedMOCHI(dims, fuse=args.fuse).to(device)
    for m in MODS:
        model.encoders[m].load_state_dict(encs[m].state_dict())
        model.decoders[m].load_state_dict(decs[m].state_dict())
    opt = Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== phase2 shared {args.fuse} fuse: own-z recon + LOO impute ===")
    best, pat = float("inf"), 0
    for ep in range(1, args.epochs2 + 1):
        tr = train_epoch(model, train_loader, opt, device)
        met = block_zrmse(model, val_ds, device)
        mark = ""
        if met["avg"] < best:
            best, pat, mark = met["avg"], 0, " *"
            torch.save({"model": model.state_dict(), "dims": dims, "shared_dim": 256,
                        "fuse": args.fuse, "epoch": ep, "val": met}, save_dir / "shared_best.ckpt")
        else:
            pat += 1
        print(f"ep {ep:03d} train tot={tr['total']:.4f} recon={tr['recon']:.4f} "
              f"loo={tr['loo']:.4f} con={tr['con']:.4f}  "
              f"val zRMSE avg={met['avg']:.4f} P={met['protein']:.4f} "
              f"R={met['rna']:.4f} M={met['methyl']:.4f}{mark}")
        if ep > 15 and pat >= args.patience:
            print(f"early stop at {ep}")
            break
    print(f"best val block zRMSE={best:.4f}  saved {save_dir / 'shared_best.ckpt'}")


if __name__ == "__main__":
    main()
