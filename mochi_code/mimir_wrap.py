#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
공식 MIMIR 소스(baselines/MIMIR/src)를 BRCA 631 스플릿에 붙인다.
baselines/ 는 수정하지 않는다.

은닉 크기: RNA 512, protein 128, methyl 256, shared 256
(논문: mRNA 512, miRNA 128, methylation 256).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

MIMIR_ROOT = Path("/home/dyan/nmf/baselines/MIMIR")
if str(MIMIR_ROOT) not in sys.path:
    sys.path.insert(0, str(MIMIR_ROOT))

from src.data_utils import (  # noqa: E402
    MultiOmicDataset, SingleModalityDataset, get_dataloader,
)
from src.impute1 import impute_missing_values  # noqa: E402
from src.mae_masked import (  # noqa: E402
    MultiModalWithSharedSpace, build_pretrain_ae_for_modality,
    eval_finetune_epoch, eval_modality_epoch_masked, extract_encoder_decoder_from_pretrained,
    finetune_epoch, load_modality_with_config, pretrain_modality_epoch,
    save_modality_with_config,
)
from src.translation import impute_missing_modalities_for_scenario  # noqa: E402

MODS = ("protein", "rna", "methyl")
HIDDEN = {"rna": [512], "protein": [128], "methyl": [256]}


def frames_from_dir(data_dir, split, stats=None):
    data_dir = Path(data_dir)
    raw = {}
    for m, fn in (("rna", "rna"), ("protein", "protein"), ("methyl", "methy")):
        raw[m] = pd.read_csv(data_dir / f"{fn}.{split}.tsv", sep="\t", index_col=0).T.astype(np.float32)
    idx = raw["rna"].index.intersection(raw["protein"].index).intersection(raw["methyl"].index)
    for m in MODS:
        raw[m] = raw[m].loc[idx]
    if stats is None:
        stats = {}
        for m in MODS:
            mu = raw[m].mean(axis=0)
            sd = raw[m].std(axis=0).replace(0.0, 1.0).fillna(1.0)
            stats[m] = (mu, sd)
    z = {}
    for m in MODS:
        mu, sd = stats[m]
        z[m] = (raw[m] - mu) / sd
    return z, stats, list(idx)


def _pretrain_one(name, df_tr, df_va, device, epochs=70, patience=12):
    input_dim = df_tr.shape[1]
    hidden = HIDDEN[name]
    cfg = dict(
        input_dim=input_dim, hidden_layers=hidden, activation_dropout=0.05,
        denoising=True, mask_p=0.3, tied=False, mask_value=0.0, loss_on_masked=True,
    )
    ae, hid = build_pretrain_ae_for_modality(
        input_dim, hidden, activation_dropout=0.05, denoising=True, mask_p=0.3,
        tied=False, mask_value=0.0, loss_on_masked=True,
    )
    ae = ae.to(device)
    opt = Adam(ae.parameters(), lr=1e-3, weight_decay=1e-5)
    tr_loader = DataLoader(SingleModalityDataset(df_tr), batch_size=64, shuffle=True)
    va_loader = DataLoader(SingleModalityDataset(df_va), batch_size=64, shuffle=False)
    best, pat, state = float("inf"), 0, None
    for ep in range(1, epochs + 1):
        pretrain_modality_epoch(ae, tr_loader, opt, device, alpha_mask=0.5)
        va = eval_modality_epoch_masked(ae, va_loader, device)
        val = va[1] if isinstance(va, tuple) else float(va)
        if isinstance(va, dict):
            val = float(va.get("masked", va.get("overall", list(va.values())[0])))
        elif isinstance(va, (tuple, list)):
            val = float(va[-1])
        mark = ""
        if val < best:
            best, pat, mark = val, 0, " *"
            state = {k: v.detach().cpu().clone() for k, v in ae.state_dict().items()}
        else:
            pat += 1
        if ep == 1 or ep % 10 == 0 or mark:
            print(f"  MIMIR AE {name} ep {ep:03d} val={val:.4f}{mark}")
        if ep > 8 and pat >= patience:
            print(f"  MIMIR AE {name} early stop at {ep}")
            break
    ae.load_state_dict(state)
    return ae, hid, cfg


def _eval_num(va):
    if isinstance(va, dict):
        return float(va.get("total_loss", va.get("masked", list(va.values())[0])))
    if isinstance(va, (tuple, list)):
        return float(va[-1])
    return float(va)


def train_mimir(data_dir, save_dir, device, phase1_epochs=70, phase2_epochs=120):
    save_dir = Path(save_dir)
    ae_dir = save_dir / "aes"
    ae_dir.mkdir(parents=True, exist_ok=True)

    tr, stats, _ = frames_from_dir(data_dir, "train")
    va, _, _ = frames_from_dir(data_dir, "val", stats=stats)
    te, _, _ = frames_from_dir(data_dir, "test", stats=stats)

    aes, hidden_dims, mask_values, configs = {}, {}, {}, {}
    for m in MODS:
        print(f"=== MIMIR phase1 {m} {tr[m].shape} ===")
        ae, hid, cfg = _pretrain_one(m, tr[m], va[m], device, epochs=phase1_epochs)
        save_modality_with_config(ae, cfg, str(ae_dir / f"{m}_ae"))
        aes[m], hidden_dims[m], configs[m] = ae, hid, cfg
        mask_values[m] = 0.0

    encoders, decoders = {}, {}
    for m in MODS:
        enc, dec = extract_encoder_decoder_from_pretrained(aes[m])
        encoders[m], decoders[m] = enc, dec

    all_df = {m: pd.concat([tr[m], va[m], te[m]], axis=0) for m in MODS}
    common = list(all_df["rna"].index)
    n_tr, n_va, n_te = len(tr["rna"]), len(va["rna"]), len(te["rna"])
    train_idx = list(range(0, n_tr))
    val_idx = list(range(n_tr, n_tr + n_va))
    ds = MultiOmicDataset({m: all_df[m].loc[common] for m in MODS})
    train_loader = get_dataloader(ds, batch_size=64, shuffle=True, split_idx=train_idx)
    val_loader = get_dataloader(ds, batch_size=64, shuffle=False, split_idx=val_idx)

    model = MultiModalWithSharedSpace(
        encoders=encoders, decoders=decoders, hidden_dims=hidden_dims,
        shared_dim=256, proj_depth=1,
    ).to(device)
    opt = Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)

    print("=== MIMIR phase2 shared ===")
    best, pat, state = float("inf"), 0, None
    for ep in range(1, phase2_epochs + 1):
        tr_stats = finetune_epoch(
            model=model, dataloader=train_loader, optimizer=opt, device=device,
            mask_values=mask_values, lambda_contrastive=1.0, lambda_recon=1.0,
            lambda_impute=1.0, modality_dropout_prob=0.4, feature_mask_p=0.15,
            alpha_mask_recon=0.5, two_path_clean_for_contrast=False,
        )
        va_stats = eval_finetune_epoch(
            model=model, dataloader=val_loader, device=device,
            mask_values=mask_values, lambda_contrastive=1.0, lambda_recon=1.0,
            lambda_impute=1.0, feature_mask_p=0.15, alpha_mask_recon=0.5,
            two_path_clean_for_contrast=False,
        )
        val = float(va_stats["total_loss"])
        mark = ""
        if val < best:
            best, pat, mark = val, 0, " *"
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            pat += 1
        print(f"  MIMIR shared ep {ep:03d} train={tr_stats['total_loss']:.4f} "
              f"val={val:.4f} recon={va_stats['recon_loss']:.4f} "
              f"impute={va_stats['impute_loss']:.4f}{mark}")
        if ep > 15 and pat >= 20:
            print(f"  MIMIR shared early stop at {ep}")
            break
    model.load_state_dict(state)
    ckpt = {
        "state": state, "hidden_dims": hidden_dims, "mask_values": mask_values,
        "shared_dim": 256, "proj_depth": 1, "modalities": list(MODS),
        "val": best, "n_train": n_tr, "n_val": n_va, "n_test": n_te,
    }
    path = save_dir / "shared_best.pt"
    torch.save(ckpt, path)
    print(f"saved {path} val={best:.4f}")
    return model, mask_values


def load_mimir(save_dir, device):
    save_dir = Path(save_dir)
    ck = torch.load(save_dir / "shared_best.pt", map_location=device, weights_only=False)
    encoders, decoders, hidden_dims, mask_values = {}, {}, {}, {}
    for m in ck["modalities"]:
        ae, hid, cfg = load_modality_with_config(
            str(save_dir / "aes" / f"{m}_ae.pt"), map_location=device)
        enc, dec = extract_encoder_decoder_from_pretrained(ae)
        encoders[m], decoders[m] = enc.to(device), dec.to(device)
        hidden_dims[m] = hid
        mask_values[m] = cfg.get("mask_value", 0.0)
    model = MultiModalWithSharedSpace(
        encoders=encoders, decoders=decoders, hidden_dims=hidden_dims,
        shared_dim=ck["shared_dim"], proj_depth=ck["proj_depth"],
    ).to(device)
    model.load_state_dict(ck["state"])
    model.eval()
    return model, mask_values


def predict_values(model, mask_values, corrupted_dfs, device, batch_size=64):
    """칸 결측(NaN)을 채운다. 반환: {mod: ndarray}."""
    raw = impute_missing_values(
        model=model, mask_values=mask_values, data_corrupted=corrupted_dfs,
        batch_size=batch_size, device=device, self_weight=10.0,
    )
    out = {}
    for m, (xt, samples) in raw.items():
        df = pd.DataFrame(xt.numpy(), index=samples, columns=corrupted_dfs[m].columns)
        out[m] = df.loc[corrupted_dfs[m].index].to_numpy(np.float32)
    return out


def predict_block(model, mask_values, present_dfs, target, columns, index, device, batch_size=64):
    """타깃 모달리티 전체를 나머지에서 복원한다. 행 순서는 index."""
    raw = impute_missing_modalities_for_scenario(
        model=model, mask_values=mask_values, data_present=present_dfs,
        target_modalities=[target], batch_size=batch_size, device=device,
    )
    xt, samples = raw[target]
    df = pd.DataFrame(xt.numpy(), index=samples, columns=columns)
    return df.loc[index].to_numpy(np.float32)
