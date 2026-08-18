#!/usr/bin/env python3
"""
범용 Tri-joint 학습 스크립트 (샘플 수/특징 수 가변 지원)
"""
from __future__ import annotations

import argparse
import os
import json
import numpy as np
import torch

from dataloader_universal import load_and_align_omics, get_triple_dataloaders_flexible
from models import Generator, Critic


def set_seed(seed: int | None):
    if seed is None:
        return
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class WGAN_GP_Trainer_Tri:
    def __init__(self, Gp, Gr, Gm, Dp, Dr, Dm, device,
                 lr_g=1e-4, lr_d=1e-4, lambda_gp=10.0, n_critic=5,
                 dynamic_masking: bool = False, mask_rate_range=(0.05, 0.5)):
        self.Gp, self.Gr, self.Gm = Gp.to(device), Gr.to(device), Gm.to(device)
        self.Dp, self.Dr, self.Dm = Dp.to(device), Dr.to(device), Dm.to(device)
        self.device = device
        self.lambda_gp, self.n_critic = lambda_gp, n_critic
        self.dynamic_masking, self.mask_rate_range = dynamic_masking, mask_rate_range

        self.optimizer_g = torch.optim.Adam(
            list(Gp.parameters()) + list(Gr.parameters()) + list(Gm.parameters()),
            lr=lr_g, betas=(0.5, 0.9))
        self.optimizer_d = torch.optim.Adam(
            list(Dp.parameters()) + list(Dr.parameters()) + list(Dm.parameters()),
            lr=lr_d, betas=(0.5, 0.9))

    def _gp(self, D, real, fake):
        alpha = torch.rand(real.size(0), 1, device=self.device).expand_as(real)
        inter = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
        out = D(inter)
        grad = torch.autograd.grad(out, inter, grad_outputs=torch.ones_like(out),
                                   create_graph=True, retain_graph=True, only_inputs=True)[0]
        return ((grad.norm(2, dim=1) - 1) ** 2).mean()

    def _dyn_mask(self, real_mask, like):
        observed = (1.0 - real_mask).clamp(0, 1)
        rate = float(np.random.uniform(self.mask_rate_range[0], self.mask_rate_range[1]))
        return (torch.rand_like(like) < rate).float() * observed

    def train_step(self, batch):
        rna = batch["x_rna"].to(self.device)
        prot = batch["x_prot"].to(self.device)
        methy = batch["x_methy"].to(self.device)
        m_rna = batch["m_rna"].to(self.device)
        m_prot = batch["m_prot"].to(self.device)
        m_methy = batch["m_methy"].to(self.device)

        for _ in range(self.n_critic):
            self.optimizer_d.zero_grad()
            y_p_fake = self.Gp(torch.cat([rna, methy], dim=1), src=None).detach()
            gp = self._gp(self.Dp, prot, y_p_fake)
            loss_Dp = -(self.Dp(prot).mean() - self.Dp(y_p_fake).mean()) + self.lambda_gp * gp

            y_r_fake = self.Gr(torch.cat([prot, methy], dim=1), src=None).detach()
            gr = self._gp(self.Dr, rna, y_r_fake)
            loss_Dr = -(self.Dr(rna).mean() - self.Dr(y_r_fake).mean()) + self.lambda_gp * gr

            y_m_fake = self.Gm(torch.cat([rna, prot], dim=1), src=None).detach()
            gm = self._gp(self.Dm, methy, y_m_fake)
            loss_Dm = -(self.Dm(methy).mean() - self.Dm(y_m_fake).mean()) + self.lambda_gp * gm

            (loss_Dp + loss_Dr + loss_Dm).backward()
            self.optimizer_d.step()

        self.optimizer_g.zero_grad()
        y_p = self.Gp(torch.cat([rna, methy], dim=1), src=None)
        y_r = self.Gr(torch.cat([prot, methy], dim=1), src=None)
        y_m = self.Gm(torch.cat([rna, prot], dim=1), src=None)
        g_wgan = -(self.Dp(y_p).mean() + self.Dr(y_r).mean() + self.Dm(y_m).mean())

        if self.dynamic_masking:
            mp = self._dyn_mask(m_prot, y_p)
            mr = self._dyn_mask(m_rna, y_r)
            mm = self._dyn_mask(m_methy, y_m)
        else:
            mp, mr, mm = m_prot, m_rna, m_methy

        mse = 0.0
        if mp.sum() > 0:
            mse += ((y_p - prot) * mp).pow(2).mean()
        if mr.sum() > 0:
            mse += ((y_r - rna) * mr).pow(2).mean()
        if mm.sum() > 0:
            mse += ((y_m - methy) * mm).pow(2).mean()

        g_total = g_wgan + 0.1 * mse
        g_total.backward()
        self.optimizer_g.step()
        return {"D": float((loss_Dp + loss_Dr + loss_Dm).item()), "G": float(g_total.item())}

    def train_epoch(self, loader):
        self.Gp.train(); self.Gr.train(); self.Gm.train()
        self.Dp.train(); self.Dr.train(); self.Dm.train()
        logsD, logsG = [], []
        for batch in loader:
            out = self.train_step(batch)
            logsD.append(out["D"]); logsG.append(out["G"])
        return {"D": float(np.mean(logsD)), "G": float(np.mean(logsG))}

    @torch.no_grad()
    def validate_rmse(self, valid_loader, seed: int = 999, mask_rate: float = 0.30):
        rng = np.random.default_rng(seed)
        def fixed_mask(real_mask):
            observed = (1.0 - real_mask).clamp(0, 1)
            samp = torch.from_numpy((rng.random(observed.shape) < mask_rate).astype(np.float32)).to(observed.device)
            return samp * observed

        self.Gp.eval(); self.Gr.eval(); self.Gm.eval()
        sum_sq = {"protein": 0.0, "rna": 0.0, "methyl": 0.0}
        cnt = {"protein": 0.0, "rna": 0.0, "methyl": 0.0}
        for batch in valid_loader:
            rna = batch["x_rna"].to(self.device)
            prot = batch["x_prot"].to(self.device)
            methy = batch["x_methy"].to(self.device)
            m_rna = batch["m_rna"].to(self.device)
            m_prot = batch["m_prot"].to(self.device)
            m_methy = batch["m_methy"].to(self.device)

            mp, mr, mm = fixed_mask(m_prot), fixed_mask(m_rna), fixed_mask(m_methy)
            yp = self.Gp(torch.cat([rna, methy], dim=1), src=None)
            yr = self.Gr(torch.cat([prot, methy], dim=1), src=None)
            ym = self.Gm(torch.cat([rna, prot], dim=1), src=None)
            sum_sq["protein"] += ((prot - yp) * mp).pow(2).sum().item(); cnt["protein"] += mp.sum().item()
            sum_sq["rna"] += ((rna - yr) * mr).pow(2).sum().item(); cnt["rna"] += mr.sum().item()
            sum_sq["methyl"] += ((methy - ym) * mm).pow(2).sum().item(); cnt["methyl"] += mm.sum().item()
        rmse = {k: float(np.sqrt(sum_sq[k] / max(cnt[k], 1.0))) for k in sum_sq}
        rmse["avg"] = float(np.mean([rmse["protein"], rmse["rna"], rmse["methyl"]]))
        return rmse


def main():
    parser = argparse.ArgumentParser(description="범용 멀티오믹스 학습 (1~3개 modality)")
    parser.add_argument("--rna", default=None, help="Omics slot 1 TSV (features x samples)")
    parser.add_argument("--methy", default=None, help="Omics slot 2 TSV (features x samples)")
    parser.add_argument("--protein", default=None, help="Omics slot 3 TSV (features x samples)")
    parser.add_argument("--output_dir", required=True, help="결과 저장 디렉토리")

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--use_attention", action="store_true")
    parser.add_argument("--n_critic", type=int, default=5)
    parser.add_argument("--early_stopping_patience", type=int, default=10)
    parser.add_argument("--min_epochs", type=int, default=5)
    parser.add_argument("--dynamic_masking", action="store_true")
    parser.add_argument("--mask_rate_min", type=float, default=0.05)
    parser.add_argument("--mask_rate_max", type=float, default=0.50)
    parser.add_argument("--val_mask_rate", type=float, default=0.30)
    parser.add_argument("--val_mask_seed", type=int, default=123)
    parser.add_argument("--sample_id_mode", type=str, default="none", choices=["none", "tcga12", "tcga15"])
    parser.add_argument("--sample_join", type=str, default="intersection", choices=["intersection", "union"])
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()

    n_provided = sum(1 for p in [args.rna, args.methy, args.protein] if p)
    if n_provided < 2:
        parser.error("최소 2개 이상의 omics 파일이 필요합니다 (--rna, --methy, --protein 중 2개 이상).")

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    tables = load_and_align_omics(
        rna_path=args.rna,
        methy_path=args.methy,
        protein_path=args.protein,
        sample_id_mode=args.sample_id_mode,
        sample_join=args.sample_join,
    )

    train_loader, valid_loader, data_info = get_triple_dataloaders_flexible(
        rna=tables.rna,
        methy=tables.methy,
        protein=tables.protein,
        batch_size=args.batch_size,
        split=(0.8, 0.2),
        shuffle=True,
        seed=42,
    )

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    dim_r = tables.rna.shape[0]
    dim_p = tables.protein.shape[0]
    dim_m = tables.methy.shape[0]

    Gp = Generator(input_size=dim_r + dim_m, output_size=dim_p,
                   use_attn=args.use_attention, n_heads=4, d_head=64,
                   target_type="protein", src_size=dim_r + dim_m)
    Gr = Generator(input_size=dim_p + dim_m, output_size=dim_r,
                   use_attn=args.use_attention, n_heads=4, d_head=64,
                   target_type="rna", src_size=dim_p + dim_m)
    Gm = Generator(input_size=dim_r + dim_p, output_size=dim_m,
                   use_attn=args.use_attention, n_heads=4, d_head=64,
                   target_type="methyl", src_size=dim_r + dim_p)

    Dp = Critic(input_size=dim_p, hidden_dim=512)
    Dr = Critic(input_size=dim_r, hidden_dim=512)
    Dm = Critic(input_size=dim_m, hidden_dim=512)

    trainer = WGAN_GP_Trainer_Tri(
        Gp, Gr, Gm, Dp, Dr, Dm, device,
        lr_g=args.learning_rate,
        lr_d=args.learning_rate,
        lambda_gp=10.0,
        n_critic=args.n_critic,
        dynamic_masking=args.dynamic_masking,
        mask_rate_range=(args.mask_rate_min, args.mask_rate_max),
    )

    best_metric = float("inf")
    patience_counter = 0

    meta = {
        "dims": {"rna": dim_r, "protein": dim_p, "methy": dim_m},
        "features": {
            "rna": list(tables.rna.index),
            "protein": list(tables.protein.index),
            "methy": list(tables.methy.index),
        },
        "sample_ids": tables.sample_ids,
        "data_info": data_info,
        "config": vars(args),
    }

    for epoch in range(args.num_epochs):
        log = trainer.train_epoch(train_loader)
        rmse = trainer.validate_rmse(valid_loader, seed=args.val_mask_seed, mask_rate=args.val_mask_rate)
        avg_rmse = rmse["avg"]
        improved = avg_rmse < best_metric

        status = "🏆" if improved else "📈"
        print(f"Epoch {epoch+1:3d}/{args.num_epochs} | {status} G: {log['G']:7.3f} | D: {log['D']:8.3f} "
              f"| Val RMSE avg={avg_rmse:.6f} (P={rmse['protein']:.6f}, R={rmse['rna']:.6f}, M={rmse['methyl']:.6f}) "
              f"| Best={best_metric:.6f}")

        if improved:
            best_metric = avg_rmse
            patience_counter = 0
            ck_best = {
                "Gp": Gp.state_dict(), "Gr": Gr.state_dict(), "Gm": Gm.state_dict(),
                "Dp": Dp.state_dict(), "Dr": Dr.state_dict(), "Dm": Dm.state_dict(),
                "epoch": epoch, "val_rmse": rmse, "meta": meta
            }
            torch.save(ck_best, os.path.join(args.output_dir, "tri_best.ckpt"))
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or (epoch + 1) == args.num_epochs:
            ck = {
                "Gp": Gp.state_dict(), "Gr": Gr.state_dict(), "Gm": Gm.state_dict(),
                "Dp": Dp.state_dict(), "Dr": Dr.state_dict(), "Dm": Dm.state_dict(),
                "epoch": epoch, "meta": meta
            }
            torch.save(ck, os.path.join(args.output_dir, f"tri_epoch_{epoch+1}.ckpt"))

        if epoch >= args.min_epochs and patience_counter >= args.early_stopping_patience:
            print(f"\n🛑 Early Stopping! {args.early_stopping_patience} 에포크 동안 개선 없음")
            break

    ck_final = {
        "Gp": Gp.state_dict(), "Gr": Gr.state_dict(), "Gm": Gm.state_dict(),
        "Dp": Dp.state_dict(), "Dr": Dr.state_dict(), "Dm": Dm.state_dict(),
        "epoch": args.num_epochs - 1, "meta": meta
    }
    torch.save(ck_final, os.path.join(args.output_dir, "tri_final.ckpt"))

    with open(os.path.join(args.output_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✅ 학습 완료! 결과 저장: {args.output_dir}")


if __name__ == "__main__":
    main()
