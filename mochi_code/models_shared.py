#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
공유 잠재공간 MOCHI.

MIMIR과 같은 두 경로:
  - 재구성: 각 오믹스는 자기 z로만 디코드
  - 번역/블록: 다른 오믹스 z를 평균 또는 교차-어텐션으로 합쳐 디코드
칸 결측 추론은 자기 z에 가중(self_weight=10). GAN 없음.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MODS = ("protein", "rna", "methyl")
HIDDEN = {"rna": 512, "protein": 128, "methyl": 256}


class TokenFuse(nn.Module):
    """있는 모달리티 토큰만 보는 교차-어텐션. 출력은 샘플당 공유 z 하나."""

    def __init__(self, d_model=256, n_heads=4, pdrop=0.1):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=pdrop, batch_first=True)
        self.ff1 = nn.Linear(d_model, 4 * d_model)
        self.ff2 = nn.Linear(4 * d_model, d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(pdrop)

    def forward(self, tokens: torch.Tensor, present: torch.Tensor):
        """
        tokens:  [B, M, D]
        present: [B, M] bool, True = 이 토큰을 씀
        """
        b = tokens.size(0)
        q = self.query.expand(b, -1, -1)
        keep = present.clone()
        empty = ~keep.any(dim=1)
        if empty.any():
            keep[empty, 0] = True
        ignore = ~keep
        attn_out, w = self.attn(q, tokens, tokens, key_padding_mask=ignore, need_weights=True)
        h = self.ln1(q.squeeze(1) + self.drop(attn_out.squeeze(1)))
        y = self.ln2(h + self.drop(self.ff2(F.relu(self.ff1(h)))))
        self.last_attn = w.detach()
        return y


class MaskedMean(nn.Module):
    def forward(self, tokens: torch.Tensor, present: torch.Tensor):
        keep = present.clone()
        empty = ~keep.any(dim=1)
        if empty.any():
            keep[empty, 0] = True
        w = keep.float().unsqueeze(-1)
        return (tokens * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)


class SharedMOCHI(nn.Module):
    def __init__(self, dims: Dict[str, int], shared_dim=256, n_heads=4, fuse="attn"):
        super().__init__()
        self.mods = tuple(MODS)
        self.shared_dim = shared_dim
        self.fuse_name = fuse
        self.encoders = nn.ModuleDict()
        self.decoders = nn.ModuleDict()
        self.projections = nn.ModuleDict()
        self.rev = nn.ModuleDict()
        for m in self.mods:
            h = HIDDEN[m]
            self.encoders[m] = nn.Linear(dims[m], h)
            self.decoders[m] = nn.Linear(h, dims[m])
            self.projections[m] = nn.Linear(h, shared_dim)
            self.rev[m] = nn.Linear(shared_dim, h)
        self.fuse = TokenFuse(shared_dim, n_heads=n_heads) if fuse == "attn" else MaskedMean()

    def encode_z(self, xs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = {}
        for m, x in xs.items():
            h = self.encoders[m](x)
            out[m] = self.projections[m](h)
        return out

    def _stack(self, zs: Dict[str, torch.Tensor], present: Dict[str, torch.Tensor], ref):
        b, d = ref.size(0), self.shared_dim
        tokens = ref.new_zeros(b, len(self.mods), d)
        mask = torch.zeros(b, len(self.mods), dtype=torch.bool, device=ref.device)
        for i, m in enumerate(self.mods):
            if m in zs:
                tokens[:, i] = zs[m]
                mask[:, i] = present[m]
        return tokens, mask

    def fuse_zs(self, zs: Dict[str, torch.Tensor], present: Dict[str, torch.Tensor]) -> torch.Tensor:
        ref = next(iter(zs.values()))
        tokens, mask = self._stack(zs, present, ref)
        return self.fuse(tokens, mask)

    def fused_z(self, xs: Dict[str, torch.Tensor], present: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.fuse_zs(self.encode_z(xs), present)

    def decode(self, z: torch.Tensor, targets: Iterable[str]) -> Dict[str, torch.Tensor]:
        return {m: self.decoders[m](self.rev[m](z)) for m in targets}

    def reconstruct_own(self, xs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """각 오믹스는 자기 z로만 재구성. MIMIR recon 경로."""
        zs = self.encode_z(xs)
        return {m: self.decoders[m](self.rev[m](z)) for m, z in zs.items()}

    def weighted_z(self, zs: Dict[str, torch.Tensor], target: str, self_weight=10.0) -> torch.Tensor:
        """칸 결측: 자기 z에 self_weight, 나머지는 1. MIMIR impute_missing_values와 동일."""
        z_list, w_list = [], []
        for m, z in zs.items():
            z_list.append(z)
            w_list.append(self_weight if m == target else 1.0)
        w = torch.tensor(w_list, device=z_list[0].device, dtype=z_list[0].dtype)
        w = w / w.sum()
        return (torch.stack(z_list, dim=0) * w.view(-1, 1, 1)).sum(dim=0)

    def reconstruct(self, xs: Dict[str, torch.Tensor], present: Dict[str, torch.Tensor],
                    targets: Optional[Iterable[str]] = None) -> Dict[str, torch.Tensor]:
        z = self.fused_z(xs, present)
        return self.decode(z, targets if targets is not None else xs.keys())

    def loo_reconstruct(self, xs: Dict[str, torch.Tensor], present: Dict[str, torch.Tensor],
                        target: str) -> torch.Tensor:
        xs_loo = {m: x for m, x in xs.items() if m != target}
        present_loo = {m: p.clone() for m, p in present.items() if m != target}
        z = self.fused_z(xs_loo, present_loo)
        return self.decode(z, [target])[target]


def contrastive_loss(zs: Dict[str, torch.Tensor], present: Dict[str, torch.Tensor], tau=0.1):
    mods = [m for m in zs if present[m].any()]
    if len(mods) < 2:
        return next(iter(zs.values())).new_zeros(())
    loss, n = 0.0, 0
    for i, a in enumerate(mods):
        for b in mods[i + 1:]:
            both = present[a] & present[b]
            if both.sum() < 2:
                continue
            z1, z2 = F.normalize(zs[a][both], dim=1), F.normalize(zs[b][both], dim=1)
            sim = z1 @ z2.T / tau
            labels = torch.arange(z1.size(0), device=z1.device)
            loss = loss + F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)
            n += 2
    if n == 0:
        return next(iter(zs.values())).new_zeros(())
    return loss / n


def mse_valid(pred, tgt, valid):
    if valid.any():
        return ((pred - tgt)[valid] ** 2).mean()
    return pred.new_zeros(())


def drop_modalities(present: Dict[str, torch.Tensor], p=0.4):
    out = {m: v.clone() for m, v in present.items()}
    ref = next(iter(out.values()))
    b, device = ref.size(0), ref.device
    for m in out:
        out[m] = out[m] & (torch.rand(b, device=device) >= p)
    stacked = torch.stack([out[m] for m in out], dim=1)
    empty = ~stacked.any(dim=1)
    if empty.any():
        idx = empty.nonzero(as_tuple=False).squeeze(1)
        choice = torch.randint(0, len(out), (idx.numel(),), device=device)
        keys = list(out.keys())
        for j, row in enumerate(idx.tolist()):
            out[keys[int(choice[j].item())]][row] = True
    return out


def mask_cells(x, obs, p):
    extra = (torch.rand_like(x) < p) & obs
    xin = x.clone()
    xin[extra] = 0.0
    return xin, extra


@torch.no_grad()
def predict_shared(model: SharedMOCHI, tabs: Dict[str, np.ndarray], device,
                   missing: Optional[str] = None, batch_size=64, self_weight=10.0) -> Dict[str, np.ndarray]:
    """missing=None: 칸 결측(자기 z 가중 + 다른 z). missing=mod: 블록(다른 z만)."""
    model.eval()
    n = next(iter(tabs.values())).shape[0]
    acc = {m: [] for m in MODS}
    for i in range(0, n, batch_size):
        sl = slice(i, i + batch_size)
        xs, present = {}, {}
        b = None
        for m in MODS:
            if missing is not None and m == missing:
                continue
            t = torch.from_numpy(np.asarray(tabs[m][sl], dtype=np.float32)).to(device)
            xs[m] = t
            present[m] = torch.ones(t.size(0), dtype=torch.bool, device=device)
            b = t.size(0)
        if missing is not None:
            present[missing] = torch.zeros(b, dtype=torch.bool, device=device)
            hat = model.reconstruct(xs, present, targets=MODS)
        else:
            zs = model.encode_z(xs)
            hat = {}
            for tgt in MODS:
                if model.fuse_name == "mean":
                    z = model.weighted_z(zs, tgt, self_weight=self_weight)
                else:
                    z = model.fuse_zs(zs, present)
                hat[tgt] = model.decode(z, [tgt])[tgt]
        for m in MODS:
            acc[m].append(hat[m].cpu().numpy())
    return {m: np.concatenate(acc[m], 0) for m in MODS}


def load_shared(path, dims, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = SharedMOCHI(
        dims, shared_dim=ck.get("shared_dim", 256), fuse=ck.get("fuse", "attn"),
    ).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model
