#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PIMMS-DAE (Webel et al. Nat Commun 2024).

pimms-learn AETransformer(model='DAE', hidden_layers=[512], latent_dim=50)와
같은 구조: 넓은 행렬에 자기지도학습 DAE. 멀티오믹스는 이어 붙여 한 행렬로 넣는다.
환경에 pimms-learn을 설치하지 않고 같은 인터페이스만 재현한다.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class _DAE(nn.Module):
    def __init__(self, n_in, hidden=512, latent=50):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(),
            nn.Linear(hidden, latent), nn.ReLU(),
        )
        self.dec = nn.Sequential(
            nn.Linear(latent, hidden), nn.ReLU(),
            nn.Linear(hidden, n_in),
        )

    def forward(self, x):
        return self.dec(self.enc(x))


class PimmsDAE:
    def __init__(self, hidden=512, latent=50, mask_p=0.2, lr=1e-3,
                 batch_size=32, epochs=100, patience=15, device="cpu"):
        self.hidden, self.latent = hidden, latent
        self.mask_p, self.lr = mask_p, lr
        self.batch_size, self.epochs, self.patience = batch_size, epochs, patience
        self.device = torch.device(device)
        self.net = None

    def _prep(self, X):
        X = np.asarray(X, dtype=np.float32)
        obs = np.isfinite(X)
        Xin = np.where(obs, X, 0.0).astype(np.float32)
        return Xin, obs

    def fit(self, Xtr, Xva=None):
        Xin, obs = self._prep(Xtr)
        n_in = Xin.shape[1]
        self.net = _DAE(n_in, self.hidden, self.latent).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        ds = TensorDataset(torch.from_numpy(Xin), torch.from_numpy(obs.astype(np.float32)))
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True, drop_last=False)
        va = None if Xva is None else self._prep(Xva)
        best, pat, state = float("inf"), 0, None
        for epoch in range(1, self.epochs + 1):
            self.net.train()
            for xb, ob in loader:
                xb, ob = xb.to(self.device), ob.to(self.device)
                extra = (torch.rand_like(xb) < self.mask_p) & (ob > 0.5)
                xin = xb.clone()
                xin[extra] = 0.0
                pred = self.net(xin)
                valid = ob > 0.5
                if valid.any():
                    loss = ((pred - xb) ** 2)[valid].mean()
                else:
                    continue
                opt.zero_grad()
                loss.backward()
                opt.step()
            val = self._val_mse(va) if va is not None else float(loss.item())
            mark = ""
            if val < best:
                best, pat, mark = val, 0, " *"
                state = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
            else:
                pat += 1
            if epoch == 1 or epoch % 10 == 0 or mark:
                print(f"  PIMMS-DAE epoch {epoch:3d} val_mse={val:.4f}{mark}")
            if epoch > 8 and pat >= self.patience:
                print(f"  PIMMS-DAE early stop at {epoch}")
                break
        self.net.load_state_dict(state)
        self.net.eval()
        return self

    @torch.no_grad()
    def _val_mse(self, va):
        Xin, obs = va
        x = torch.from_numpy(Xin).to(self.device)
        ob = torch.from_numpy(obs.astype(np.float32)).to(self.device)
        extra = (torch.rand_like(x) < self.mask_p) & (ob > 0.5)
        xin = x.clone()
        xin[extra] = 0.0
        self.net.eval()
        pred = self.net(xin)
        valid = extra
        if valid.any():
            return float(((pred - x) ** 2)[valid].mean().item())
        valid = ob > 0.5
        return float(((pred - x) ** 2)[valid].mean().item())

    @torch.no_grad()
    def transform(self, X):
        """결측(NaN)만 채우고 관측 칸은 그대로 둔다."""
        self.net.eval()
        Xin, obs = self._prep(X)
        x = torch.from_numpy(Xin).to(self.device)
        pred = self.net(x).cpu().numpy()
        out = np.where(obs, np.asarray(X, dtype=np.float32), pred)
        return out.astype(np.float32)
