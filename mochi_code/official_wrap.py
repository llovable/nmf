#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
공식 OmicsNMF / OmiTrans 네트워크를 우리 BRCA split에 붙인다.

고친 것 (공정 비교용, 아키텍처·손실은 공식 그대로)
- 공식 train.py는 test 로더에서 G를 한 번 더 업데이트한다. 여기서는 하지 않는다.
- 공식 로더는 교집합을 랜덤 80/20으로 나눈다. 우리는 고정 train/val/test를 쓴다.
- 점수는 우리 z-RMSE. OmicsNMF 최종층이 ReLU라 타깃은 train min으로 비음수화한 뒤
  예측을 다시 z-점수로 되돌린다.

baselines/ 코드는 수정하지 않는다.
"""

from __future__ import annotations

import importlib.util
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.decomposition import NMF
from sklearn.exceptions import ConvergenceWarning
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path("/home/dyan/nmf")
OMICSNMF_MODELS = ROOT / "baselines/OmicsNMF/codes/models.py"
OMITRANS_NET = ROOT / "baselines/OmiTrans/models/networks.py"
OMITRANS_LOSS = ROOT / "baselines/OmiTrans/models/losses.py"

PAIRS_1TO1 = (
    ("protein", "rna"),
    ("rna", "protein"),
    ("methyl", "rna"),
)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_onmf = None
_ot_net = None
_ot_loss = None


def omicsnmf_modules():
    global _onmf
    if _onmf is None:
        _onmf = _load(OMICSNMF_MODELS, "omicsnmf_official_models")
    return _onmf


def omitrans_modules():
    global _ot_net, _ot_loss
    if _ot_net is None:
        _ot_net = _load(OMITRANS_NET, "omitrans_official_networks")
        _ot_loss = _load(OMITRANS_LOSS, "omitrans_official_losses")
    return _ot_net, _ot_loss


def nmf_U_V(X, k=10, init="random", random_state=1111):
    """OmicsNMF/codes/utils.py::nmf_U_V 와 동일. 비음수 clip만 추가."""
    X = np.clip(np.asarray(X, dtype=np.float64), 1e-8, None)
    k = int(max(2, min(k, X.shape[0], X.shape[1])))
    model = NMF(n_components=k, init=init, random_state=random_state)
    U = model.fit_transform(X)
    V = model.components_
    return U, V


class NonnegShift:
    """ReLU 생성기와 NMF를 위해 특징별 최솟값을 0 이상으로 옮긴다."""

    def __init__(self, Y_train: np.ndarray):
        Y_train = np.asarray(Y_train, dtype=np.float32)
        self.shift = np.maximum(0.0, -np.nanmin(Y_train, axis=0)).astype(np.float32)

    def to_nn(self, Y):
        return np.asarray(Y, dtype=np.float32) + self.shift

    def from_nn(self, Ynn):
        return np.asarray(Ynn, dtype=np.float32) - self.shift


def load_omicsnmf_ckpt(path, src_dim, tgt_dim, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    mdl = OmicsNMFOfficial(src_dim, tgt_dim, device)
    mdl.G.load_state_dict(ck["G"])
    sh = NonnegShift(np.zeros((1, tgt_dim), np.float32))
    sh.shift = np.asarray(ck["shift"], dtype=np.float32)
    mdl.shift = sh
    mdl.G.eval()
    return mdl


def load_omitrans_ckpt(path, src_dim, tgt_dim, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    mdl = OmiTransOfficial(src_dim, tgt_dim, device)
    mdl.G.load_state_dict(ck["G"])
    mdl.G.eval()
    return mdl


def _rmse(y, yhat):
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))


def _arrays(ds, src, tgt):
    tables = {"rna": ds.rna_f, "protein": ds.prot_f, "methyl": ds.methy_f}
    return tables[src], tables[tgt]


# ---------------------------------------------------------------------------
# OmicsNMF
# ---------------------------------------------------------------------------

class OmicsNMFOfficial:
    def __init__(self, src_dim, tgt_dim, device, k=10, clip=0.01, n_critic=5,
                 lr=5e-5, M=0.1, U=0.1):
        m = omicsnmf_modules()
        self.G = m.Generator(input_size=src_dim, output_size=tgt_dim).to(device)
        self.D = m.Critic(input_size=tgt_dim).to(device)
        self.device = device
        self.clip, self.n_critic = clip, n_critic
        self.M, self.U = M, U
        self.k = k
        self.optD = optim.RMSprop(self.D.parameters(), lr=lr)
        self.optG = optim.RMSprop(self.G.parameters(), lr=lr)
        self.mse = nn.MSELoss()
        self.shift = None
        self.U_global = None

    def fit_basis(self, Y_nn: np.ndarray):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            U, _ = nmf_U_V(Y_nn.T, k=self.k)
        self.U_global = torch.tensor(U, dtype=torch.float32, device=self.device)
        self.k = U.shape[1]

    def train_step(self, src, tgt_nn):
        src = src.to(self.device)
        tgt_nn = tgt_nn.to(self.device)
        for _ in range(self.n_critic):
            self.optD.zero_grad()
            fake = self.G(src).detach()
            loss_d = -(self.D(tgt_nn).mean() - self.D(fake).mean())
            loss_d.backward()
            self.optD.step()
            for p in self.D.parameters():
                p.data.clamp_(-self.clip, self.clip)

        self.optG.zero_grad()
        fake = self.G(src)
        loss_mse = self.mse(fake, tgt_nn)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            U_hat, _ = nmf_U_V(fake.T.detach().cpu().numpy(), k=self.k)
        loss_u = self.mse(self.U_global, torch.tensor(U_hat, dtype=torch.float32, device=self.device))
        loss_g = -self.D(fake).mean() + self.U * loss_u + self.M * loss_mse
        loss_g.backward()
        self.optG.step()
        return float(loss_mse.item())

    @torch.no_grad()
    def predict_z(self, src: np.ndarray) -> np.ndarray:
        self.G.eval()
        x = torch.from_numpy(np.asarray(src, dtype=np.float32)).to(self.device)
        ynn = self.G(x).cpu().numpy()
        self.G.train()
        return self.shift.from_nn(ynn)

    def rmse_z(self, src, tgt_z):
        return _rmse(tgt_z, self.predict_z(src))


def train_omicsnmf_pair(Xtr, Ytr, Xva, Yva, device, epochs=80, patience=15,
                        batch_size=32, k=10):
    shift = NonnegShift(Ytr)
    Ytr_nn, Yva_nn = shift.to_nn(Ytr), shift.to_nn(Yva)
    mdl = OmicsNMFOfficial(Xtr.shape[1], Ytr.shape[1], device, k=k)
    mdl.shift = shift
    mdl.fit_basis(Ytr_nn)

    n = len(Xtr)
    best, pat, best_state = float("inf"), 0, None
    for epoch in range(1, epochs + 1):
        mdl.G.train(); mdl.D.train()
        perm = np.random.permutation(n)
        mses = []
        for i in range(0, n, batch_size):
            sl = perm[i:i + batch_size]
            if len(sl) < 2:
                continue
            mses.append(mdl.train_step(
                torch.from_numpy(Xtr[sl]), torch.from_numpy(Ytr_nn[sl])))
        val = mdl.rmse_z(Xva, Yva)
        mark = ""
        if val < best:
            best, pat, mark = val, 0, " *"
            best_state = {k: v.detach().cpu().clone() for k, v in mdl.G.state_dict().items()}
        else:
            pat += 1
        print(f"  epoch {epoch:3d} mse={np.mean(mses):.4f} val_zRMSE={val:.4f}{mark}")
        if epoch > 8 and pat >= patience:
            print(f"  early stop at {epoch}")
            break
    mdl.G.load_state_dict(best_state)
    return mdl, best


# ---------------------------------------------------------------------------
# OmiTrans (FCG + FCD, λ_dist=100, L1, vanilla GAN)
# ---------------------------------------------------------------------------

class OmiTransOfficial:
    def __init__(self, src_dim, tgt_dim, device, lambda_dist=100.0, dist="L1",
                 lr_g=2e-4, lr_d=2e-4):
        net, loss = omitrans_modules()
        # 공식 FCG는 3D 채널 텐서용으로 BatchNorm→InstanceNorm 치환을 한다.
        # 우리 입력은 샘플×특징 2D 이므로 BatchNorm1d를 그대로 쓴다 (기본 norm_type=batch).
        self.G = net.FCG(src_dim, tgt_dim, norm_layer=nn.BatchNorm1d).to(device)
        self.D = net.FCD(tgt_dim, src_dim, norm_layer=nn.BatchNorm1d).to(device)
        net.init_weights(self.G, "normal", 0.02)
        net.init_weights(self.D, "normal", 0.02)
        self.device = device
        self.lambda_dist = lambda_dist
        self.gan = loss.GANLossObj("vanilla").to(device)
        self.dist = loss.get_dist_loss(dist)
        self.optG = optim.Adam(self.G.parameters(), lr=lr_g, betas=(0.5, 0.999))
        self.optD = optim.Adam(self.D.parameters(), lr=lr_d, betas=(0.5, 0.999))

    def _d(self, fake_or_real, cond):
        # 공식 FCD는 dim=2에서 concat. 2D 입력이면 마지막 축으로 붙인다.
        return self.D.mul_fc(torch.cat([fake_or_real, cond], dim=1))

    def train_step(self, src, tgt):
        src = src.to(self.device)
        tgt = tgt.to(self.device)
        fake = self.G(src)

        self.optD.zero_grad()
        loss_d = (self.gan(self._d(fake.detach(), src), False) + self.gan(self._d(tgt, src), True)) / 2
        loss_d.backward()
        self.optD.step()

        self.optG.zero_grad()
        fake = self.G(src)
        loss_gan = self.gan(self._d(fake, src), True)
        loss_dist = self.dist(fake, tgt)
        loss_g = loss_gan + self.lambda_dist * loss_dist
        loss_g.backward()
        self.optG.step()
        return float(loss_dist.item())

    @torch.no_grad()
    def predict(self, src: np.ndarray) -> np.ndarray:
        self.G.eval()
        x = torch.from_numpy(np.asarray(src, dtype=np.float32)).to(self.device)
        y = self.G(x).cpu().numpy()
        self.G.train()
        return y

    def rmse(self, src, tgt):
        return _rmse(tgt, self.predict(src))


def train_omitrans_pair(Xtr, Ytr, Xva, Yva, device, epochs=100, patience=15,
                        batch_size=16, lambda_dist=100.0):
    mdl = OmiTransOfficial(Xtr.shape[1], Ytr.shape[1], device, lambda_dist=lambda_dist)
    ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Ytr))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
    best, pat, best_state = float("inf"), 0, None
    for epoch in range(1, epochs + 1):
        mdl.G.train(); mdl.D.train()
        dists = [mdl.train_step(x, y) for x, y in loader]
        val = mdl.rmse(Xva, Yva)
        mark = ""
        if val < best:
            best, pat, mark = val, 0, " *"
            best_state = {k: v.detach().cpu().clone() for k, v in mdl.G.state_dict().items()}
        else:
            pat += 1
        print(f"  epoch {epoch:3d} L1={np.mean(dists):.4f} val_zRMSE={val:.4f}{mark}")
        if epoch > 8 and pat >= patience:
            print(f"  early stop at {epoch}")
            break
    mdl.G.load_state_dict(best_state)
    return mdl, best
