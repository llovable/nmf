#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NMF-Transformer MOCHI.

재구성·LOO의 주 경로는 오믹스별 AE 히든.
NMF 계수 W는 두 갈래로 쓰인다. Transformer의 보조 토큰이자,
디코더 출력에 더해지는 저랭크 잔차 γ(Ŵ - W̄)H의 입력이다.
후자는 재학습 없이 기여도를 잘라낼 수 있다. 다만 그 방법은
set_lowrank(False)이지 γ를 0으로 덮어쓰는 것이 아니다.
gamma_nonneg 모델에서 실효 게이트는 softplus(γ)이므로 γ를 0으로 두면
softplus(0)=0.693이 되어 녹아웃이 아니라 게이트를 오히려 키운다.
칸 결측은 자기 히든 가중 + LOO 히든.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models_shared import HIDDEN, MODS

K_DEFAULT = 20
D_MODEL = 128


W_HEAD_ACTS = ("relu", "softplus")


def _inv_softplus(y: float) -> float:
    """softplus(x) = y 를 만족하는 x. gamma_nonneg일 때 초기값을 맞추는 데 쓴다."""
    y = max(float(y), 1e-8)
    if y > 20.0:
        return y
    return float(np.log(np.expm1(y)))


def _inv_softplus_tensor(y: torch.Tensor, floor: float = 1e-8) -> torch.Tensor:
    """벡터용 inv-softplus. 0 성분은 큰 음수 bias로 보내 softplus≈0이 되게 한다."""
    y = y.clamp(min=floor)
    return torch.where(y > 20, y, torch.log(torch.expm1(y)))


class FrozenNMF(nn.Module):
    """sklearn NMF 사전. W = ReLU((x+shift) H⁺)."""

    def __init__(self, H: torch.Tensor, shift: torch.Tensor, HHt_inv: torch.Tensor,
                 w_mean: Optional[torch.Tensor] = None):
        super().__init__()
        self.register_buffer("H", H)
        self.register_buffer("HHt_inv", HHt_inv)
        if shift.dim() == 1:
            shift = shift.view(1, -1)
        self.register_buffer("shift", shift)
        k = int(H.size(0))
        if w_mean is None:
            w_mean = torch.zeros(k, dtype=H.dtype, device=H.device)
        self.register_buffer("w_mean", w_mean.view(-1))
        self.k = k

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        ys = (x + self.shift).clamp(min=1e-8)
        return F.relu(ys @ self.H.T @ self.HHt_inv)

    def decode_dev(self, W: torch.Tensor) -> torch.Tensor:
        """코호트 평균 성분 구성에서의 편차만 z 공간으로 되돌린다.

        x + shift ≈ W H 이므로 평균을 빼면 shift가 소거되고, 환자별 성분
        활성의 편차가 그대로 남는다.
        """
        return (W - self.w_mean) @ self.H


class NMFTransformer(nn.Module):
    """AE 히든 평균이 주 경로. NMF W 토큰은 Transformer 잔차에만 씀."""

    def __init__(self, k=K_DEFAULT, d_model=D_MODEL, n_heads=4, n_layers=2, pdrop=0.1,
                 use_nmf_tokens=True, use_transformer=True):
        super().__init__()
        self.k = k
        self.d_model = d_model
        self.use_nmf_tokens = use_nmf_tokens
        self.use_transformer = use_transformer
        self.mods = tuple(MODS)
        n_mods = len(self.mods)
        self.proj_h = nn.ModuleDict({m: nn.Linear(HIDDEN[m], d_model) for m in self.mods})
        self.comp_emb = nn.Parameter(torch.randn(n_mods, k, d_model) * 0.02)
        self.mod_emb = nn.Parameter(torch.randn(n_mods, 1, d_model) * 0.02)
        self.h_emb = nn.Parameter(torch.randn(n_mods, 1, d_model) * 0.02)
        self.w_in = nn.Linear(1, d_model)
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=4 * d_model, dropout=pdrop,
            batch_first=True, activation="gelu", norm_first=True)
        try:
            self.encoder = nn.TransformerEncoder(enc_layer, n_layers, enable_nested_tensor=False)
        except TypeError:
            self.encoder = nn.TransformerEncoder(enc_layer, n_layers)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=pdrop, batch_first=True)
        self.ln = nn.LayerNorm(d_model)
        self.delta = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)

    def mean_z(self, hs: Dict[str, torch.Tensor], keep: Dict[str, torch.Tensor],
               skip: Optional[str] = None) -> torch.Tensor:
        ref = next(iter(hs.values()))
        acc = ref.new_zeros(ref.size(0), self.d_model)
        wsum = ref.new_zeros(ref.size(0), 1)
        for m, h in hs.items():
            if m == skip:
                continue
            present = keep[m].float().unsqueeze(-1)
            acc = acc + self.proj_h[m](h) * present
            wsum = wsum + present
        return acc / wsum.clamp_min(1.0)

    def _stack(self, hs: Dict[str, torch.Tensor], Ws: Dict[str, torch.Tensor],
               keep: Dict[str, torch.Tensor]):
        ref = next(iter(hs.values())) if hs else next(iter(Ws.values()))
        b, device = ref.size(0), ref.device
        toks, pads = [], []
        for i, m in enumerate(self.mods):
            present = keep.get(m, torch.zeros(b, dtype=torch.bool, device=device))
            if m in hs:
                htok = self.proj_h[m](hs[m]).unsqueeze(1) + self.h_emb[i]
            else:
                htok = ref.new_zeros(b, 1, self.d_model)
            toks.append(htok)
            pads.append((~present).unsqueeze(1))
            if self.use_nmf_tokens:
                W = Ws[m] if m in Ws else ref.new_zeros(b, self.k)
                wtok = self.w_in(W.unsqueeze(-1)) + self.comp_emb[i] + self.mod_emb[i]
                toks.append(wtok)
                pads.append((~present).unsqueeze(1).expand(-1, self.k))
        tokens = torch.cat(toks, dim=1)
        pad = torch.cat(pads, dim=1)
        empty = pad.all(dim=1)
        if empty.any():
            pad = pad.clone()
            pad[empty, 0] = False
        return tokens, pad

    def fused_z(self, hs: Dict[str, torch.Tensor], Ws: Dict[str, torch.Tensor],
                keep: Dict[str, torch.Tensor], skip: Optional[str] = None) -> torch.Tensor:
        keep_use = dict(keep)
        if skip is not None:
            ref = next(iter(hs.values())) if hs else next(iter(Ws.values()))
            keep_use[skip] = torch.zeros(ref.size(0), dtype=torch.bool, device=ref.device)
        z0 = self.mean_z(hs, keep_use, skip=skip)
        if not self.use_transformer:
            return z0
        tokens, pad = self._stack(hs, Ws, keep_use)
        memory = self.encoder(tokens, src_key_padding_mask=pad)
        q = self.query.expand(tokens.size(0), -1, -1)
        attn_out, _ = self.attn(q, memory, memory, key_padding_mask=pad, need_weights=False)
        return self.ln(z0 + self.delta(attn_out.squeeze(1)))


class NMFTransformerMOCHI(nn.Module):
    def __init__(self, dims: Dict[str, int], tokenizers: Dict[str, FrozenNMF],
                 k=K_DEFAULT, d_model=D_MODEL, n_heads=4, n_layers=2,
                 use_nmf_tokens=True, use_transformer=True,
                 use_lowrank=True, gamma_init=0.3, gamma_nonneg=False,
                 w_head_act: str = "relu"):
        super().__init__()
        if w_head_act not in W_HEAD_ACTS:
            raise ValueError(f"w_head_act는 {W_HEAD_ACTS} 중 하나여야 합니다: {w_head_act}")
        self.mods = tuple(MODS)
        self.k = k
        self.d_model = d_model
        self.use_nmf_tokens = use_nmf_tokens
        self.use_transformer = use_transformer
        self.use_lowrank = use_lowrank
        self.gamma_nonneg = bool(gamma_nonneg)
        self.w_head_act = w_head_act
        self.encoders = nn.ModuleDict({m: nn.Linear(dims[m], HIDDEN[m]) for m in self.mods})
        self.decoders = nn.ModuleDict({m: nn.Linear(HIDDEN[m], dims[m]) for m in self.mods})
        self.tokenizers = nn.ModuleDict(tokenizers)
        self.to_h = nn.ModuleDict({m: nn.Linear(d_model, HIDDEN[m]) for m in self.mods})
        # 융합 좌표에서 타깃의 NMF 계수를 예측하는 머리. 저랭크 잔차 경로의 입력이 된다.
        # 가중은 0에서 출발하고 절편은 코호트 평균 계수가 활성화 뒤에 나오게 맞춰,
        # 학습 초기에 잔차 (Ŵ − W̄)H 가 정확히 0이다.
        self.w_head = nn.ModuleDict({m: nn.Linear(d_model, k) for m in self.mods})
        for m in self.mods:
            nn.init.zeros_(self.w_head[m].weight)
            with torch.no_grad():
                mean = self.tokenizers[m].w_mean
                if self.w_head_act == "softplus":
                    self.w_head[m].bias.copy_(_inv_softplus_tensor(mean))
                else:
                    self.w_head[m].bias.copy_(mean)
        # gamma는 저랭크 잔차의 게이트다. 파라미터 이름은 예전 체크포인트와 맞춘다.
        # gamma_nonneg=True면 이 값은 raw이고 실효 게이트는 softplus(raw) ≥ 0 이다.
        raw_init = float(gamma_init)
        if self.gamma_nonneg:
            if float(gamma_init) < 0.05:
                # softplus는 raw가 크게 음수인 구간에서 기울기가 0에 수렴한다.
                # gamma_init≈0으로 시작하면 raw≈-9가 되어 게이트가 그 자리에 갇힌다.
                # (실측: gamma_init=0, gamma_nonneg=True로 40 epoch 학습 시 0.000 고정)
                raise ValueError(
                    f"gamma_nonneg=True에서는 gamma_init >= 0.05 이어야 합니다 "
                    f"(받은 값 {gamma_init}). 저랭크 경로를 끄려면 use_lowrank=False를 쓰세요.")
            raw_init = _inv_softplus(float(gamma_init))
        self.gamma = nn.Parameter(torch.full((len(self.mods),), raw_init))
        self.fuse = NMFTransformer(
            k=k, d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            use_nmf_tokens=use_nmf_tokens, use_transformer=use_transformer,
        )

    def _gamma(self, target: str) -> torch.Tensor:
        g = self.gamma[self.mods.index(target)]
        return F.softplus(g) if self.gamma_nonneg else g

    def effective_gamma(self) -> torch.Tensor:
        """로그·보고용 실효 게이트 값. gamma_nonneg 여부와 무관하게 같은 의미."""
        with torch.no_grad():
            return F.softplus(self.gamma) if self.gamma_nonneg else self.gamma.clone()

    def set_lowrank(self, on: bool):
        """저랭크 NMF 잔차 경로를 켜고 끈다.

        gamma를 0으로 덮어쓰는 방식은 gamma_nonneg일 때 softplus(0)=0.693이 되어
        녹아웃이 아니라 게이트를 키우는 결과가 된다. 녹아웃은 반드시 이 함수를 쓴다.
        """
        self.use_lowrank = bool(on)
        return self

    def predict_W(self, z: torch.Tensor, target: str) -> torch.Tensor:
        """융합 좌표에서 타깃 NMF 계수. 계수는 비음수여야 한다.

        relu는 성분별로 정확히 0을 낼 수 있다. softplus는 같은 비음수를
        유지하되 정확한 영은 만들지 않는다. Hamming이 독립 귀무와 같았던
        서명이 ReLU 절단 때문인지를 이 선택으로 가른다.
        """
        raw = self.w_head[target](z)
        if self.w_head_act == "softplus":
            return F.softplus(raw)
        return F.relu(raw)

    def decode_target(self, target: str, h: torch.Tensor,
                      W: Optional[torch.Tensor] = None) -> torch.Tensor:
        """디코더 출력에 저랭크 생물학 구조를 잔차로 더한다."""
        out = self.decoders[target](h)
        if self.use_lowrank and W is not None:
            out = out + self._gamma(target) * self.tokenizers[target].decode_dev(W)
        return out

    def encode_h(self, xs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {m: self.encoders[m](x) for m, x in xs.items()}

    def encode_W(self, xs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {m: self.tokenizers[m].encode(x) for m, x in xs.items()}

    def reconstruct_own(self, xs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = {}
        for m, x in xs.items():
            W = self.tokenizers[m].encode(x) if self.use_lowrank else None
            out[m] = self.decode_target(m, self.encoders[m](x), W)
        return out

    def fused_for(self, xs: Dict[str, torch.Tensor], present: Dict[str, torch.Tensor],
                  target: str) -> torch.Tensor:
        xs_loo = {m: x for m, x in xs.items() if m != target}
        hs = self.encode_h(xs_loo)
        Ws = self.encode_W(xs_loo)
        keep = {m: p.clone() for m, p in present.items() if m != target}
        return self.fuse.fused_z(hs, Ws, keep, skip=target)

    def loo_hidden(self, xs: Dict[str, torch.Tensor], present: Dict[str, torch.Tensor],
                   target: str) -> torch.Tensor:
        return self.to_h[target](self.fused_for(xs, present, target))

    def loo_parts(self, xs: Dict[str, torch.Tensor], present: Dict[str, torch.Tensor],
                  target: str):
        """(은닉, 예측 NMF 계수). 보조 손실이 계수를 감독한다."""
        z = self.fused_for(xs, present, target)
        W = self.predict_W(z, target) if self.use_lowrank else None
        return self.to_h[target](z), W

    def loo_reconstruct(self, xs: Dict[str, torch.Tensor], present: Dict[str, torch.Tensor],
                        target: str) -> torch.Tensor:
        h, W = self.loo_parts(xs, present, target)
        return self.decode_target(target, h, W)

    def mixed_hidden(self, xs: Dict[str, torch.Tensor], present: Dict[str, torch.Tensor],
                     target: str, self_weight=10.0) -> torch.Tensor:
        h_own = self.encoders[target](xs[target])
        h_loo = self.loo_hidden(xs, present, target)
        sw = float(self_weight)
        return (sw * h_own + h_loo) / (sw + 1.0)

    def mixed_reconstruct(self, xs: Dict[str, torch.Tensor], present: Dict[str, torch.Tensor],
                          target: str, self_weight=10.0) -> torch.Tensor:
        """칸 결측: 자기 좌표를 지배적으로 신뢰하되 계수도 같은 비율로 섞는다."""
        sw = float(self_weight)
        h_own = self.encoders[target](xs[target])
        h_loo, W_loo = self.loo_parts(xs, present, target)
        h = (sw * h_own + h_loo) / (sw + 1.0)
        W = None
        if self.use_lowrank:
            W_own = self.tokenizers[target].encode(xs[target])
            W = (sw * W_own + W_loo) / (sw + 1.0)
        return self.decode_target(target, h, W)


@torch.no_grad()
def predict_nmf_tf(model: NMFTransformerMOCHI, tabs: Dict[str, np.ndarray], device,
                   missing: Optional[str] = None, batch_size=64,
                   self_weight=10.0) -> Dict[str, np.ndarray]:
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
            hat = {missing: model.loo_reconstruct(xs, present, missing)}
            for m in MODS:
                if m != missing:
                    hat[m] = model.reconstruct_own({m: xs[m]})[m]
        else:
            hat = {}
            for tgt in MODS:
                hat[tgt] = model.mixed_reconstruct(xs, present, tgt, self_weight=self_weight)
        for m in MODS:
            acc[m].append(hat[m].cpu().numpy())
    return {m: np.concatenate(acc[m], 0) for m in MODS}


def load_nmf_tf(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    dims, k = ck["dims"], ck.get("k", K_DEFAULT)
    tokenizers = {}
    sd = ck["model"]
    for m in MODS:
        prefix = f"tokenizers.{m}."
        H = sd[prefix + "H"]
        shift = sd[prefix + "shift"]
        inv = sd[prefix + "HHt_inv"]
        tokenizers[m] = FrozenNMF(H, shift, inv, sd.get(prefix + "w_mean"))
    model = NMFTransformerMOCHI(
        dims, tokenizers, k=k, d_model=ck.get("d_model", D_MODEL),
        n_heads=ck.get("n_heads", 4), n_layers=ck.get("n_layers", 2),
        use_nmf_tokens=ck.get("use_nmf_tokens", True),
        use_transformer=ck.get("use_transformer", True),
        use_lowrank=ck.get("use_lowrank", "gamma" in sd),
        # 예전 체크포인트에는 이 키가 없다. 없으면 무제약 gamma·ReLU 머리로 읽어 재현된다.
        gamma_nonneg=ck.get("gamma_nonneg", False),
        w_head_act=ck.get("w_head_act", "relu"),
    ).to(device)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if unexpected:
        raise RuntimeError(f"예상 밖 가중치: {unexpected}")
    allowed = ("w_head.", "gamma")
    stale = [k for k in missing if not (k.startswith(allowed) or k.endswith("w_mean"))]
    if stale:
        raise RuntimeError(f"빠진 가중치: {stale}")
    model.eval()
    return model
