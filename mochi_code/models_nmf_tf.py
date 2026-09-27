#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NMF-Transformer MOCHI.

재구성·LOO의 주 경로는 오믹스별 AE 히든.
NMF 계수 W는 Transformer 보조 토큰이고, add_residual이면 디코더에
저랭크 잔차 γ(Ŵ − W̄)H를 더한다. aux_w_only 학습은 잔차를 더하지 않고
W를 보조 손실로만 쓴다.
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


class EncMLP(nn.Module):
    """인코더: 선형 한 층 대신 LN+GELU 2단."""

    def __init__(self, d_in: int, d_h: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_h),
            nn.LayerNorm(d_h),
            nn.GELU(),
            nn.Linear(d_h, d_h),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DecMLP(nn.Module):
    """디코더: 히든 → 특징. 인코더와 대칭."""

    def __init__(self, d_h: int, d_out: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_h, d_h),
            nn.LayerNorm(d_h),
            nn.GELU(),
            nn.Linear(d_h, d_out),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


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
        # 모달리티별 전역 게이트(파라미터 3개). present인 모달리티에만 softmax로 정규화해
        # mean_z 기준점을 균등 평균 대신 신뢰도 가중 평균으로 만든다.
        self.mod_weights = nn.Parameter(torch.ones(n_mods))
        self.mod_idx = {m: i for i, m in enumerate(self.mods)}
        self.proj_h = nn.ModuleDict({m: nn.Linear(HIDDEN[m], d_model) for m in self.mods})
        self.comp_emb = nn.Parameter(torch.randn(n_mods, k, d_model) * 0.02)
        self.mod_emb = nn.Parameter(torch.randn(n_mods, 1, d_model) * 0.02)
        self.h_emb = nn.Parameter(torch.randn(n_mods, 1, d_model) * 0.02)
        self.w_in = nn.Linear(1, d_model)
        # 레거시 단일 query. 지금은 어떤 경로도 이걸 읽지 않는다(모든 fused_z
        # 호출이 target을 넘긴다). 남겨두는 이유는 둘이다. 옛 ckpt가
        # load_state_dict에서 unexpected로 걸리지 않게 하고, load_nmf_tf가
        # 이 값을 query_by_mod의 시드로 쓴다. 파라미터만 남겨서는 호환이 되지
        # 않는다 — 시드 복사가 실제 호환 장치다.
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        # 타깃 오믹스별 query. LOO 복원 타깃에 따라 다른 retrieval 관점을 준다.
        self.query_by_mod = nn.ParameterDict({
            m: nn.Parameter(torch.randn(1, 1, d_model) * 0.02) for m in self.mods
        })
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
        # 소프트맥스 게이트를 미리 계산해 스칼라 가중으로 사용한다.
        w_gate = torch.softmax(self.mod_weights, dim=0)
        for m, h in hs.items():
            if m == skip:
                continue
            present = keep[m].float().unsqueeze(-1)
            g = w_gate[self.mod_idx[m]]
            acc = acc + self.proj_h[m](h) * present * g
            wsum = wsum + present * g
        return acc / wsum.clamp_min(1e-8)

    def _stack(self, hs: Dict[str, torch.Tensor], Ws: Dict[str, torch.Tensor],
               keep: Dict[str, torch.Tensor],
               comp: Optional[Dict[str, torch.Tensor]] = None):
        """comp가 오면 스칼라 계수 토큰 대신 내용을 담은 성분 토큰을 쓴다.

        기존 토큰은 계수 스칼라 하나를 Linear(1,d)로 올리고 자유 임베딩을 더한 것이라,
        성분이 어떤 특징으로 이루어졌는지가 들어가지 않는다. comp[m]은 [B,k,d_model]로
        성분마다의 내용을 이미 담고 있다 (NMFTransformerMOCHI.component_tokens 참고).
        """
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
                if comp is not None:
                    wtok = comp[m] + self.mod_emb[i] if m in comp else \
                        ref.new_zeros(b, self.k, self.d_model)
                else:
                    W = Ws[m] if m in Ws else ref.new_zeros(b, self.k)
                    wtok = self.w_in(W.unsqueeze(-1)) + self.comp_emb[i] + self.mod_emb[i]
                toks.append(wtok)
                pads.append((~present).unsqueeze(1).expand(-1, wtok.size(1)))
        tokens = torch.cat(toks, dim=1)
        pad = torch.cat(pads, dim=1)
        empty = pad.all(dim=1)
        if empty.any():
            pad = pad.clone()
            pad[empty, 0] = False
        return tokens, pad

    def fused_z(self, hs: Dict[str, torch.Tensor], Ws: Dict[str, torch.Tensor],
                keep: Dict[str, torch.Tensor], skip: Optional[str] = None,
                target: Optional[str] = None,
                comp: Optional[Dict[str, torch.Tensor]] = None,
                use_attn: Optional[bool] = None) -> torch.Tensor:
        """use_attn=False면 이 호출만 평균 융합으로 돈다 (경로별 모듈 배치).

        측정: 블록 번역은 평균이 앞서고(0.831 대 0.860), 심한 결측의 칸은
        attention이 앞선다(MCAR 90%에서 0.914 대 0.927). 모듈을 각각 이긴 자리에 둔다.
        """
        keep_use = dict(keep)
        if skip is not None:
            ref = next(iter(hs.values())) if hs else next(iter(Ws.values()))
            keep_use[skip] = torch.zeros(ref.size(0), dtype=torch.bool, device=ref.device)
        z0 = self.mean_z(hs, keep_use, skip=skip)
        attn_on = self.use_transformer if use_attn is None else (self.use_transformer and use_attn)
        if not attn_on:
            return z0
        tokens, pad = self._stack(hs, Ws, keep_use, comp=comp)
        memory = self.encoder(tokens, src_key_padding_mask=pad)
        q_param = self.query_by_mod[target] if (target is not None and target in self.query_by_mod) else self.query
        q = q_param.expand(tokens.size(0), -1, -1)
        attn_out, _ = self.attn(q, memory, memory, key_padding_mask=pad, need_weights=False)
        return self.ln(z0 + self.delta(attn_out.squeeze(1)))


class NMFTransformerMOCHI(nn.Module):
    def __init__(self, dims: Dict[str, int], tokenizers: Dict[str, FrozenNMF],
                 k=K_DEFAULT, d_model=D_MODEL, n_heads=4, n_layers=2,
                 use_nmf_tokens=True, use_transformer=True,
                 use_lowrank=True, gamma_init=0.3, gamma_nonneg=False,
                 w_head_act: str = "relu", w_from_others=False,
                 freeze_protein_gamma=False, add_residual=True, mlp_ae=False,
                 split_latent=False, block_attn=True, cell_attn=True,
                 content_tokens=False, detach_w_head=False):
        super().__init__()
        if w_head_act not in W_HEAD_ACTS:
            raise ValueError(f"w_head_act는 {W_HEAD_ACTS} 중 하나여야 합니다: {w_head_act}")
        self.mods = tuple(MODS)
        self.k = k
        self.d_model = d_model
        self.use_nmf_tokens = use_nmf_tokens
        self.use_transformer = use_transformer
        self.use_lowrank = use_lowrank
        self.w_from_others = w_from_others
        self.freeze_protein_gamma = freeze_protein_gamma
        self.add_residual = bool(add_residual)
        self.mlp_ae = bool(mlp_ae)
        # --- 5안: 좌표 분리와 경로별 모듈 배치 ---
        # split_latent : 은닉을 [공유|전용]으로 쪼개고, 오믹스를 건너가는 융합에는
        #                공유 절반만 넣는다. 전용 절반은 자기 재구성과 칸 혼합에만 쓰인다.
        # block_attn / cell_attn : 블록 번역과 칸 보간에 각각 Transformer를 쓸지.
        #                기존 측정은 블록에 평균(0.831 대 0.860), 심한 결측의 칸에
        #                attention(0.914 대 0.927)이 앞섰다.
        # content_tokens : 성분 토큰을 계수 스칼라가 아니라 W[j]·H[j]의 인코딩으로 만든다.
        # detach_w_head : 계수 읽기가 표현을 바꾸지 못하게 기울기를 끊는다.
        self.split_latent = bool(split_latent)
        self.block_attn = bool(block_attn)
        self.cell_attn = bool(cell_attn)
        self.content_tokens = bool(content_tokens)
        self.detach_w_head = bool(detach_w_head)
        self.gamma_nonneg = bool(gamma_nonneg)
        self.w_head_act = w_head_act
        if mlp_ae:
            self.encoders = nn.ModuleDict({m: EncMLP(dims[m], HIDDEN[m]) for m in self.mods})
            self.decoders = nn.ModuleDict({m: DecMLP(HIDDEN[m], dims[m]) for m in self.mods})
        else:
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
        self.w_from = nn.ModuleDict()
        if w_from_others:
            for tgt in self.mods:
                layers = nn.ModuleDict()
                for src in self.mods:
                    if src == tgt:
                        continue
                    layer = nn.Linear(HIDDEN[src], k, bias=False)
                    nn.init.zeros_(layer.weight)
                    layers[src] = layer
                self.w_from[tgt] = layers
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
        if not self.add_residual:
            self.gamma.requires_grad_(False)
        if freeze_protein_gamma:
            with torch.no_grad():
                self.gamma[self.mods.index("protein")] = 0.0
        self.fuse = NMFTransformer(
            k=k, d_model=d_model, n_heads=n_heads, n_layers=n_layers,
            use_nmf_tokens=use_nmf_tokens, use_transformer=use_transformer,
        )

    def _gamma(self, target: str) -> torch.Tensor:
        if self.freeze_protein_gamma and target == "protein":
            return self.gamma.new_zeros(())
        g = self.gamma[self.mods.index(target)]
        return F.softplus(g) if self.gamma_nonneg else g

    def effective_gamma(self) -> torch.Tensor:
        """로그·보고용 실효 게이트 값. gamma_nonneg 여부와 무관하게 같은 의미."""
        with torch.no_grad():
            return F.softplus(self.gamma) if self.gamma_nonneg else self.gamma.clone()

    def gamma_log(self) -> str:
        if not (self.use_lowrank and self.add_residual):
            return "unused"
        return ",".join(f"{float(g):.3f}" for g in self.effective_gamma().cpu())

    def set_lowrank(self, on: bool):
        """저랭크 NMF 잔차 경로를 켜고 끈다.

        gamma를 0으로 덮어쓰는 방식은 gamma_nonneg일 때 softplus(0)=0.693이 되어
        녹아웃이 아니라 게이트를 키우는 결과가 된다. 녹아웃은 반드시 이 함수를 쓴다.
        """
        self.use_lowrank = bool(on)
        if not on:
            self.add_residual = False
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

    def predict_W_others(self, hs: Dict[str, torch.Tensor], target: str,
                         keep: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        """관측된 다른 오믹스 은닉에서 타깃 NMF 계수를 낸다."""
        ref = next(iter(hs.values()))
        mean = self.tokenizers[target].w_mean
        if self.w_head_act == "softplus":
            raw = _inv_softplus_tensor(mean).unsqueeze(0).expand(ref.size(0), -1).clone()
        else:
            raw = mean.unsqueeze(0).expand(ref.size(0), -1).clone()
        for src, layer in self.w_from[target].items():
            if src not in hs:
                continue
            h = hs[src]
            if keep is not None and src in keep:
                h = h * keep[src].float().unsqueeze(-1)
            raw = raw + layer(h)
        if self.w_head_act == "softplus":
            return F.softplus(raw)
        return F.relu(raw)

    def decode_target(self, target: str, h: torch.Tensor,
                      W: Optional[torch.Tensor] = None) -> torch.Tensor:
        """디코더 출력. add_residual이면 저랭크 편차를 더한다."""
        out = self.decoders[target](h)
        if self.use_lowrank and self.add_residual and W is not None:
            out = out + self._gamma(target) * self.tokenizers[target].decode_dev(W)
        return out

    def encode_h(self, xs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {m: self.encoders[m](x) for m, x in xs.items()}

    def mask_specific(self, h: torch.Tensor) -> torch.Tensor:
        """오믹스를 건너가는 경로에 넣을 은닉. 전용 절반을 그래프 밖 0으로 바꾼다.

        자르기만 해서는 분리가 생기지 않는다. 전용 절반이 블록 손실의 기울기를
        받지 못하게 여기서 끊어야, 인코더가 교차 신호를 전용 쪽에 숨기지 못한다.
        """
        if not self.split_latent:
            return h
        d = h.size(-1) // 2
        shared, private = h.split([d, h.size(-1) - d], dim=-1)
        return torch.cat([shared, shared.new_zeros(private.shape)], dim=-1)

    def component_tokens(self, xs: Dict[str, torch.Tensor],
                         Ws: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """성분마다 [B,k,d_model] 토큰. 내용은 W[j]·H[j]를 기존 인코더에 통과시킨 것.

        이 환자 프로필 중 성분 j가 설명하는 부분을 인코더의 좌표계로 옮긴 것이므로,
        계수 스칼라 하나를 올리던 예전 토큰과 달리 성분의 정체가 들어간다.
        기존 인코더를 재사용하므로 새 파라미터가 없다.

        성분마다 따로 두는 것이 핵심이다. 합치면 Σⱼ W[j]H[j] ≈ x + shift 이므로
        선형 인코더에서 h 토큰에 상수를 더한 것이 되어 정보가 사라진다.
        """
        out = {}
        for m, W in Ws.items():
            if m not in xs:
                continue
            H = self.tokenizers[m].H                      # [k, d_in]
            b, k = W.shape
            parts = W.unsqueeze(-1) * H.unsqueeze(0)      # [B, k, d_in]
            h = self.encoders[m](parts.reshape(b * k, -1))
            h = self.mask_specific(h)
            out[m] = self.fuse.proj_h[m](h).reshape(b, k, -1)
        return out

    def encode_W(self, xs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return {m: self.tokenizers[m].encode(x) for m, x in xs.items()}

    def reconstruct_own(self, xs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = {}
        for m, x in xs.items():
            W = self.tokenizers[m].encode(x) if self.use_lowrank else None
            out[m] = self.decode_target(m, self.encoders[m](x), W)
        return out

    def _fuse_inputs(self, xs: Dict[str, torch.Tensor], present: Dict[str, torch.Tensor],
                     target: str):
        xs_loo = {m: x for m, x in xs.items() if m != target}
        hs = {m: self.mask_specific(h) for m, h in self.encode_h(xs_loo).items()}
        Ws = self.encode_W(xs_loo)
        keep = {m: p.clone() for m, p in present.items() if m != target}
        comp = self.component_tokens(xs_loo, Ws) if self.content_tokens else None
        return hs, Ws, keep, comp

    def fused_for(self, xs: Dict[str, torch.Tensor], present: Dict[str, torch.Tensor],
                  target: str, path: str = "block") -> torch.Tensor:
        hs, Ws, keep, comp = self._fuse_inputs(xs, present, target)
        return self.fuse.fused_z(hs, Ws, keep, skip=target, target=target, comp=comp,
                                 use_attn=(self.cell_attn if path == "cell" else self.block_attn))

    def loo_hidden(self, xs: Dict[str, torch.Tensor], present: Dict[str, torch.Tensor],
                   target: str, path: str = "block") -> torch.Tensor:
        return self.to_h[target](self.fused_for(xs, present, target, path=path))

    def loo_parts(self, xs: Dict[str, torch.Tensor], present: Dict[str, torch.Tensor],
                  target: str, path: str = "block"):
        """(은닉, 예측 NMF 계수). 보조 손실이 계수를 감독한다."""
        hs, Ws, keep, comp = self._fuse_inputs(xs, present, target)
        z = self.fuse.fused_z(hs, Ws, keep, skip=target, target=target, comp=comp,
                              use_attn=(self.cell_attn if path == "cell" else self.block_attn))
        W = None
        if self.use_lowrank:
            if self.w_from_others:
                hw = {m: h.detach() for m, h in hs.items()} if self.detach_w_head else hs
                W = self.predict_W_others(hw, target, keep=keep)
            else:
                W = self.predict_W(z.detach() if self.detach_w_head else z, target)
        return self.to_h[target](z), W

    def loo_reconstruct(self, xs: Dict[str, torch.Tensor], present: Dict[str, torch.Tensor],
                        target: str) -> torch.Tensor:
        h, W = self.loo_parts(xs, present, target, path="block")
        return self.decode_target(target, h, W)

    def mixed_hidden(self, xs: Dict[str, torch.Tensor], present: Dict[str, torch.Tensor],
                     target: str, self_weight=10.0) -> torch.Tensor:
        h_own = self.encoders[target](xs[target])
        h_loo = self.loo_hidden(xs, present, target, path="cell")
        sw = float(self_weight)
        return (sw * h_own + h_loo) / (sw + 1.0)

    def mixed_reconstruct(self, xs: Dict[str, torch.Tensor], present: Dict[str, torch.Tensor],
                          target: str, self_weight=10.0) -> torch.Tensor:
        """칸 결측: 자기 좌표를 지배적으로 신뢰하되 계수도 같은 비율로 섞는다."""
        sw = float(self_weight)
        h_own = self.encoders[target](xs[target])
        h_loo, W_loo = self.loo_parts(xs, present, target, path="cell")
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


def _ckpt_arch(ck, sd):
    """플래그와 가중치 모양이 맞는지 검사한다. 침묵 기본값으로 활성함수를 바꾸지 않는다."""
    has_mlp_w = any(k.startswith("encoders.") and ".net." in k for k in sd)
    has_w_from = any(k.startswith("w_from.") for k in sd)
    era_aux = any(k in ck for k in ("mlp_ae", "add_residual", "w_from_others"))

    if "mlp_ae" in ck:
        mlp_ae = bool(ck["mlp_ae"])
        if mlp_ae != has_mlp_w:
            raise RuntimeError(f"mlp_ae={mlp_ae} 인데 인코더 MLP 가중치={has_mlp_w}")
    elif has_mlp_w:
        raise RuntimeError("인코더가 MLP 가중치인데 mlp_ae 플래그가 없다")
    else:
        mlp_ae = False

    if "w_from_others" in ck:
        w_from_others = bool(ck["w_from_others"])
        if w_from_others != has_w_from:
            raise RuntimeError(f"w_from_others={w_from_others} 인데 w_from 가중치={has_w_from}")
    else:
        w_from_others = has_w_from

    if "add_residual" in ck:
        add_residual = bool(ck["add_residual"])
    else:
        add_residual = True

    if "w_head_act" in ck:
        w_head_act = ck["w_head_act"]
        if w_head_act not in W_HEAD_ACTS:
            raise RuntimeError(f"알 수 없는 w_head_act={w_head_act}")
    elif era_aux:
        w_head_act = "softplus"
        print("경고: w_head_act 없음 → softplus (aux_w/wide 계보)")
    else:
        w_head_act = "relu"

    return dict(
        mlp_ae=mlp_ae, w_from_others=w_from_others, add_residual=add_residual,
        w_head_act=w_head_act,
        gamma_nonneg=bool(ck.get("gamma_nonneg", False)),
        freeze_protein_gamma=bool(ck.get("freeze_protein_gamma", False)),
        use_lowrank=ck.get("use_lowrank", "gamma" in sd),
    )


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
    arch = _ckpt_arch(ck, sd)
    model = NMFTransformerMOCHI(
        dims, tokenizers, k=k, d_model=ck.get("d_model", D_MODEL),
        n_heads=ck.get("n_heads", 4), n_layers=ck.get("n_layers", 2),
        use_nmf_tokens=ck.get("use_nmf_tokens", True),
        use_transformer=ck.get("use_transformer", True),
        use_lowrank=arch["use_lowrank"],
        gamma_nonneg=arch["gamma_nonneg"],
        w_head_act=arch["w_head_act"],
        w_from_others=arch["w_from_others"],
        freeze_protein_gamma=arch["freeze_protein_gamma"],
        add_residual=arch["add_residual"],
        mlp_ae=arch["mlp_ae"],
        split_latent=ck.get("split_latent", False),
        block_attn=ck.get("block_attn", True),
        cell_attn=ck.get("cell_attn", True),
        content_tokens=ck.get("content_tokens", False),
        detach_w_head=ck.get("detach_w_head", False),
    ).to(device)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if unexpected:
        raise RuntimeError(f"예상 밖 가중치: {unexpected}")
    # 타깃별 query가 없는 ckpt는 단일 query로 학습된 모델이다. 그대로 두면
    # query_by_mod가 randn 초기값으로 남아 LOO retrieval이 조용히 달라진다
    # (실측: 융합 출력 max|diff| 8.6e-2). 학습된 query를 모든 타깃에 복사해
    # 옛 동작을 정확히 재현한다. mod_weights는 ones 초기값의 softmax가 균등이라
    # 옛 균등 평균과 이미 같으므로 손대지 않는다.
    if "fuse.query" in sd and any(k.startswith("fuse.query_by_mod.") for k in missing):
        with torch.no_grad():
            for mod in model.fuse.query_by_mod:
                model.fuse.query_by_mod[mod].copy_(sd["fuse.query"])
        print("옛 ckpt: 단일 query를 타깃별 query로 복사했다")
    allowed = ("w_head.", "gamma", "w_from.", "fuse.mod_weights", "fuse.query_by_mod.")
    stale = [k for k in missing if not (k.startswith(allowed) or k.endswith("w_mean"))]
    if stale:
        raise RuntimeError(f"빠진 가중치: {stale}")
    model.eval()
    return model
