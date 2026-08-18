# models_v6.py
# Ridge 잔차 + 오믹스별 인코더 + 타깃별 게이트 + 조건부 critic.
#
# 단백질은 메틸화가 선형 성능을 깎는다. 핵은 RNA-only Ridge이고,
# 메틸 선형항은 샘플별 시그모이드 게이트로 0까지 내려갈 수 있다.
import torch
import torch.nn as nn
from torch import Tensor


class FrozenRidge(nn.Module):
    """sklearn Ridge.coef_ (n_out, n_in), intercept_ (n_out,) → ŷ = x @ W + b."""

    def __init__(self, coef, intercept):
        super().__init__()
        w = torch.tensor(coef.T, dtype=torch.float32)   # [n_in, n_out]
        b = torch.tensor(intercept, dtype=torch.float32)
        self.register_buffer("W", w)
        self.register_buffer("b", b)

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.W + self.b


class ModalityEncoder(nn.Module):
    def __init__(self, in_dim: int, d: int = 256, pdrop: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, d),
            nn.ReLU(),
            nn.Dropout(pdrop),
            nn.Linear(d, d),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class GatedResidualHead(nn.Module):
    """두 소스 임베딩을 타깃별 쿼리로 게이트한 뒤, 0으로 시작하는 잔차를 낸다.

    score_bias: 두 소스 로짓에 더함. 단백질은 RNA를 선호하도록 [+, −]로 둔다.
    """

    def __init__(self, d: int, out_dim: int, pdrop: float = 0.2, score_bias=(0.0, 0.0)):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(d))
        self.score_bias = nn.Parameter(torch.tensor(score_bias, dtype=torch.float32))
        self.fc1 = nn.Linear(d, d)
        self.fc2 = nn.Linear(d, out_dim)
        self.drop = nn.Dropout(pdrop)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, z_a: Tensor, z_b: Tensor):
        sa = (z_a * self.query).sum(dim=-1, keepdim=True)
        sb = (z_b * self.query).sum(dim=-1, keepdim=True)
        logits = torch.cat([sa, sb], dim=1) + self.score_bias.view(1, 2)
        alpha = torch.softmax(logits, dim=1)  # [B, 2]
        z = alpha[:, 0:1] * z_a + alpha[:, 1:2] * z_b
        res = self.fc2(self.drop(torch.relu(self.fc1(z))))
        return res, alpha


class MethDropGate(nn.Module):
    """메틸 선형항 게이트. bias=-4 → σ≈0.018 이라 시작은 거의 끈 상태."""

    def __init__(self, d: int, bias_init: float = -4.0):
        super().__init__()
        self.proj = nn.Linear(d, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.constant_(self.proj.bias, bias_init)

    def forward(self, z_m: Tensor) -> Tensor:
        return torch.sigmoid(self.proj(z_m))  # [B, 1]


class MOCHIGated(nn.Module):
    """
    단백질: Ŷ_p = Ridge(RNA) + g_m · Ridge(meth) + G_p(α_r E_r + α_m E_m)
    RNA/메틸: Ŷ = Ridge_2→1(X_obs) + G(게이트 임베딩)
    g_m ∈ (0,1) 이라 메틸을 완전히 버릴 수 있다.
    """

    def __init__(self, dim_r: int, dim_p: int, dim_m: int, d: int = 256):
        super().__init__()
        self.enc_r = ModalityEncoder(dim_r, d)
        self.enc_p = ModalityEncoder(dim_p, d)
        self.enc_m = ModalityEncoder(dim_m, d)
        self.head_p = GatedResidualHead(d, dim_p, score_bias=(1.0, -1.0))  # RNA > meth
        self.head_r = GatedResidualHead(d, dim_r)
        self.head_m = GatedResidualHead(d, dim_m)
        self.meth_drop = MethDropGate(d)
        self.last_alpha = {}
        self.last_meth_gate = None

    def encode(self, rna, prot, methy):
        return self.enc_r(rna), self.enc_p(prot), self.enc_m(methy)

    def residuals(self, rna, prot, methy):
        zr, zp, zm = self.encode(rna, prot, methy)
        g_m = self.meth_drop(zm)
        zm_p = g_m * zm  # 단백질 잔차에도 메틸을 같이 끔
        res_p, a_p = self.head_p(zr, zm_p)
        res_r, a_r = self.head_r(zp, zm)
        res_m, a_m = self.head_m(zr, zp)
        self.last_meth_gate = g_m
        self.last_alpha = {
            "protein": a_p.detach(),  # [:,0]=RNA, [:,1]=meth
            "rna": a_r.detach(),      # [:,0]=prot, [:,1]=meth
            "methyl": a_m.detach(),   # [:,0]=RNA, [:,1]=prot
        }
        return res_p, res_r, res_m, g_m
