# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional


class CrossAttentionBlock(nn.Module):
    """
    진짜 cross-attention 지원 블록.
    - src: [B, d_in_src]  → m개의 가상 토큰으로 투영 (메모리)
    - tgt: [B, d_in_tgt]  → 1개의 쿼리로 투영 (질의)
    - 멀티헤드 어텐션 후 FFN + GroupNorm 잔차
    """
    def __init__(
        self,
        d_in_src: int,
        d_in_tgt: int,
        d_model: int = 256,
        ff: int = 512,
        pdrop: float = 0.1,
        n_src_tokens: int = 8,
        n_heads: int = 4 , #헤드 수 조절
        d_head: int = 64,
    ):
        super().__init__()
        # 설정
        self.n_src_tokens = n_src_tokens
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_qkv = n_heads * d_head  # computed from heads × head_dim

        # 투영
        self.src_proj = nn.Linear(d_in_src, n_src_tokens * d_model, bias=False)  # [B, m*d_model]
        self.tgt_proj = nn.Linear(d_in_tgt, d_model, bias=False)                 # [B, d_model]

        # Q, K, V, O
        self.q = nn.Linear(d_model, self.d_qkv, bias=False)
        self.k = nn.Linear(d_model, self.d_qkv, bias=False)
        self.v = nn.Linear(d_model, self.d_qkv, bias=False)
        self.o = nn.Linear(self.d_qkv, d_model, bias=False)

        # FFN
        self.ff1 = nn.Linear(d_model, ff)
        self.ff2 = nn.Linear(ff, d_model)

        # 정규화/드롭아웃 (요청에 따라 GroupNorm 유지)
        self.ln1 = nn.GroupNorm(8, d_model)
        self.ln2 = nn.GroupNorm(8, d_model)
        self.drop = nn.Dropout(pdrop)

    def forward(self, src: Tensor, tgt: Tensor):
        """
        src, tgt: [B, d_in_*]
        return:
            y: [B, d_model]
            attn: [B, n_heads, 1, m] (해석용 가중치)
        """
        B, m = src.size(0), self.n_src_tokens
        S = self.src_proj(src).view(B, m, -1)     # [B, m, d_model]
        T = self.tgt_proj(tgt).unsqueeze(1)       # [B, 1, d_model] (single query)

        # 멀티헤드 분해
        q = self.q(T).view(B, 1, self.n_heads, self.d_head).transpose(1, 2)  # [B, h, 1, d]
        k = self.k(S).view(B, m, self.n_heads, self.d_head).transpose(1, 2)  # [B, h, m, d]
        v = self.v(S).view(B, m, self.n_heads, self.d_head).transpose(1, 2)  # [B, h, m, d]

        # 어텐션 (수치안정: max-shift)
        scores = (q @ k.transpose(-2, -1)) / (self.d_head ** 0.5)             # [B, h, 1, m]
        scores = scores - scores.max(dim=-1, keepdim=True).values
        attn = torch.softmax(scores, dim=-1)                                   # [B, h, 1, m]
        attn_out = (attn @ v).transpose(1, 2).contiguous().view(B, 1, -1)     # [B, 1, h*d]
        attn_out = self.o(attn_out).squeeze(1)                                 # [B, d_model]

        # 잔차 + FFN
        h = self.ln1(T.squeeze(1) + self.drop(attn_out))                       # [B, d_model]
        f = self.ff2(F.relu(self.ff1(h)))
        y = self.ln2(h + self.drop(f))                                         # [B, d_model]

        return y, attn


class Generator(nn.Module):
    """
    소스 → 타깃 모달리티 생성기 (WGAN-GP에서 G에 해당).
    - self-attention: model(xs, src=None)  => 내부에서 src=xs 처리
    - cross-attention: model(xs, src=src_tensor)  => 서로 다른 차원 지원(src_size 지정)
    - 타깃 도메인별 출력 규칙: methyl(σ 0~1), rna(softplus ≥0), protein(제한 없음)
    """
    def __init__(
        self,
        input_size: int,                 # 타깃 입력 차원(MLP 인풋, 보통 소스와 동일로 사용)
        output_size: int,                # 타깃 출력 차원
        hidden_dim = [1024, 512],
        use_attn: bool = False,
        nonneg_output: bool = True,
        n_src_tokens: int = 8,
        n_heads: int = 4,
        d_head: int = 64,
        target_type: str = "rna",        # "methyl" | "rna" | "protein"
        src_size: int = None,            # cross 소스 차원 (None이면 input_size)
    ):
        super().__init__()
        self.use_attn = use_attn
        self.n_heads = n_heads
        self.d_head = d_head
        self.target_type = target_type
        self.src_size = src_size or input_size

        # target_type 우선: protein이면 음수 허용을 기본으로
        self.nonneg_output = (False if target_type == "protein" else nonneg_output)

        if use_attn:
            self.attn = CrossAttentionBlock(
                d_in_src=self.src_size,
                d_in_tgt=input_size,
                d_model=256,
                ff=512,
                pdrop=0.1,
                n_src_tokens=n_src_tokens,
                n_heads=n_heads,
                d_head=d_head,
            )
            # attn 출력(d_model) → 입력차원으로 투사
            self.attn_proj = nn.Linear(256, input_size, bias=False)
            # 게이트(0~1): sigmoid(-2.2) ≈ 0.10, feature-wise 게이트로 확장
            self.alpha = nn.Parameter(torch.full((input_size,), -2.2))
            # 초기: attn_proj를 0으로 해서 residual이 거의 identity
            nn.init.zeros_(self.attn_proj.weight)

        # MLP 본체
        self.layer1 = nn.Linear(input_size, hidden_dim[0])
        self.layer2 = nn.Linear(hidden_dim[0], hidden_dim[-1])
        self.layer3 = nn.Linear(hidden_dim[-1], output_size)

        # 정규화/활성/드롭아웃 (요청: GroupNorm)
        self.norm1 = nn.GroupNorm(8, hidden_dim[0])
        self.norm2 = nn.GroupNorm(8, hidden_dim[-1])
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.2)

    def forward(self, xs: Tensor, src: Optional[Tensor] = None) -> Tensor:
        """
        xs: [B, input_size]  (타깃 입력 또는 기본 입력)
        src: [B, src_size]   (None이면 self-attention으로 src=xs)
        """
        if self.use_attn:
            if src is None:
                src = xs  # self-attention
            xs_attn, attn_weights = self.attn(src, xs)     # [B, d_model], [B, h, 1, m]
            xs_attn = self.attn_proj(xs_attn)              # [B, input_size]
            gate = torch.sigmoid(self.alpha)               # [input_size]
            xs = xs + gate * xs_attn
            # 해석/시각화용 가중치(메모리 절약 위해 CPU 보관 권장)
            self.last_attn_weights = attn_weights.detach().to("cpu")
            del attn_weights

        xs = self.relu(self.norm1(self.layer1(xs)))
        xs = self.drop(xs)
        xs = self.relu(self.norm2(self.layer2(xs)))
        xs = self.drop(xs)

        out = self.layer3(xs)

        # 모달리티별 출력 범위 (사후 clip은 끄는 것이 권장)
        if self.target_type == "methyl":
            out = torch.sigmoid(out)          # 0~1
        elif self.target_type == "rna":
            out = F.softplus(out)             # ≥0
        elif self.target_type == "protein":
            if self.nonneg_output:            # 안전장치(보통 False)
                out = F.relu(out)
        else:
            if self.nonneg_output:
                out = F.relu(out)

        return out


class Critic(nn.Module):
    """
    WGAN의 판별자(critic). 시그모이드 없이 생 점수 반환.
    학습 루프에서 gradient penalty(WGAN-GP) 추가 권장.
    """
    def __init__(self, input_size: int, hidden_dim: int = 512):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.classifier = nn.Linear(hidden_dim // 2, 1)

        # 요청: GroupNorm 유지 (hidden_dim % 8 == 0 권장)
        self.norm1 = nn.GroupNorm(8, hidden_dim)
        self.norm2 = nn.GroupNorm(8, hidden_dim // 2)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.2)

    def forward(self, xs: Tensor) -> Tensor:
        xs = self.relu(self.norm1(self.layer1(xs)))
        xs = self.drop(xs)
        xs = self.relu(self.norm2(self.layer2(xs)))
        xs = self.drop(xs)
        return self.classifier(xs)  # [B, 1]
