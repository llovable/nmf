# models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils import spectral_norm
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


class ModalityCrossAttention(nn.Module):
    """
    소스 오믹스를 모달리티별로 토큰화한 뒤, 학습된 query가 attend.
    concat 벡터를 임의로 쪼개는 방식이 아니라 모달리티 경계가 있는 교차-어텐션.
    어텐션 가중치 shape: [B, heads, 1, n_modalities * n_tokens]
    """
    def __init__(
        self,
        src_dims,
        d_model: int = 256,
        n_tokens: int = 4,
        n_heads: int = 4,
        d_head: int = 64,
        pdrop: float = 0.1,
        ff: int = 512,
    ):
        super().__init__()
        self.src_dims = tuple(int(d) for d in src_dims)
        self.n_tokens = n_tokens
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_model = d_model
        self.d_qkv = n_heads * d_head
        self.mod_projs = nn.ModuleList()
        for d in self.src_dims:
            self.mod_projs.append(nn.Sequential(
                nn.Linear(d, d_model),
                nn.ReLU(),
                nn.Linear(d_model, n_tokens * d_model),
            ))
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.q = nn.Linear(d_model, self.d_qkv, bias=False)
        self.k = nn.Linear(d_model, self.d_qkv, bias=False)
        self.v = nn.Linear(d_model, self.d_qkv, bias=False)
        self.o = nn.Linear(self.d_qkv, d_model, bias=False)
        self.ff1 = nn.Linear(d_model, ff)
        self.ff2 = nn.Linear(ff, d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(pdrop)

    def forward(self, xs: Tensor):
        B = xs.size(0)
        chunks = torch.split(xs, self.src_dims, dim=1)
        toks = [proj(ch).view(B, self.n_tokens, self.d_model) for proj, ch in zip(self.mod_projs, chunks)]
        S = torch.cat(toks, dim=1)                          # [B, M*n_tok, d]
        m = S.size(1)
        T = self.query.expand(B, -1, -1)                    # [B, 1, d]
        q = self.q(T).view(B, 1, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k(S).view(B, m, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v(S).view(B, m, self.n_heads, self.d_head).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / (self.d_head ** 0.5)
        scores = scores - scores.max(dim=-1, keepdim=True).values
        attn = torch.softmax(scores, dim=-1)                # [B, h, 1, M*n_tok]
        attn_out = (attn @ v).transpose(1, 2).contiguous().view(B, 1, -1)
        attn_out = self.o(attn_out).squeeze(1)
        h = self.ln1(T.squeeze(1) + self.drop(attn_out))
        f = self.ff2(F.relu(self.ff1(h)))
        y = self.ln2(h + self.drop(f))
        return y, attn


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Generator(nn.Module):
    """
    소스 → 타깃 모달리티 생성기 (WGAN-GP에서 G에 해당).
    - self-attention: model(xs, src=None)  => 내부에서 src=xs 처리
    - cross-attention: model(xs, src=src_tensor)  => 서로 다른 차원 지원(src_size 지정)
    - 타깃 도메인별 출력 규칙: methyl(σ 0~1), rna(softplus ≥0), protein(제한 없음)
    - lightweight=True: 고차원 입력을 bottleneck으로 줄인 뒤 어텐션/MLP (파라미터 폭증 방지)
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
        bottleneck_dim: Optional[int] = None,
        lightweight: bool = False,
        src_dims=None,                   # 예: (rna_dim, meth_dim) — 모달리티 교차-어텐션
        apply_output_range: bool = True, # z-score 학습 시 False
        n_mod_tokens: int = 4,
        use_linear_skip: bool = False,   # ŷ = low-rank linear(x) + residual
        skip_rank: int = 64,
    ):
        super().__init__()
        self.use_attn = use_attn
        self.n_heads = n_heads
        self.d_head = d_head
        self.target_type = target_type
        self.src_size = src_size or input_size
        self.input_size = input_size
        self.src_dims = tuple(src_dims) if src_dims is not None else None
        self.apply_output_range = apply_output_range
        self.use_mod_attn = bool(use_attn and self.src_dims is not None)
        self.use_linear_skip = bool(use_linear_skip)

        if lightweight:
            hidden_dim = hidden_dim if hidden_dim != [1024, 512] else [512, 256]
            if bottleneck_dim is None:
                bottleneck_dim = 256

        self.bottleneck_dim = bottleneck_dim
        # 모달리티 어텐션이 인코더일 때는 concat bottleneck을 쓰지 않음 (섞인 뒤에 attend하지 않음)
        self.use_bottleneck = (bottleneck_dim is not None) and (not self.use_mod_attn)

        # target_type 우선: protein이면 음수 허용을 기본으로
        self.nonneg_output = (False if target_type == "protein" else nonneg_output)

        if self.use_mod_attn:
            self.mod_attn = ModalityCrossAttention(
                src_dims=self.src_dims,
                d_model=256,
                n_tokens=n_mod_tokens,
                n_heads=n_heads,
                d_head=d_head,
            )
            self.xs_in = None
            self.src_in = None
            mlp_in = 256
        elif self.use_bottleneck:
            b = bottleneck_dim
            self.xs_in = nn.Linear(input_size, b)
            if self.src_size != input_size:
                self.src_in = nn.Linear(self.src_size, b)
            else:
                self.src_in = None
            attn_src = b
            attn_tgt = b
            mlp_in = b
        else:
            self.xs_in = None
            self.src_in = None
            attn_src = self.src_size
            attn_tgt = input_size
            mlp_in = input_size

        if use_attn and not self.use_mod_attn:
            self.attn = CrossAttentionBlock(
                d_in_src=attn_src,
                d_in_tgt=attn_tgt,
                d_model=256,
                ff=512,
                pdrop=0.1,
                n_src_tokens=n_src_tokens,
                n_heads=n_heads,
                d_head=d_head,
            )
            # 구버전 경로: 게이트 잔차 (어텐션이 꺼지기 쉬움)
            self.attn_proj = nn.Linear(256, mlp_in, bias=False)
            self.alpha = nn.Parameter(torch.full((mlp_in,), -2.2))
            nn.init.zeros_(self.attn_proj.weight)

        # MLP 본체
        self.layer1 = nn.Linear(mlp_in, hidden_dim[0])
        self.layer2 = nn.Linear(hidden_dim[0], hidden_dim[-1])
        self.layer3 = nn.Linear(hidden_dim[-1], output_size)
        if self.use_linear_skip:
            # 저랭크 선형 경로: 소표본에서 평균 회귀를 막고, 비선형은 잔차만 학습
            self.skip_u = nn.Linear(input_size, skip_rank, bias=False)
            self.skip_v = nn.Linear(skip_rank, output_size, bias=True)
            nn.init.zeros_(self.layer3.weight)
            nn.init.zeros_(self.layer3.bias)

        # 정규화/활성/드롭아웃 (요청: GroupNorm)
        self.norm1 = nn.GroupNorm(8, hidden_dim[0])
        self.norm2 = nn.GroupNorm(8, hidden_dim[-1])
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.2)

    def _project_inputs(self, xs: Tensor, src: Optional[Tensor]):
        if not self.use_bottleneck:
            return xs, src
        h = self.relu(self.xs_in(xs))
        if src is None:
            return h, None
        if self.src_in is None:
            src_h = self.relu(self.xs_in(src))
        else:
            src_h = self.relu(self.src_in(src))
        return h, src_h

    def forward(self, xs: Tensor, src: Optional[Tensor] = None) -> Tensor:
        """
        xs: [B, input_size]  (타깃 입력 또는 기본 입력)
        src: [B, src_size]   (None이면 self-attention으로 src=xs)
        """
        raw = xs
        if self.use_mod_attn:
            h, attn_weights = self.mod_attn(xs)
            self.last_attn_weights = attn_weights.detach().to("cpu")
            xs = h
        else:
            xs, src = self._project_inputs(xs, src)
            if self.use_attn:
                if src is None:
                    src = xs
                xs_attn, attn_weights = self.attn(src, xs)
                xs_attn = self.attn_proj(xs_attn)
                gate = torch.sigmoid(self.alpha)
                xs = xs + gate * xs_attn
                self.last_attn_weights = attn_weights.detach().to("cpu")
                del attn_weights

        xs = self.relu(self.norm1(self.layer1(xs)))
        xs = self.drop(xs)
        xs = self.relu(self.norm2(self.layer2(xs)))
        xs = self.drop(xs)

        out = self.layer3(xs)
        if self.use_linear_skip:
            out = out + self.skip_v(self.skip_u(raw))

        if self.apply_output_range:
            if self.target_type == "methyl":
                out = torch.sigmoid(out)
            elif self.target_type == "rna":
                out = F.softplus(out)
            elif self.target_type == "protein":
                if self.nonneg_output:
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
    def __init__(self, input_size: int, hidden_dim: int = 512, use_spectral_norm: bool = False):
        super().__init__()
        lin = spectral_norm if use_spectral_norm else (lambda m: m)
        self.layer1 = lin(nn.Linear(input_size, hidden_dim))
        self.layer2 = lin(nn.Linear(hidden_dim, hidden_dim // 2))
        self.classifier = lin(nn.Linear(hidden_dim // 2, 1))

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


class ConditionalCritic(nn.Module):
    """
    조건부 WGAN critic: D(target | source).
    무조건부 critic은 '단백질처럼 보이는지'만 보고, 소스와의 대응은 보지 않는다.
    pix2pix (Isola et al.), OmiImp, ImpuGAN과 같은 조건부 판별.
    gradient penalty는 target 쪽에만 건다 (조건은 고정).
    """
    def __init__(self, target_size: int, cond_size: int, hidden_dim: int = 512,
                 embed_dim: int = 256, use_spectral_norm: bool = True):
        super().__init__()
        lin = spectral_norm if use_spectral_norm else (lambda m: m)
        self.tgt_in = lin(nn.Linear(target_size, embed_dim))
        self.cond_in = lin(nn.Linear(cond_size, embed_dim))
        self.layer1 = lin(nn.Linear(embed_dim * 2, hidden_dim))
        self.layer2 = lin(nn.Linear(hidden_dim, hidden_dim // 2))
        self.classifier = lin(nn.Linear(hidden_dim // 2, 1))
        self.norm1 = nn.GroupNorm(8, hidden_dim)
        self.norm2 = nn.GroupNorm(8, hidden_dim // 2)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.2)

    def forward(self, target: Tensor, cond: Tensor) -> Tensor:
        h = torch.cat([self.relu(self.tgt_in(target)), self.relu(self.cond_in(cond))], dim=1)
        h = self.relu(self.norm1(self.layer1(h)))
        h = self.drop(h)
        h = self.relu(self.norm2(self.layer2(h)))
        h = self.drop(h)
        return self.classifier(h)
