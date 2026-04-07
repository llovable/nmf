import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional


class CrossAttentionBlock(nn.Module):
    """
    Cross-attention 블록.
    """
    def __init__(
        self,
        d_in_src: int,
        d_in_tgt: int,
        d_model: int = 256,
        ff: int = 512,
        pdrop: float = 0.1,
        n_src_tokens: int = 8,
        n_heads: int = 4,
        d_head: int = 64,
    ):
        super().__init__()
        self.n_src_tokens = n_src_tokens
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_qkv = n_heads * d_head

        self.src_proj = nn.Linear(d_in_src, n_src_tokens * d_model, bias=False)
        self.tgt_proj = nn.Linear(d_in_tgt, d_model, bias=False)

        self.q = nn.Linear(d_model, self.d_qkv, bias=False)
        self.k = nn.Linear(d_model, self.d_qkv, bias=False)
        self.v = nn.Linear(d_model, self.d_qkv, bias=False)
        self.o = nn.Linear(self.d_qkv, d_model, bias=False)

        self.ff1 = nn.Linear(d_model, ff)
        self.ff2 = nn.Linear(ff, d_model)

        self.ln1 = nn.GroupNorm(8, d_model)
        self.ln2 = nn.GroupNorm(8, d_model)
        self.drop = nn.Dropout(pdrop)

    def forward(self, src: Tensor, tgt: Tensor):
        B, m = src.size(0), self.n_src_tokens
        S = self.src_proj(src).view(B, m, -1)
        T = self.tgt_proj(tgt).unsqueeze(1)

        q = self.q(T).view(B, 1, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k(S).view(B, m, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v(S).view(B, m, self.n_heads, self.d_head).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / (self.d_head ** 0.5)
        scores = scores - scores.max(dim=-1, keepdim=True).values
        attn = torch.softmax(scores, dim=-1)
        attn_out = (attn @ v).transpose(1, 2).contiguous().view(B, 1, -1)
        attn_out = self.o(attn_out).squeeze(1)

        h = self.ln1(T.squeeze(1) + self.drop(attn_out))
        f = self.ff2(F.relu(self.ff1(h)))
        y = self.ln2(h + self.drop(f))

        return y, attn


class Generator(nn.Module):
    """
    소스 → 타깃 모달리티 생성기.
    """
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_dim = [1024, 512],
        use_attn: bool = False,
        nonneg_output: bool = True,
        n_src_tokens: int = 8,
        n_heads: int = 4,
        d_head: int = 64,
        target_type: str = "rna",
        src_size: int = None,
    ):
        super().__init__()
        self.use_attn = use_attn
        self.n_heads = n_heads
        self.d_head = d_head
        self.target_type = target_type
        self.src_size = src_size or input_size

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
            self.attn_proj = nn.Linear(256, input_size, bias=False)
            self.alpha = nn.Parameter(torch.full((input_size,), -2.2))
            nn.init.zeros_(self.attn_proj.weight)

        self.layer1 = nn.Linear(input_size, hidden_dim[0])
        self.layer2 = nn.Linear(hidden_dim[0], hidden_dim[-1])
        self.layer3 = nn.Linear(hidden_dim[-1], output_size)

        self.norm1 = nn.GroupNorm(8, hidden_dim[0])
        self.norm2 = nn.GroupNorm(8, hidden_dim[-1])
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.2)

    def forward(self, xs: Tensor, src: Optional[Tensor] = None) -> Tensor:
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
    WGAN Critic.
    """
    def __init__(self, input_size: int, hidden_dim: int = 512):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.classifier = nn.Linear(hidden_dim // 2, 1)

        self.norm1 = nn.GroupNorm(8, hidden_dim)
        self.norm2 = nn.GroupNorm(8, hidden_dim // 2)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.2)

    def forward(self, xs: Tensor) -> Tensor:
        xs = self.relu(self.norm1(self.layer1(xs)))
        xs = self.drop(xs)
        xs = self.relu(self.norm2(self.layer2(xs)))
        xs = self.drop(xs)
        return self.classifier(xs)
