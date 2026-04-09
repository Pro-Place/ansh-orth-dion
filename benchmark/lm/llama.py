"""
LLaMA-style model (~300M parameters) for optimizer comparison.

Architecture (matching LLaMA 3 style):
  - RMSNorm (not LayerNorm)
  - SwiGLU FFN (gate + up + down projections)
  - Rotary positional embeddings (RoPE)
  - No bias anywhere
  - Weight tying between embedding and LM head
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


def precompute_rope(dim: int, max_seq: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq)
    freqs = torch.outer(t, freqs)
    cos = freqs.cos()
    sin = freqs.sin()
    return cos, sin


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply rotary embeddings. x: (B, n_heads, T, head_dim)."""
    T = x.shape[2]
    cos = cos[:T].unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim/2)
    sin = sin[:T].unsqueeze(0).unsqueeze(0)
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class Attention(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

    def forward(self, x: Tensor, rope_cos: Tensor, rope_sin: Tensor) -> Tensor:
        B, T, C = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, rope_cos, rope_sin)
        k = apply_rope(k, rope_cos, rope_sin)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).reshape(B, T, C)
        return self.wo(y)


class SwiGLUFFN(nn.Module):
    """SwiGLU feed-forward: gate * silu(up) then down."""
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, ffn_dim: int):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = Attention(dim, n_heads)
        self.ffn_norm = RMSNorm(dim)
        self.ffn = SwiGLUFFN(dim, ffn_dim)

    def forward(self, x: Tensor, rope_cos: Tensor, rope_sin: Tensor) -> Tensor:
        x = x + self.attn(self.attn_norm(x), rope_cos, rope_sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class LLaMA(nn.Module):
    """
    LLaMA-style causal LM.

    Presets targeting ~300M params:
      dim=1024, n_layers=24, n_heads=16, ffn_dim=2816 → ~310M
      dim=1024, n_layers=20, n_heads=16, ffn_dim=2816 → ~265M
      dim=1152, n_layers=20, n_heads=16, ffn_dim=3072 → ~330M
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        max_seq_len: int = 1024,
        dim: int = 1024,
        n_layers: int = 24,
        n_heads: int = 16,
        ffn_dim: int = 2816,
        tie_weights: bool = True,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.dim = dim

        self.tok_emb = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            TransformerBlock(dim, n_heads, ffn_dim)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        if tie_weights:
            self.lm_head.weight = self.tok_emb.weight

        # Precompute RoPE
        rope_cos, rope_sin = precompute_rope(dim // n_heads, max_seq_len)
        self.register_buffer("rope_cos", rope_cos, persistent=False)
        self.register_buffer("rope_sin", rope_sin, persistent=False)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.Embedding):
                torch.nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx: Tensor) -> Tensor:
        B, T = idx.shape
        x = self.tok_emb(idx)
        for layer in self.layers:
            x = layer(x, self.rope_cos, self.rope_sin)
        x = self.norm(x)
        return self.lm_head(x)

    def count_params(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        unique = sum(p.numel() for p in set(self.parameters()))
        matrix_2d = sum(p.numel() for p in self.parameters() if p.ndim == 2)
        return {"total": total, "unique": unique, "matrix_2d": matrix_2d}


def create_llama_300m(**overrides) -> LLaMA:
    """Create ~300M param LLaMA model."""
    kwargs = dict(
        vocab_size=50257, max_seq_len=1024,
        dim=1024, n_layers=24, n_heads=16, ffn_dim=2816,
        tie_weights=True,
    )
    kwargs.update(overrides)
    model = LLaMA(**kwargs)
    c = model.count_params()
    print(f"LLaMA: {c['unique']/1e6:.1f}M unique params "
          f"({c['matrix_2d']/1e6:.1f}M matrix)")
    return model
