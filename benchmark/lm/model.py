"""
Small GPT-2 model (~45M parameters).

Architecture:
  - Vocabulary: 50257 (GPT-2 tokenizer)
  - Context length: 1024
  - Hidden dim: 768
  - Heads: 12
  - Layers: 6
  - FFN inner: 3072 (4x hidden)
  - Total params: ~45.2M
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CausalSelfAttention(nn.Module):
    def __init__(self, hidden: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert hidden % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = hidden // n_heads
        self.qkv = nn.Linear(hidden, 3 * hidden, bias=False)
        self.proj = nn.Linear(hidden, hidden, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each (B, T, nh, hd)
        q = q.transpose(1, 2)  # (B, nh, T, hd)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention with causal mask
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.attn_dropout.p if self.training else 0.0
        )
        y = y.transpose(1, 2).reshape(B, T, C)
        return self.resid_dropout(self.proj(y))


class MLP(nn.Module):
    def __init__(self, hidden: int, inner: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(hidden, inner, bias=False)
        self.fc2 = nn.Linear(inner, hidden, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.fc2(F.gelu(self.fc1(x), approximate="tanh")))


class TransformerBlock(nn.Module):
    def __init__(self, hidden: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden)
        self.attn = CausalSelfAttention(hidden, n_heads, dropout)
        self.ln2 = nn.LayerNorm(hidden)
        self.mlp = MLP(hidden, 4 * hidden, dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class SmallGPT2(nn.Module):
    """
    Small GPT-2 (~45M params).

    Parameter count breakdown:
      - Token embedding:  50257 * 768 = 38,597,376
      - Position embedding: 1024 * 768 = 786,432
      - 6 Transformer blocks:
        - qkv:  768 * 2304 = 1,769,472
        - proj:  768 * 768 = 589,824
        - fc1:  768 * 3072 = 2,359,296
        - fc2:  3072 * 768 = 2,359,296
        - ln1, ln2: 768 * 2 * 2 = 3,072
        - Total per block: 7,080,960
        - Total 6 blocks: 42,485,760... wait that's too many

    Adjusted: 6 layers gives ~45M with weight tying.
    With weight tying (lm_head shares token embedding):
      wte: 38.6M + wpe: 0.8M + 6 blocks * ~7.1M/block = ~39.4M + 42.5M...

    Actually let's just build it and count.
    """

    def __init__(
        self,
        vocab_size: int = 50257,
        max_seq_len: int = 1024,
        hidden: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        dropout: float = 0.0,
        tie_weights: bool = True,
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.hidden = hidden

        self.wte = nn.Embedding(vocab_size, hidden)
        self.wpe = nn.Embedding(max_seq_len, hidden)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(hidden, n_heads, dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(hidden)

        self.lm_head = nn.Linear(hidden, vocab_size, bias=False)
        if tie_weights:
            self.lm_head.weight = self.wte.weight

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                torch.nn.init.ones_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, idx: Tensor) -> Tensor:
        """
        Args:
            idx: (B, T) token indices
        Returns:
            logits: (B, T, vocab_size)
        """
        B, T = idx.shape
        assert T <= self.max_seq_len, f"Sequence length {T} > max {self.max_seq_len}"

        pos = torch.arange(T, device=idx.device, dtype=torch.long)
        x = self.drop(self.wte(idx) + self.wpe(pos))

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        return self.lm_head(x)

    def count_params(self) -> dict:
        """Return parameter counts by category."""
        total = sum(p.numel() for p in self.parameters())
        # Unique params (account for weight tying)
        unique = sum(p.numel() for p in set(self.parameters()))
        matrix_params = sum(
            p.numel() for p in self.parameters() if p.ndim == 2
        )
        non_matrix = total - matrix_params
        return {
            "total": total,
            "unique": unique,
            "matrix_2d": matrix_params,
            "non_matrix": non_matrix,
        }


def create_small_gpt2(**kwargs) -> SmallGPT2:
    """Create the small GPT-2 model with default config."""
    model = SmallGPT2(**kwargs)
    counts = model.count_params()
    print(f"SmallGPT2: {counts['unique']/1e6:.1f}M unique params "
          f"({counts['matrix_2d']/1e6:.1f}M matrix, "
          f"{counts['non_matrix']/1e6:.1f}M non-matrix)")
    return model
