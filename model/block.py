"""
Here I will build a transformer block, which is essentially assembling all the tiny pieces I have built so far.

A transformer block takes input: input tensor of shape (B, T, C)
and outputs a tensor of shape: (B, T, C)

Inside, however, it does perform transformations

Input
  │
  ▼
LayerNorm
  │
  ▼
Self-Attention
  │
  ▼
Residual Addition
  │
  ▼
LayerNorm
  │
  ▼
FeedForward
  │
  ▼
Residual Addition
  │
  ▼
Output

This is a pre-LayerNorm network

x1​ = x + Attention(LayerNorm(x))
x2 = x1 + FFN(LayerNorm(x1))


"""


import torch
from torch import nn

from .attention import CausalSelfAttention
from .feedforward import FeedForwardNetwork


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, max_context: int, dropout: float = 0.1) -> None:
        super().__init__()

        self.rms_norm1 = nn.RMSNorm(embed_dim)

        self.attention = CausalSelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            max_context=max_context,
            dropout=dropout
        )

        self.rms_norm2 = nn.RMSNorm(embed_dim)

        self.feed_forward = FeedForwardNetwork(
            embed_dim=embed_dim,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = x + self.attention(self.rms_norm1(x))

        x = x + self.feed_forward(self.rms_norm2(x))

        return x