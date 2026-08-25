"""
Here I will build a transformer block, which is essentially assembling all the tiny pieces I have built so far.

A transformer block takes input: input tensor of shape (B, T, C)
and outputs a tensor of shape: (B, T, C)

Inside, however, it does perform transformations

Input
  │
  ▼
RMSNorm
  │
  ▼
Self-Attention -----
  │                │
  ▼                RoPE 
Residual Addition
  │
  ▼
RMSNorm
  │
  ▼
FeedForward
  │
  ▼
Residual Addition
  │
  ▼
Output

This is a pre-RMSNorm network

x1​ = x + Attention(LayerNorm(x))
x2 = x1 + FFN(LayerNorm(x1))


"""


import torch
from torch import nn

from typing import Tuple, Optional

from .attention import CausalSelfAttention, KeyValueCache
from .feedforward import FeedForwardNetwork


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, max_context: int, feedforward_dim: int, dropout: float = 0.1) -> None:
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
            hidden_dim=feedforward_dim,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor, past_key_value: Optional[KeyValueCache] = None, use_cache: bool = False) -> torch.Tensor | Tuple[torch.Tensor, KeyValueCache]:
        if use_cache:
            attention_output, present_key_value = self.attention(
                self.rms_norm1(x),
                past_key_value=past_key_value,
                use_cache=True
            )

            x = x + attention_output
            x = x + self.feed_forward(self.rms_norm2(x))
            return x, present_key_value
        
        x = x + self.attention(self.rms_norm1(x))
        x = x + self.feed_forward(self.rms_norm2(x))

        return x