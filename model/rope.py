"""
Rotary Positional Embeddings (RoPE).

RoPE encodes position by rotating pairs of dimensions in Q and K.

Input shape:
    (B, H, T, D)

where:
    B = batch size
    H = number of attention heads
    T = sequence length
    D = head dimension

Output shape:
    (B, H, T, D)
"""

import torch
from torch import nn

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_context: int, base: float = 10000.0) -> None:
        super().__init__()

        if head_dim % 2 != 0:
            raise ValueError("Head dim must be even")

        self.head_dim = head_dim
        self.max_context = max_context
        self.base = base

        inverse_frequencies = 1.0 / (
            base ** (
                torch.arange(0, head_dim, 2).float() / head_dim
            )
        )

        positions = torch.arange(max_context).float()

        angles = torch.outer(positions, inverse_frequencies)

        self.register_buffer("cos", angles.cos())
        self.register_buffer("sin", angles.sin())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, T, D = x.shape

        if D != self.head_dim:
            raise ValueError(f"Expected head dimension {self.head_dim}, got {D}")

        if T > self.max_context:
            raise ValueError(f"Sequence length {T} exceeds maximum context length {self.max_context}")

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        cos = self.cos[:T].unsqueeze(0).unsqueeze(0)
        sin = self.sin[:T].unsqueeze(0).unsqueeze(0)

        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos

        output = torch.stack(
            (rotated_even, rotated_odd),
            dim=-1
        )

        return output.flatten(-2)

    