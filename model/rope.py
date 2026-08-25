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
    def __init__(
        self,
        head_dim: int,
        max_context: int,
        base: float = 10000.0
    ) -> None:
        super().__init__()

        if head_dim % 2 != 0:
            raise ValueError("RoPE requires an even head dimension")

        self.head_dim = head_dim
        self.max_context = max_context

        inverse_frequencies = 1.0 / (
            base ** (
                torch.arange(0, head_dim, 2).float() / head_dim
            )
        )

        positions = torch.arange(max_context).float()

        frequencies = torch.outer(
            positions,
            inverse_frequencies
        )

        cos = frequencies.cos()
        sin = frequencies.sin()

        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, x: torch.Tensor, position_offset: int = 0) -> torch.Tensor:
        _, _, sequence_length, _ = x.shape

        end_position = position_offset + sequence_length

        if end_position > self.max_context:
            raise ValueError(
                f"Sequence exceeds maximum context of {self.max_context}"
            )

        cos = self.cos[position_offset:end_position]
        sin = self.sin[position_offset:end_position]

        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        x_first, x_second = x.chunk(2, dim=-1)

        rotated_first = x_first * cos - x_second * sin
        rotated_second = x_first * sin + x_second * cos

        return torch.cat(
            [rotated_first, rotated_second],
            dim=-1
        )