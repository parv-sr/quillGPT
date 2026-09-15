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

RoPE encodes position by rotating pairs of dimensions in Q and K using 
interleaved half-rotations and supports dynamic sequence length scaling.
"""
import torch
from torch import nn


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        head_dim: int,
        max_context: int,
        base: float = 10000.0,
    ) -> None:
        super().__init__()

        if head_dim % 2 != 0:
            raise ValueError(
                "RoPE head dimension must be even."
            )

        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(
                    0,
                    head_dim,
                    2,
                    dtype=torch.float32,
                )
                / head_dim
            )
        )

        positions = torch.arange(
            max_context,
            dtype=torch.float32,
        )

        frequencies = torch.outer(
            positions,
            inv_freq,
        )

        self.register_buffer(
            "cos_cached",
            frequencies.cos(),
            persistent=False,
        )

        self.register_buffer(
            "sin_cached",
            frequencies.sin(),
            persistent=False,
        )

        self.max_context = max_context
        self.head_dim = head_dim

    @staticmethod
    def rotate_half(
        x: torch.Tensor,
    ) -> torch.Tensor:
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]

        rotated = torch.stack(
            (-x_odd, x_even),
            dim=-1,
        )

        return rotated.flatten(-2)

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int = 0,
    ) -> torch.Tensor:
        sequence_length = x.size(-2)

        end_pos = (
            start_pos
            + sequence_length
        )

        if end_pos > self.max_context:
            raise ValueError(
                f"RoPE position {end_pos} exceeds "
                f"max_context={self.max_context}"
            )

        cos = self.cos_cached[
            start_pos:end_pos
        ].to(
            dtype=x.dtype
        )

        sin = self.sin_cached[
            start_pos:end_pos
        ].to(
            dtype=x.dtype
        )

        cos = torch.repeat_interleave(
            cos,
            2,
            dim=-1,
        )

        sin = torch.repeat_interleave(
            sin,
            2,
            dim=-1,
        )

        return (
            x * cos[None, None, :, :]
            + self.rotate_half(x)
            * sin[None, None, :, :]
        )
