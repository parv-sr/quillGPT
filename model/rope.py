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


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        head_dim: int,
        max_context: int = 2048,
        base: float = 10000.0,
        scaling_factor: float = 1.0,
    ) -> None:
        super().__init__()

        if head_dim % 2 != 0:
            raise ValueError("RoPE requires an even head dimension")

        self.head_dim = head_dim
        self.max_context = max_context
        self.base = base
        self.scaling_factor = scaling_factor

    def _compute_cos_sin(
        self, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = 1.0 / (
            self.base
            ** (
                torch.arange(0, self.head_dim, 2, device=device).float()
                / self.head_dim
            )
        )
        t = (
            torch.arange(seq_len, device=device, dtype=torch.float32)
            / self.scaling_factor
        )
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        cos = emb.cos().to(dtype=dtype)
        sin = emb.sin().to(dtype=dtype)
        return cos, sin

    def forward(
        self, x: torch.Tensor, position_offset: int = 0
    ) -> torch.Tensor:
        # x shape: (B, H, T, D)
        _, _, sequence_length, _ = x.shape
        end_position = position_offset + sequence_length

        cos, sin = self._compute_cos_sin(
            end_position, device=x.device, dtype=x.dtype
        )

        cos = cos[position_offset:end_position].unsqueeze(0).unsqueeze(0)
        sin = sin[position_offset:end_position].unsqueeze(0).unsqueeze(0)

        return (x * cos) + (rotate_half(x) * sin)