import torch
from torch import nn

from .block import TransformerBlock

class Transformer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, num_layers: int, max_context: int, dropout: float = 0.1) -> None:
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    max_context=max_context,
                    dropout=dropout
                )
                for n in range(num_layers)
            ]
         )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)

        return x
