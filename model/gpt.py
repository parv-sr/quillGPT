"""
Here we assemble the entire pipeline together.
This will follow a decoder only architecture
"""

import torch
from torch import nn

from .embedding import Embedding
from .transformer import Transformer

class GPT(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int, num_layers: int, max_context: int, dropout: float = 0.1) -> None:
        super().__init__()

        self.embedding = Embedding(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            max_context=max_context,
        )

        self.transformer = Transformer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            max_context=max_context,
            dropout=dropout,
        )

        self.final_layer_norm = nn.LayerNorm(embed_dim)

        self.output_projection = nn.Linear(
            embed_dim,
            vocab_size,
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embedding(tokens)
        x = self.transformer(x)
        x = self.final_layer_norm(x)

        logits = self.output_projection(x)

        return logits
    
