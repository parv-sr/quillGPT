"""
Here we assemble the entire pipeline together.
This will follow a decoder only architecture
"""

import torch
from torch import nn

from typing import Optional, Tuple

from .attention import KeyValueCache
from .embedding import Embedding
from .transformer import Transformer

class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int, 
        embed_dim: int,
        num_heads: int, 
        num_layers: int, 
        max_context: int,
        feedforward_dim: int, 
        dropout: float = 0.1
    ) -> None:
        super().__init__()

        self.embedding = Embedding(
            vocab_size=vocab_size,
            embed_dim=embed_dim
        )

        self.transformer = Transformer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            max_context=max_context,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
        )

        self.final_layer_norm = nn.RMSNorm(embed_dim)

        self.output_projection = nn.Linear(
            embed_dim,
            vocab_size,
        )

    def forward(self, tokens: torch.Tensor, past_key_values: Optional[Tuple[KeyValueCache, ...]] = None, use_cache: bool = False) -> torch.Tensor | Tuple[torch.Tensor, Tuple[KeyValueCache, ...]]:
        x = self.embedding(tokens)

        if use_cache:
            x, present_key_values = self.transformer(
                x,
                past_key_values=past_key_values,
                use_cache=True
            )
        else:
            x = self.transformer(x)

        x = self.final_layer_norm(x)

        logits = self.output_projection(x)

        if use_cache:
            return logits, present_key_values

        return logits
     
