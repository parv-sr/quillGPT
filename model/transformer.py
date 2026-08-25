import torch
from torch import nn

from typing import Optional, Tuple, List

from .attention import KeyValueCache
from .block import TransformerBlock

class Transformer(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, num_layers: int, max_context: int, feedforward_dim: int, dropout: float = 0.1) -> None:
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    max_context=max_context,
                    feedforward_dim=feedforward_dim,
                    dropout=dropout
                )
                for n in range(num_layers)
            ]
         )

    def forward(
        self,
        x: torch.Tensor, 
        past_key_values: Optional[Tuple[KeyValueCache, ...]] = None,
        use_cache: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, Tuple[KeyValueCache, ...]]:
        
        present_key_values: List[KeyValueCache] = []

        for index, block in enumerate(self.blocks):
            past_key_value = (past_key_values[index] if past_key_values is not None else None)

            if use_cache:
                x, present_key_value = block(
                    x, past_key_value=past_key_value, use_cache=True
                )

                present_key_values.append(present_key_value)
            else:
                x = block(x)

        if use_cache:
            return x, tuple(present_key_values)

        return x
