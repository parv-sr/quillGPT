from typing import Dict, List
import torch
from torch import nn

class Embedding(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.token_embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.token_embeddings(tokens)

    


