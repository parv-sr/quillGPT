from typing import Dict, List
import torch
from torch import nn

class Embedding(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, max_context: int):
        super().__init__()
        self.token_embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim
        )
        self.position_embeddings = nn.Embedding(
            num_embeddings=max_context,
            embedding_dim=embed_dim
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, T = tokens.shape

        positions = torch.arange(T, device=tokens.device)

        token_vectors = self.token_embeddings(tokens)
        position_vectors = self.position_embeddings(positions)
        
        x = token_vectors + position_vectors

        return x
    


