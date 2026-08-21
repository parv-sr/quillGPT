from torch import nn
import torch


class FeedForwardNetwork(nn.Module):
    def __init__(self, embed_dim: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()

        hidden_dim = embed_dim * expansion

        self.linear1 = nn.Linear(embed_dim, hidden_dim)
        self.activation = nn.GELU()
        self.linear2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)

        x = self.activation(x)

        x = self.linear2(x)

        x = self.dropout(x)
        
        return x
    
