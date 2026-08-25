from torch import nn
import torch


class FeedForwardNetwork(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int = 512, dropout: float = 0.1) -> None:
        super().__init__()

        self.gate_projection = nn.Linear(embed_dim, hidden_dim)
        self.value_projection = nn.Linear(embed_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, embed_dim)

        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.activation(self.gate_projection(x))
        value = self.value_projection(x)

        x = gate * value
        x = self.output_projection(x)
        x = self.dropout(x)

        return x
