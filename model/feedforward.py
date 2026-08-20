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

        print("Input:", x.shape)

        x = self.linear1(x)
        print("After Linear1:", x.shape)

        x = self.activation(x)
        print("After GELU:", x.shape)

        x = self.linear2(x)
        print("After Linear2:", x.shape)

        x = self.dropout(x)
        print("After Dropout:", x.shape)

        return x
    

x = torch.randn(2, 5, 128)
ffn = FeedForwardNetwork(128)
out = ffn(x)