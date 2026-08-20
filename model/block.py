import torch
from torch import nn

class LayerNorm(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()

        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer_norm(x)
    

if __name__ == "__main__":
    x = torch.tensor([1.0, 2.0, 3.0, 4.0])
    layer_norm = LayerNorm(4)
    output = layer_norm(x)

    print(f"Output: {output}\n\n")
    print(f"Output mean: {output.mean()}\n\n")
    print(f"Output std dev: {output.std()}\n\n")