"""
Attention gives every token a way to selectively gather information from the other tokens.

Q, K, V Matrices
-> These are simply three different learned transformations of the same input.

Query: "What information am I looking for?"
Key: "What information do I contain / what am I relevant to?"
Value: "Here's the actual information you can take from me."

** The query and key determine how much attention one token pays to another
** The value contains the information that actually gets mixed together

Attention matrix: Q * K^T

In the attention matrix:
-> Each row corresponds to the token doing the looking
-> Each column corresponds to the token being looked at

Then softmax is applied to each row of the attention matrix, making it a probability distribution

Then finally: attention weights * Value matrix

For a decoder only pretrained transformer, a *Causal Mask* is required, so that it does not cheat in training by looking at the next token.
A causal mask hides the next tokens in the sequence.
This effectively makes it a causal self-attention.
"""

import torch
import torch.nn.functional as F
from torch import nn

class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, max_context: int, dropout: float = 0.1) -> None:
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError("Embedding dimensions must be divisible by no. of heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.output_projection = nn.Linear(
            embed_dim,
            embed_dim
        )

        self.dropout = nn.Dropout(dropout)

        mask = torch.tril(
            torch.ones(max_context, max_context)
        )

        self.register_buffer(
            "causal_mask",
            mask
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        qkv = self.qkv(x)

        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.num_heads, self.head_dim)
        k = k.view(B, T, self.num_heads, self.head_dim)
        v = v.view(B, T, self.num_heads, self.head_dim)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Shape of attention matrices after transpose become (B, H, T, D)
        # Where H -> Number of heads & D -> Dimension of head
        """
        attention_scores = q @ k.transpose(-2, -1)

        attention_scores = attention_scores / (
            self.head_dim ** 0.5
        )

        attention_scores = attention_scores.masked_fill(
            self.causal_mask[:T, :T] == 0,
            float('-inf')
        )

        attention_weights = torch.softmax(
            attention_scores,
            dim=-1
        )

        attention_weights = self.dropout(
            attention_weights
        )

        output = attention_weights @ v
        """
        output = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=True
        )
        output = output.transpose(1, 2)
        output = output.contiguous().view(B, T, C)

        output = self.output_projection(output)

        return output
    
