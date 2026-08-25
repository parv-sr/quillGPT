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

Raw math implementation:

def forward():
        ...
        ...

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

        ...
        ...

---------------------

The V2 Implementation will feature Rotational positional embeddings (RoPE).
"""

import torch
import torch.nn.functional as F
from torch import nn

from .rope import RotaryPositionalEmbedding

from typing import Optional, Tuple

KeyValueCache = tuple[torch.Tensor, torch.Tensor]

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

        self.rope = RotaryPositionalEmbedding(
            head_dim=self.head_dim,
            max_context=max_context
        )

        self.max_context = max_context

    def forward(
        self, 
        x: torch.Tensor, 
        past_key_value: Optional[KeyValueCache] = None, 
        use_cache: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, KeyValueCache]:

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

        if past_key_value is None:
            past_length = 0
        else:
            past_length = past_key_value[0].size(-2)

        q = self.rope(q, position_offset=past_length)
        k = self.rope(k, position_offset=past_length)

        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)

        total_length = k.size(-2)

        if total_length > self.max_context:
            raise ValueError(
                f"Sequence length {total_length} exceeds "
                f"maximum context {self.max_context}"
            )

        if past_key_value is None:
            attention_output = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=True
            )
        else:
            query_positions = torch.arange(past_length, past_length + T, device=x.device)
            key_positions = torch.arange(total_length, device=x.device)

            causal_mask = (query_positions.unsqueeze(1) >= key_positions.unsqueeze(0))

            attention_output = F.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=causal_mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False
            )
        
        output = attention_output.transpose(1, 2)
        output = output.contiguous().view(B, T, C)
        output = self.output_projection(output)

        if use_cache:
            return output, (k, v)
        return output
    
