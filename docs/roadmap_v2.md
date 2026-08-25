## v2 - Larger scale pretrained transformer

1. BPE tokenizer
2. RMSNorm replacing LayerNorm
3. Rotatory positional embeddings
4. SwiGLU Activation in FFN
6. KV cache (2 heads)


### More architectural refinements:

1. Preallocated memory for KV cache store
2. Pytorch optimised RoPE implementation inside MHA class