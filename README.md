# quillGPT

quillGPT is a 235M-parameter decoder-only language model implemented from scratch in PyTorch and pretrained on approximately 12 billion tokens of general internet text.

The project began as an attempt to understand transformer architectures beyond library-level abstractions. The model, training system, tokenizer pipeline, data preprocessing, inference engine, KV caching, and serving layer are implemented manually around lower-level PyTorch primitives.

quillGPT is a **base pretrained language model**, not an instruction-tuned or chat model. Its objective is simple next-token prediction.

## quillGPT v0.4.0

### Architecture

- **235,082,752 parameters**
- **12 transformer blocks**
- **1024-dimensional embeddings**
- **16 attention heads**
- **64-dimensional attention heads**
- **4096-dimensional feed-forward network**
- **16,384-token BPE vocabulary**
- **512-token context window**
- Decoder-only causal transformer
- Multi-Head Self-Attention
- Rotary Position Embeddings (RoPE)
- Pre-RMSNorm
- SwiGLU feed-forward layers
- Scaled Dot Product Attention through PyTorch SDPA

The architecture is inspired primarily by GPT-style decoder-only transformers and several architectural choices used by LLaMA.

Each transformer block follows:

```text
x = x + Attention(RMSNorm(x))
x = x + SwiGLU(RMSNorm(x))
```

The attention mechanism uses a fused QKV projection, causal self-attention, RoPE on queries and keys, and KV caching during autoregressive inference.

---

## Training

v0.4.0 was pretrained from scratch on approximately:

**12,000,000,000 tokens**

with a 235M-parameter model, corresponding to approximately:

**51 tokens per parameter**

The run was designed partly as an experiment in training a relatively small model substantially beyond the commonly cited ~20 tokens/parameter Chinchilla compute-optimal reference point.

This is not intended as a contradiction of Chinchilla scaling laws, which describe compute-optimal allocation between model size and dataset size. Instead, the experiment investigates how long a fixed small model continues to benefit from additional data.

### Training configuration

- NVIDIA RTX A4500
- BF16 mixed precision
- `torch.compile`
- PyTorch Scaled Dot Product Attention
- Fused AdamW
- Cosine learning-rate decay
- 100M-token warmup
- Peak learning rate: `2e-4`
- Minimum learning rate: `2e-5`
- Microbatch size: `12`
- Gradient accumulation: `16`
- Effective batch size: **98,304 tokens**
- Gradient clipping: `1.0`
- Weight decay: `0.1`

The complete run required approximately **120 hours** on a shared university GPU server.

Peak training memory remained around:

```text
8.53 GiB allocated
8.84 GiB reserved
```

while sustained throughput reached approximately **32–33k tokens/sec** when the GPU was uncontested.

---

## Data Pipeline

One of the major findings of this project was that data layout mattered almost as much as model configuration.

An earlier training pipeline concatenated large source files into a single sequential token stream. Although this allowed efficient sequential disk access, the model encountered extremely long homogeneous regions from individual sources.

This caused large and highly reproducible validation-loss oscillations as the model moved between different data distributions.

v0.4.0 uses a redesigned pipeline:

1. Raw text is tokenized in bounded-memory batches.
2. Tokens are stored as `uint16`.
3. The corpus is divided into approximately **1M-token shards**.
4. These shards are shuffled once during preprocessing.
5. The shuffled shards are concatenated into a sequential `train.bin`.
6. Training still performs sequential disk reads using NumPy memory maps.

This provides both:

- good statistical mixing;
- efficient sequential I/O.

The complete tokenized corpus contains approximately:

```text
Training:   12.636B tokens
Validation: 13.0M tokens
```

The model trains directly from memory-mapped binary token files without loading the complete corpus into RAM.

---

## Training Results

The validation curve improved throughout almost the entire 12B-token run.

| Tokens Seen | Validation Loss | Perplexity |
|---:|---:|---:|
| 0.25B | 4.4492 | 85.56 |
| 1.00B | 3.6256 | 37.55 |
| 2.00B | 3.4448 | 31.34 |
| 3.00B | 3.3408 | 28.24 |
| 4.00B | 3.2957 | 27.00 |
| 5.00B | 3.2210 | 25.05 |
| 6.00B | 3.1824 | 24.11 |
| 7.00B | 3.1359 | 23.01 |
| 8.00B | 3.1155 | 22.54 |
| 9.00B | 3.0749 | 21.65 |
| 10.00B | 3.0572 | 21.27 |
| 11.00B | 3.0445 | 21.00 |
| 11.75B | **3.0326** | **20.75** |
| 12.00B | 3.0339 | 20.78 |

The best checkpoint occurred at approximately **11.75B tokens**.

A particularly interesting result is that the model continued improving well beyond the ~20 tokens/parameter point.

For this model:

```text
20 tokens / parameter ≈ 4.70B tokens
```

Around that point, validation loss was approximately `3.24`.

Training to approximately 50 tokens/parameter reduced validation loss further to `3.03`.

The improvement became progressively smaller, but continued almost until the end of training.

---

## Qualitative Behaviour

v0.4.0 is the first quillGPT model to consistently produce multi-sentence English text with recognizable semantic structure.

For example:

### Prompt

```text
As humans, we must progressively work towards a brighter future by
```

### Completion

```text
recognizing and appreciating the consequences of our past mistakes.

The story is told from our perspective: this is not an easy journey.
And if we don’t strive to do that, how can we make sense out of it?
How can we achieve optimal results when we aren’t yet ready for it?
...
```

The model demonstrates:

- strong English syntax;
- reasonable short-range coherence;
- learned genre and writing-style structure;
- multi-sentence continuation;
- basic narrative generation;
- recognizable discourse structure.

It also demonstrates clear limitations.

### Known limitations

quillGPT v0.4.0 frequently:

- hallucinates facts;
- invents historical figures and organizations;
- produces incorrect numerical information;
- loses entities over longer generations;
- changes topic unexpectedly;
- struggles with long-range factual consistency;
- falls into repetition loops;
- performs poorly on technical and programming knowledge;
- produces increasingly incoherent text at high sampling temperatures.

The model often learns the **form of factual or encyclopedic writing more strongly than the underlying facts themselves**.

For example, it can produce extremely convincing Wikipedia-like prose containing almost entirely fabricated information.

This distinction between linguistic modeling and factual knowledge was one of the more interesting outcomes of the experiment.

---

## Inference

quillGPT includes a standalone inference stack built around ONNX Runtime and FastAPI.

Features include:

- ONNX model export
- autoregressive generation
- KV caching
- temperature sampling
- top-p nucleus sampling
- repetition penalty
- prompt validation
- streaming-ready backend
- FastAPI serving layer

### Start the server

```bash
uvicorn backend.api.app:app \
    --host 0.0.0.0 \
    --port 8000
```

Generation is exposed through:

```text
POST /infer/generate
```

Example request:

```json
{
  "prompt": "Beyond the last known star, the explorers discovered",
  "max_new_tokens": 100,
  "temperature": 0.8,
  "top_p": 0.9,
  "repetition_penalty": 1.15
}
```

For v0.4.0, generation generally works best around:

```text
temperature:         0.6 - 0.8
top_p:               0.85 - 0.92
repetition_penalty:  1.10 - 1.20
```

Higher temperatures rapidly expose the limitations of the small model.

---

## Evolution of quillGPT

The project originally began with a **628k-parameter transformer** trained on Tiny Shakespeare.

That model learned grammar, Shakespeare-like formatting, and basic language structure, but could not generate reliably coherent sentences.

v0.4.0 scales that experiment to:

```text
628K parameters  ->  235M parameters
Tiny Shakespeare ->  12B-token internet corpus
LayerNorm         ->  RMSNorm
GELU              ->  SwiGLU
absolute/simple positional representations -> RoPE
basic attention   ->  SDPA + KV caching
```

The difference in qualitative behavior between these models is one of the main motivations behind this project.

---

## Key Takeaways

The experiment produced several useful observations:

1. **Data mixing matters enormously.**  
   Poor corpus ordering initially appeared to resemble model overfitting.

2. **Small models continue benefiting from large token budgets.**  
   quillGPT continued reducing validation loss well beyond 20 tokens per parameter.

3. **Language ability emerges before factual reliability.**  
   The model learned syntax, genre, and local discourse much more strongly than factual knowledge.

4. **Validation loss does not fully describe generation quality.**  
   A model with steadily improving cross-entropy can still hallucinate, lose context, repeat itself, or drift between topics.

5. **Efficient preprocessing is essential at billion-token scale.**  
   A naive tokenizer implementation briefly consumed over 200 GB of RAM. Streaming tokenization, `uint16` storage, memory mapping, and offline shuffling made the final pipeline practical.

6. **A relatively small transformer can exhibit surprisingly rich generative behavior.**  
   At 235M parameters, quillGPT is far smaller than modern production LLMs, yet still develops recognizable language generation and genre imitation.

---

## Status

quillGPT v0.4.0 concludes the current pretraining experiment.

Future work may explore:

- larger context windows;
- better corpus composition;
- increased model capacity;
- grouped-query attention;
- supervised instruction tuning;
- improved factual evaluation;
- standardized language-model benchmarks.

---

### Author

**Parv Sharma**  
FLAME University

Built as an independent research and engineering project exploring transformer architecture, scaling, and language-model pretraining.