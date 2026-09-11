# quillGPT - A pretrained transformer

quillGPT is a pretrained transformer trained on 12B tokens from various datasets from the internet.  
It is trained to generate general English text.  

Written manually in PyTorch around abstractions of `nn.Module`


### Model features (v0.3.0):

* 235M Parameters
* 12 transformer blocks
* Multi head Self-attention based on "Attention is all you need" (Vaswani et al, Google 2017)
* 16 Attention heads
* SWiGLU Activation
* GPT-3 style decoder-only architecture
* LlaMA-style RoPE & Pre-RMSNorm implementation

### Inference Features:

* Inference engine powered by ONNX.
* Prompt handling, caching and streaming responses all handled by an ONNX runtime.
* A lightweight FastAPI layer to use inference in a real product.
* Validation of human prompts done at API layer.


#### How to run:

1. In the quillGPT directory, run this command-

```bash
uvicorn backend.api.app:app --reload --host "0.0.0.0" --port 8000
```

This will start the inference server and load the model and inference engine into active memory.  

```/infer/generate``` Endpoint will generate the responses.

##### Authored by: Parv Sharma, FLAME University