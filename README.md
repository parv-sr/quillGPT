# quillGPT - A pretrained transformer

quillGPT is a pretrained transformer trained on Andrej Karpathy's `tinyshakespeare` dataset.  
It is trained to generate old English, shakespeare-like text.  

Written manually in PyTorch around abstractions of `nn.Module`


### Model features:

* 628,433 Parameters
* 4 transformer blocks
* Multi head Self-attention based on "Attention is all you need" (Vaswani et al, Google 2017)
* 4 Attention heads
* GeLU Activation
* GPT-3 style decoder-only architecture
* Training dataset: `tinyshakespeare` (https://github.com/karpathy/char-rnn)

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

V2 Features 235M parameters.

##### Authored by: Parv Sharma, FLAME University