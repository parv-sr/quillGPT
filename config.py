class Config:
    vocab_size = 256
    embed_dim = 384
    num_heads = 12
    num_layers = 8
    feedforward_dim = 1024

    max_context = 512

    dropout = 0.1

    batch_size = 32

    learning_rate = 3e-4

    epochs = 10

    version: str = "0.0.2"