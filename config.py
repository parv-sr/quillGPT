class Config:
    vocab_size = 65

    embed_dim = 128
    num_heads = 4
    num_layers = 3
    feedforward_dim = 256

    max_context = 128

    dropout = 0.3

    batch_size = 64

    learning_rate = 1e-4

    epochs = 5

    version: str = "0.0.1"