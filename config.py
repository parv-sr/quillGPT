class Config:
    vocab_size = 65

    embed_dim = 128
    num_heads = 4
    num_layers = 4
    feedforward_dim = 512

    max_context = 128

    dropout = 0.1

    batch_size = 64

    learning_rate = 3e-4

    epochs = 20

    version: str = "0.0.1"