class Config:
    vocab_size = 16384

    embed_dim = 2048
    num_heads = 8
    num_layers = 12
    feedforward_dim = 8192

    max_context = 512

    dropout = 0.1

    batch_size = 64

    learning_rate = 5e-4
    min_learning_rate = 3e-5

    weight_decay = 0.1

    warmup_fraction = 0.05

    epochs = 5

    version: str = "0.1.4"