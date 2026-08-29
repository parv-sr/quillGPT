class Config:
    vocab_size = 8192

    embed_dim = 512
    num_heads = 8
    num_layers = 12
    feedforward_dim = 2048

    max_context = 512

    dropout = 0.15

    batch_size = 64

    learning_rate = 5e-4
    min_learning_rate = 3e-5

    weight_decay = 0.1

    warmup_fraction = 0.03

    epochs = 15

    version: str = "0.1.4"