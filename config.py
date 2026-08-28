class Config:
    vocab_size = 4096

    embed_dim = 512
    num_heads = 8
    num_layers = 12
    feedforward_dim = 1536

    max_context = 512

    dropout = 0.1

    batch_size = 32

    learning_rate = 3e-4
    min_learning_rate = 3e-5

    weight_decay = 0.1

    warmup_fraction = 0.05

    epochs = 10

    version: str = "0.1.3"