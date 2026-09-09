class Config:
    vocab_size = 16384

    embed_dim = 1024
    num_heads = 16
    num_layers = 12
    feedforward_dim = 4096
    max_context = 512

    dropout = 0.1

    batch_size = 12

    gradient_accumulation_steps = 16

    max_train_tokens = 12_000_000_000

    learning_rate = 3e-4
    min_learning_rate = 3e-5
    weight_decay = 0.1

    warmup_tokens = 100_000_000

    validation_interval_tokens = 250_000_000

    validation_batches = 128

    checkpoint_interval_tokens = 500_000_000

    num_workers = 2

    version = "0.3.0"