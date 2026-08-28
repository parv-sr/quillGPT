from pathlib import Path
from typing import List

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders


class BPETokenizer:

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size: int = vocab_size
        self._init_empty_tokenizer()

    def _init_empty_tokenizer(self) -> None:
        self.tokenizer = Tokenizer(
            models.BPE(unk_token=None)
        )

        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
            add_prefix_space=False
        )

        self.tokenizer.decoder = decoders.ByteLevel()

    def train(self, text: str) -> None:
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=2,
            show_progress=True,
            special_tokens=["<|endoftext|>"]
        )

        self.tokenizer.train_from_iterator(
            [text],
            trainer=trainer
        )

        self.vocab_size = self.tokenizer.get_vocab_size()

    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text).ids

    def decode(self, token_ids: List[int]) -> str:
        return self.tokenizer.decode(token_ids)

    def save(self, path: Path | str) -> None:
        self.tokenizer.save(str(path))

    def load(self, path: Path | str) -> None:
        self.tokenizer = Tokenizer.from_file(str(path))
        self.vocab_size = self.tokenizer.get_vocab_size()