import json

from pathlib import Path
from typing import Dict, List, Tuple

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders


class BPETokenizer:
    def __init__(self, vocab_size: int) -> None:
        self.vocab_size: int = vocab_size
        self._init_empty_tokenizer()

    def _init_empty_tokenizer(self) -> None:
        self.tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        self.tokenizer.decoder = decoders.ByteLevel()

    def train(self, text: str) -> None:
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=2,
            show_progress=True,
            special_tokens=["<unk>"],
        )

        self.tokenizer.train_from_iterator([text], trainer=trainer)

        self.vocab_size = self.tokenizer.get_vocab_size()

    def encode(self, text: str) -> List[int]:
        encoding = self.tokenizer.encode(text)
        return encoding.ids

    def decode(self, token_ids: List[int]) -> str:
        return self.tokenizer.decode(token_ids)

    def save(self, path: Path | str) -> None:
        self.tokenizer.save(str(path))

    def load(self, path: Path | str) -> None:
        path_str = str(path)

        try:
            self.tokenizer = Tokenizer.from_file(path_str)
            self.vocab_size = self.tokenizer.get_vocab_size()
        except Exception:
            with open(path_str, "r", encoding="utf-8") as f:
                data = json.load(f)

            vocab: Dict[str, int] = {v: int(k) for k, v in data["vocab"].items()}
            merges: List[Tuple[str, str]] = [(item[0], item[1]) for item in data["merges"]]

            bpe_model = models.BPE(vocab=vocab, merges=merges, unk_token="<unk>")
            self.tokenizer = Tokenizer(bpe_model)

            self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

            self.tokenizer.decoder = decoders.ByteLevel()

            self.vocab_size = self.tokenizer.get_vocab_size()