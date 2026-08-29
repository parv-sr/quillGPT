from pathlib import Path
from typing import Iterable, Iterator, List, Union
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, normalizers


class BPETokenizer:
    SPECIAL_TOKENS = ["<unk>", "<pad>", "<bos>", "<eos>"]

    def __init__(self, vocab_size: int = 30000) -> None:
        self.target_vocab_size: int = vocab_size
        self._init_tokenizer()

    def _init_tokenizer(self) -> None:
        self.tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

        self.tokenizer.normalizer = normalizers.NFKC()

        self.tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Digits(individual_digits=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
        ])

        self.tokenizer.decoder = decoders.ByteLevel()

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    def train(self, input_data: Union[str, List[str], Path, Iterable[str]], min_frequency: int = 2) -> None:
        """
        Universal training method expected by train.py.
        Handles string text, lists of texts, generators, or path instances.
        """
        trainer = trainers.BpeTrainer(
            vocab_size=self.target_vocab_size,
            min_frequency=min_frequency,
            show_progress=True,
            special_tokens=self.SPECIAL_TOKENS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
        )

        if isinstance(input_data, (str, Path)) and Path(input_data).is_file():
            # If a single file path is passed
            self.tokenizer.train(files=[str(input_data)], trainer=trainer)
        elif isinstance(input_data, list) and all(isinstance(x, (str, Path)) and Path(x).is_file() for x in input_data):
            # If a list of file paths is passed
            self.tokenizer.train(files=[str(p) for p in input_data], trainer=trainer)
        elif isinstance(input_data, str):
            # If raw corpus text string is passed
            self.tokenizer.train_from_iterator([input_data], trainer=trainer)
        else:
            # If an iterable/generator/list of text chunks is passed
            self.tokenizer.train_from_iterator(input_data, trainer=trainer)

    def train_from_files(self, file_paths: List[Union[str, Path]], min_frequency: int = 2) -> None:
        """Fast Rust-native training directly from a list of text files."""
        self.train(input_data=file_paths, min_frequency=min_frequency)

    def train_from_iterator(self, iterator: Iterator[str], min_frequency: int = 2) -> None:
        """Train from a memory-efficient python generator yielding chunks/lines."""
        self.train(input_data=iterator, min_frequency=min_frequency)

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        encoding = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return encoding.ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def save(self, path: Union[Path, str]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(str(path))

    def load(self, path: Union[Path, str]) -> None:
        self.tokenizer = Tokenizer.from_file(str(path))