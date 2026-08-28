import json
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple
from tqdm import tqdm


class BPETokenizer:
    def __init__(self, vocab_size: int) -> None:
        self.vocab_size: int = vocab_size
        self.vocab: Dict[int, str] = {}
        self.token_to_id: Dict[str, int] = {}
        self.merges: Dict[Tuple[str, str], str] = {}

    def build_initial_vocabulary(self, text: str) -> None:
        characters: List[str] = sorted(set(text))

        self.vocab = {token_id: character for token_id, character in enumerate(characters)}
        self.token_to_id = {token: token_id for token_id, token in self.vocab.items()}

    def train(self, text: str, max_words: int = 10000) -> None:
        self.build_initial_vocabulary(text)

        word_counts: Counter[str] = Counter(text.split())
        word_splits: Dict[Tuple[str, ...], int] = {
            tuple(list(w)): count for w, count in word_counts.most_common(max_words)
        }

        target_merges: int = max(0, self.vocab_size - len(self.vocab))
        pbar: tqdm = tqdm(total=target_merges, desc="Training Tokenizer", dynamic_ncols=True, leave=True)

        while len(self.vocab) < self.vocab_size:
            pair_counts: Counter[Tuple[str, str]] = Counter()

            for word_tuple, count in word_splits.items():
                for i in range(len(word_tuple) - 1):
                    pair_counts[(word_tuple[i], word_tuple[i + 1])] += count

            if not pair_counts:
                break

            most_common_pair, frequency = pair_counts.most_common(1)[0]

            if frequency < 2:
                break

            merged_token: str = most_common_pair[0] + most_common_pair[1]
            self.merges[most_common_pair] = merged_token

            new_token_id: int = len(self.vocab)
            self.vocab[new_token_id] = merged_token
            self.token_to_id[merged_token] = new_token_id

            new_word_splits: Dict[Tuple[str, ...], int] = {}

            for word_tuple, count in word_splits.items():
                new_word: List[str] = []
                idx: int = 0

                while idx < len(word_tuple):
                    if (
                        idx < len(word_tuple) - 1
                        and word_tuple[idx] == most_common_pair[0]
                        and word_tuple[idx + 1] == most_common_pair[1]
                    ):
                        new_word.append(merged_token)
                        idx += 2
                    else:
                        new_word.append(word_tuple[idx])
                        idx += 1

                new_word_splits[tuple(new_word)] = count

            word_splits = new_word_splits
            pbar.update(1)
            pbar.set_postfix_str(f"Vocab Size={len(self.vocab)}/{self.vocab_size}", refresh=False)

        pbar.close()

    def encode(self, text: str) -> List[int]:
        tokens: List[str] = list(text)

        for pair, merged_token in self.merges.items():
            merged: List[str] = []
            idx: int = 0

            while idx < len(tokens):
                if idx < len(tokens) - 1 and tokens[idx] == pair[0] and tokens[idx + 1] == pair[1]:
                    merged.append(merged_token)
                    idx += 2
                else:
                    merged.append(tokens[idx])
                    idx += 1

            tokens = merged

        return [self.token_to_id[token] for token in tokens if token in self.token_to_id]

    def decode(self, token_ids: List[int]) -> str:
        return "".join(self.vocab[token_id] for token_id in token_ids if token_id in self.vocab)

    def save(self, path: Path | str) -> None:
        data = {
            "vocab_size": self.vocab_size,
            "vocab": {str(k): v for k, v in self.vocab.items()},
            "merges": [[k[0], k[1], v] for k, v in self.merges.items()]
        }

        Path(path).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def load(self, path: Path | str) -> None:
        data = json.loads(Path(path).read_text(encoding="utf-8"))

        self.vocab_size = data["vocab_size"]
        self.vocab = {int(k): v for k, v in data["vocab"].items()}
        self.token_to_id = {v: k for k, v in self.vocab.items()}
        self.merges = {(item[0], item[1]): item[2] for item in data["merges"]}
            