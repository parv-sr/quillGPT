import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np
from tqdm import tqdm

from config import Config
from data.bpe_tokenizer import BPETokenizer


CHUNK_CHARS = 4 * 1024 * 1024
ENCODE_BATCH_CHUNKS = 8

VALIDATION_FRACTION = 0.001

RANDOM_SEED = 42


def iter_text_chunks(path: Path) -> Iterable[str]:
    """
    Yield bounded-size chunks, preferentially ending on line boundaries.

    This avoids reading an entire potentially huge file into RAM.
    """
    buffer: list[str] = []
    buffered_chars = 0

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            buffer.append(line)
            buffered_chars += len(line)

            if buffered_chars >= CHUNK_CHARS:
                chunk = "".join(buffer)
                buffer = []
                buffered_chars = 0

                if chunk:
                    yield chunk

        if buffer:
            chunk = "".join(buffer)
            if chunk:
                yield chunk


def split_files_by_size(
    file_paths: list[Path],
    validation_fraction: float,
    seed: int,
) -> tuple[list[Path], list[Path]]:
    """
    File-level train/validation split, but approximately by bytes instead
    of file count. This handles corpora where files have very different sizes.
    """
    if len(file_paths) < 2:
        raise ValueError(
            "Need at least two cleaned files for a file-level "
            "train/validation split."
        )

    paths = list(file_paths)

    rng = random.Random(seed)
    rng.shuffle(paths)

    total_bytes = sum(p.stat().st_size for p in paths)
    target_val_bytes = max(1, int(total_bytes * validation_fraction))

    val_files: list[Path] = []
    train_files: list[Path] = []

    val_bytes = 0

    for i, path in enumerate(paths):
        files_remaining = len(paths) - i

        if val_bytes < target_val_bytes and files_remaining > 1:
            val_files.append(path)
            val_bytes += path.stat().st_size
        else:
            train_files.append(path)

    if not train_files:
        train_files.append(val_files.pop())

    if not val_files:
        val_files.append(train_files.pop())

    return train_files, val_files


def encode_and_write_batch(
    tokenizer,
    texts: list[str],
    output_file,
) -> int:
    """
    Tokenize a bounded batch and immediately stream uint16 IDs to disk.
    """
    if not texts:
        return 0
    
    encode_batch = getattr(
        tokenizer,
        "encode_batch_fast",
        tokenizer.encode_batch,
    )

    encodings = encode_batch(
        texts,
        add_special_tokens=False,
    )
    written = 0

    for encoding in encodings:
        ids = encoding.ids

        if not ids:
            continue

        arr = np.asarray(ids, dtype=np.uint16)
        arr.tofile(output_file)
        written += arr.size

    return written


def write_token_file(
    file_paths: list[Path],
    output_path: Path,
    tokenizer,
    eos_id: int,
    description: str,
) -> int:
    total_tokens = 0

    eos = np.asarray([eos_id], dtype=np.uint16)

    with output_path.open("wb") as output_file:
        progress = tqdm(
            file_paths,
            desc=description,
            unit="file",
            dynamic_ncols=True,
        )

        for path in progress:
            text_batch: list[str] = []

            for chunk in iter_text_chunks(path):
                text_batch.append(chunk)

                if len(text_batch) >= ENCODE_BATCH_CHUNKS:
                    total_tokens += encode_and_write_batch(
                        tokenizer,
                        text_batch,
                        output_file,
                    )

                    text_batch.clear()

            if text_batch:
                total_tokens += encode_and_write_batch(
                    tokenizer,
                    text_batch,
                    output_file,
                )

            eos.tofile(output_file)
            total_tokens += 1

            progress.set_postfix_str(
                f"{total_tokens / 1e9:.3f}B tokens",
                refresh=False,
            )

    return total_tokens


def prepare_corpus() -> None:
    config = Config()

    cleaned_dir = Path("data/cleaned")
    tokens_dir = Path("data/tokens")
    tokens_dir.mkdir(parents=True, exist_ok=True)

    file_paths = sorted(cleaned_dir.rglob("*.txt"))

    if not file_paths:
        raise FileNotFoundError(
            f"No .txt files found in {cleaned_dir}"
        )

    tokenizer_path = Path(
        f"bpe_tokenizer_{config.vocab_size}.json"
    )

    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"{tokenizer_path} does not exist. "
            "Corpus preparation will not retrain the tokenizer automatically."
        )

    tokenizer_wrapper = BPETokenizer(config.vocab_size)
    tokenizer_wrapper.load(tokenizer_path)

    tokenizer = tokenizer_wrapper.tokenizer

    eos_id = tokenizer.token_to_id("<eos>")

    if eos_id is None:
        raise RuntimeError(
            "<eos> does not exist in the loaded tokenizer."
        )

    if tokenizer.get_vocab_size() > np.iinfo(np.uint16).max:
        raise ValueError(
            "Vocabulary is too large for uint16 token storage."
        )

    train_files, val_files = split_files_by_size(
        file_paths,
        validation_fraction=VALIDATION_FRACTION,
        seed=RANDOM_SEED,
    )

    train_bin_path = tokens_dir / "train.bin"
    val_bin_path = tokens_dir / "validation.bin"

    print(f"Cleaned files: {len(file_paths):,}")
    print(f"Train files:   {len(train_files):,}")
    print(f"Val files:     {len(val_files):,}")
    print(f"Tokenizer:     {tokenizer_path}")
    print(f"EOS ID:        {eos_id}")

    train_token_count = write_token_file(
        train_files,
        train_bin_path,
        tokenizer,
        eos_id,
        "Tokenizing train",
    )

    val_token_count = write_token_file(
        val_files,
        val_bin_path,
        tokenizer,
        eos_id,
        "Tokenizing validation",
    )

    metadata = {
        "train_tokens": int(train_token_count),
        "validation_tokens": int(val_token_count),
        "train_files": len(train_files),
        "validation_files": len(val_files),
        "vocab_size": tokenizer.get_vocab_size(),
        "dtype": "uint16",
        "validation_fraction": VALIDATION_FRACTION,
        "eos_id": eos_id,
        "seed": RANDOM_SEED,
    }

    meta_path = tokens_dir / "meta.json"

    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print()
    print(f"Train tokens:      {train_token_count:,}")
    print(f"Validation tokens: {val_token_count:,}")
    print(
        f"Train binary:      "
        f"{train_bin_path.stat().st_size / (1024**3):.2f} GiB"
    )
    print(
        f"Validation binary: "
        f"{val_bin_path.stat().st_size / (1024**3):.2f} GiB"
    )


if __name__ == "__main__":
    prepare_corpus()