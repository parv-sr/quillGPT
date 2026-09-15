import json
import random
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
from tqdm import tqdm

from config import Config
from data.bpe_tokenizer import BPETokenizer


CHUNK_CHARS = 4 * 1024 * 1024
ENCODE_BATCH_CHUNKS = 8
TOKEN_SHARD_SIZE = 1_000_000
VALIDATION_FRACTION = 0.001
RANDOM_SEED = 42


def iter_text_chunks(path: Path) -> Iterable[str]:
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


def encode_batch(tokenizer, texts: list[str]) -> list[np.ndarray]:
    if not texts:
        return []

    encode_fn = getattr(
        tokenizer,
        "encode_batch_fast",
        tokenizer.encode_batch,
    )

    encodings = encode_fn(
        texts,
        add_special_tokens=False,
    )

    arrays: list[np.ndarray] = []

    for encoding in encodings:
        if not encoding.ids:
            continue

        arrays.append(
            np.asarray(
                encoding.ids,
                dtype=np.uint16,
            )
        )

    return arrays


def flush_shard(
    token_buffer: np.ndarray,
    shard_dir: Path,
    shard_index: int,
) -> tuple[np.ndarray, int]:
    if token_buffer.size < TOKEN_SHARD_SIZE:
        return token_buffer, shard_index

    shard = token_buffer[:TOKEN_SHARD_SIZE]

    shard_path = shard_dir / f"shard_{shard_index:06d}.bin"

    shard.tofile(shard_path)

    remaining = token_buffer[TOKEN_SHARD_SIZE:].copy()

    return remaining, shard_index + 1


def build_token_shards(
    file_paths: list[Path],
    tokenizer,
    eos_id: int,
    shard_dir: Path,
) -> tuple[list[Path], int]:
    shard_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for old_shard in shard_dir.glob("*.bin"):
        old_shard.unlink()

    eos = np.asarray(
        [eos_id],
        dtype=np.uint16,
    )

    token_buffer = np.empty(
        0,
        dtype=np.uint16,
    )

    shard_index = 0
    total_tokens = 0

    progress = tqdm(
        file_paths,
        desc="Tokenizing corpus",
        unit="file",
        dynamic_ncols=True,
    )

    for path in progress:
        text_batch: list[str] = []

        for text_chunk in iter_text_chunks(path):
            text_batch.append(text_chunk)

            if len(text_batch) < ENCODE_BATCH_CHUNKS:
                continue

            arrays = encode_batch(
                tokenizer,
                text_batch,
            )

            text_batch.clear()

            if arrays:
                arrays.append(eos[:0])

                new_tokens = np.concatenate(arrays)

                token_buffer = np.concatenate(
                    (
                        token_buffer,
                        new_tokens,
                    )
                )

                total_tokens += new_tokens.size

                while token_buffer.size >= TOKEN_SHARD_SIZE:
                    token_buffer, shard_index = flush_shard(
                        token_buffer,
                        shard_dir,
                        shard_index,
                    )

        if text_batch:
            arrays = encode_batch(
                tokenizer,
                text_batch,
            )

            if arrays:
                new_tokens = np.concatenate(arrays)

                token_buffer = np.concatenate(
                    (
                        token_buffer,
                        new_tokens,
                    )
                )

                total_tokens += new_tokens.size

                while token_buffer.size >= TOKEN_SHARD_SIZE:
                    token_buffer, shard_index = flush_shard(
                        token_buffer,
                        shard_dir,
                        shard_index,
                    )

        token_buffer = np.concatenate(
            (
                token_buffer,
                eos,
            )
        )

        total_tokens += 1

        while token_buffer.size >= TOKEN_SHARD_SIZE:
            token_buffer, shard_index = flush_shard(
                token_buffer,
                shard_dir,
                shard_index,
            )

        progress.set_postfix_str(
            f"{total_tokens / 1e9:.3f}B tokens",
            refresh=False,
        )

    if token_buffer.size:
        shard_path = shard_dir / f"shard_{shard_index:06d}.bin"

        token_buffer.tofile(shard_path)

        shard_index += 1

    shards = sorted(
        shard_dir.glob("shard_*.bin")
    )

    return shards, total_tokens


def split_shards(
    shards: list[Path],
    validation_fraction: float,
    seed: int,
) -> tuple[list[Path], list[Path]]:
    if len(shards) < 2:
        raise RuntimeError(
            "Not enough token shards to construct "
            "train and validation sets."
        )

    shuffled = list(shards)

    rng = random.Random(seed)
    rng.shuffle(shuffled)

    validation_count = max(
        1,
        round(
            len(shuffled)
            * validation_fraction
        ),
    )

    validation_shards = shuffled[:validation_count]
    train_shards = shuffled[validation_count:]

    if not train_shards:
        train_shards.append(
            validation_shards.pop()
        )

    rng.shuffle(train_shards)
    rng.shuffle(validation_shards)

    return train_shards, validation_shards


def concatenate_shards(
    shards: list[Path],
    output_path: Path,
    description: str,
) -> int:
    total_tokens = 0

    with output_path.open("wb") as output_file:
        progress = tqdm(
            shards,
            desc=description,
            unit="shard",
            dynamic_ncols=True,
        )

        for shard_path in progress:
            shard = np.memmap(
                shard_path,
                dtype=np.uint16,
                mode="r",
            )

            shard.tofile(output_file)

            total_tokens += len(shard)

            del shard

            progress.set_postfix_str(
                f"{total_tokens / 1e9:.3f}B tokens",
                refresh=False,
            )

    return total_tokens


def prepare_corpus() -> None:
    config = Config()

    cleaned_dir = Path("data/cleaned")
    tokens_dir = Path("data/tokens")
    shard_dir = tokens_dir / "shards_tmp"

    tokens_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_paths = sorted(
        cleaned_dir.rglob("*.txt")
    )

    if not file_paths:
        raise FileNotFoundError(
            f"No .txt files found in {cleaned_dir}"
        )

    tokenizer_path = Path(
        f"bpe_tokenizer_{config.vocab_size}.json"
    )

    if not tokenizer_path.exists():
        raise FileNotFoundError(
            f"{tokenizer_path} does not exist."
        )

    tokenizer_wrapper = BPETokenizer(
        config.vocab_size
    )

    tokenizer_wrapper.load(
        tokenizer_path
    )

    tokenizer = tokenizer_wrapper.tokenizer

    eos_id = tokenizer.token_to_id("<eos>")

    if eos_id is None:
        raise RuntimeError(
            "<eos> does not exist in the tokenizer."
        )

    if (
        tokenizer.get_vocab_size()
        > np.iinfo(np.uint16).max
    ):
        raise ValueError(
            "Vocabulary is too large for uint16."
        )

    print(f"Cleaned files: {len(file_paths):,}")
    print(f"Tokenizer:     {tokenizer_path}")
    print(f"EOS ID:        {eos_id}")
    print(
        f"Shard size:    "
        f"{TOKEN_SHARD_SIZE:,} tokens"
    )

    shards, raw_token_count = build_token_shards(
        file_paths,
        tokenizer,
        eos_id,
        shard_dir,
    )

    print()
    print(f"Temporary shards: {len(shards):,}")
    print(f"Raw tokens:       {raw_token_count:,}")

    train_shards, validation_shards = split_shards(
        shards,
        VALIDATION_FRACTION,
        RANDOM_SEED,
    )

    train_bin_path = (
        tokens_dir / "train.bin"
    )

    validation_bin_path = (
        tokens_dir / "validation.bin"
    )

    train_token_count = concatenate_shards(
        train_shards,
        train_bin_path,
        "Building train.bin",
    )

    validation_token_count = concatenate_shards(
        validation_shards,
        validation_bin_path,
        "Building validation.bin",
    )

    metadata = {
        "train_tokens": int(
            train_token_count
        ),
        "validation_tokens": int(
            validation_token_count
        ),
        "total_tokens": int(
            train_token_count
            + validation_token_count
        ),
        "cleaned_files": len(
            file_paths
        ),
        "train_shards": len(
            train_shards
        ),
        "validation_shards": len(
            validation_shards
        ),
        "token_shard_size": (
            TOKEN_SHARD_SIZE
        ),
        "validation_fraction": (
            VALIDATION_FRACTION
        ),
        "vocab_size": (
            tokenizer.get_vocab_size()
        ),
        "dtype": "uint16",
        "eos_id": eos_id,
        "seed": RANDOM_SEED,
    }

    meta_path = (
        tokens_dir / "meta.json"
    )

    with meta_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    print()
    print(
        f"Train tokens:      "
        f"{train_token_count:,}"
    )
    print(
        f"Validation tokens: "
        f"{validation_token_count:,}"
    )
    print(
        f"Train binary:      "
        f"{train_bin_path.stat().st_size / (1024 ** 3):.2f} GiB"
    )
    print(
        f"Validation binary: "
        f"{validation_bin_path.stat().st_size / (1024 ** 3):.2f} GiB"
    )

    shutil.rmtree(
        shard_dir,
        ignore_errors=True,
    )

    print("Temporary shards removed.")


if __name__ == "__main__":
    prepare_corpus()