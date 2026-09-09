import ctypes
import gc
import json
import random
from pathlib import Path
import numpy as np
from config import Config
from data.bpe_tokenizer import BPETokenizer


def trim_memory() -> None:
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def prepare_corpus() -> None:
    config = Config()
    cleaned_dir = Path("data/cleaned")
    tokens_dir = Path("data/tokens")
    tokens_dir.mkdir(parents=True, exist_ok=True)

    file_paths = sorted([str(p) for p in cleaned_dir.rglob("*.txt")])
    if not file_paths:
        file_paths = sorted([str(p) for p in cleaned_dir.glob("*.txt")])
    if not file_paths:
        raise FileNotFoundError(f"No .txt files found in {cleaned_dir}")

    tokenizer = BPETokenizer(config.vocab_size)
    tokenizer_path = f"bpe_tokenizer_{config.vocab_size}.json"

    if Path(tokenizer_path).exists():
        tokenizer.load(tokenizer_path)
    else:
        tokenizer.train_from_files(file_paths)
        tokenizer.save(tokenizer_path)

    rng = random.Random(42)
    shuffled_paths = list(file_paths)
    rng.shuffle(shuffled_paths)

    num_files = len(shuffled_paths)
    num_train = max(1, int(num_files * 0.8))
    train_files = shuffled_paths[:num_train]
    val_files = shuffled_paths[num_train:]
    if not val_files and len(train_files) > 1:
        val_files = train_files[-1:]
        train_files = train_files[:-1]

    train_bin_path = tokens_dir / "train.bin"
    val_bin_path = tokens_dir / "validation.bin"

    batch_size = 1024

    train_token_count = 0
    with open(train_bin_path, "wb") as f_train:
        for i in range(0, len(train_files), batch_size):
            block_paths = train_files[i : i + batch_size]
            texts = []
            for fp in block_paths:
                with open(fp, "r", encoding="utf-8", errors="ignore") as f_in:
                    texts.append(f_in.read())
            encodings = tokenizer.tokenizer.encode_batch(texts)
            del texts
            block_ids = [enc.ids for enc in encodings if enc.ids]
            del encodings
            if block_ids:
                flat_ids = [id_val for enc_ids in block_ids for id_val in enc_ids]
                del block_ids
                arr = np.array(flat_ids, dtype=np.uint16)
                del flat_ids
                f_train.write(arr.tobytes())
                train_token_count += int(len(arr))
                del arr
            f_train.flush()
            trim_memory()

    val_token_count = 0
    with open(val_bin_path, "wb") as f_val:
        for i in range(0, len(val_files), batch_size):
            block_paths = val_files[i : i + batch_size]
            texts = []
            for fp in block_paths:
                with open(fp, "r", encoding="utf-8", errors="ignore") as f_in:
                    texts.append(f_in.read())
            encodings = tokenizer.tokenizer.encode_batch(texts)
            del texts
            block_ids = [enc.ids for enc in encodings if enc.ids]
            del encodings
            if block_ids:
                flat_ids = [id_val for enc_ids in block_ids for id_val in enc_ids]
                del block_ids
                arr = np.array(flat_ids, dtype=np.uint16)
                del flat_ids
                f_val.write(arr.tobytes())
                val_token_count += int(len(arr))
                del arr
            f_val.flush()
            trim_memory()

    metadata = {
        "train_tokens": train_token_count,
        "validation_tokens": val_token_count,
        "train_files": len(train_files),
        "validation_files": len(val_files),
        "vocab_size": config.vocab_size,
        "dtype": "uint16",
    }

    meta_path = tokens_dir / "meta.json"
    with open(meta_path, "w", encoding="utf-8") as f_meta:
        json.dump(metadata, f_meta, indent=2)


if __name__ == "__main__":
    prepare_corpus()
