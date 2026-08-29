from pathlib import Path
from collections import Counter
from typing import Any, Dict, List, Tuple
import json
import math
import re
import statistics
import time

import numpy as np
import onnxruntime as ort

from config import Config
from data.corpus import TextCorpus
from data.bpe_tokenizer import BPETokenizer


MODEL_PATH = Path(r"/home/parv/quillGPT/quillGPT/artifacts/models/quillGPT_v0.1.3.onnx")
CORPUS_PATH = Path("data/raw")
TOKENIZER_PATH = Path("bpe_tokenizer.json")


class DiagnosticReport:
    def __init__(self) -> None:
        self.results: Dict[str, Any] = {}
        self.failures: List[str] = []
        self.warnings: List[str] = []

    def add(self, name: str, result: Any) -> None:
        self.results[name] = result

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def fail(self, message: str) -> None:
        self.failures.append(message)

    def section(self, title: str) -> None:
        print()
        print("=" * 80)
        print(title)
        print("=" * 80)

    def print_summary(self) -> None:
        self.section("DIAGNOSTIC SUMMARY")

        print(f"Tests completed: {len(self.results)}")
        print(f"Warnings: {len(self.warnings)}")
        print(f"Failures: {len(self.failures)}")

        if self.warnings:
            print("\nWarnings:")
            for warning in self.warnings:
                print(f"  - {warning}")

        if self.failures:
            print("\nFailures:")
            for failure in self.failures:
                print(f"  - {failure}")


class ONNXDiagnosticEngine:
    def __init__(self, model_path: Path) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.model_path = model_path
        self.session = ort.InferenceSession(str(model_path), providers=self._providers())

        self.inputs = self.session.get_inputs()
        self.outputs = self.session.get_outputs()

        self.input_name = self.inputs[0].name
        self.output_name = self.outputs[0].name

    @staticmethod
    def _providers() -> List[str]:
        available = ort.get_available_providers()

        if "CUDAExecutionProvider" in available:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]

        return ["CPUExecutionProvider"]

    @property
    def providers(self) -> List[str]:
        return self.session.get_providers()

    def predict(self, tokens: np.ndarray) -> np.ndarray:
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape (batch, sequence)")

        tokens = np.asarray(tokens, dtype=np.int64)

        outputs = self.session.run(
            [self.output_name],
            {self.input_name: tokens}
        )

        return outputs[0]


class TokenizerDiagnostics:
    def __init__(self, tokenizer: BPETokenizer, corpus: str, vocab_size: int) -> None:
        self.tokenizer = tokenizer
        self.corpus = corpus
        self.vocab_size = vocab_size

    def run(self, report: DiagnosticReport) -> None:
        report.section("TOKENIZER DIAGNOSTICS")

        tokenizer_vocab_size = self.tokenizer.tokenizer.get_vocab_size()

        report.add(
            "tokenizer_vocab_size",
            {
                "configured": self.vocab_size,
                "actual": tokenizer_vocab_size,
                "difference": tokenizer_vocab_size - self.vocab_size
            }
        )

        samples = [
            "Hello, world!",
            "This is a test sentence.",
            "Mr. Darcy said, \"I am not convinced.\"",
            "The quick brown fox jumps over the lazy dog.",
            "There must not be any violence, for it is a sin to be wrathful.",
            "My father’s family name being Pirrip, and my christian name Philip.",
            "One\nTwo\nThree",
            "It's wasn't couldn't shouldn't.",
            "1234567890",
            "— em dash — en dash… ellipsis",
        ]

        roundtrip_results: List[Dict[str, Any]] = []

        for text in samples:
            ids = self.tokenizer.encode(text)
            decoded = self.tokenizer.decode(ids)

            roundtrip_results.append(
                {
                    "input": text,
                    "token_count": len(ids),
                    "ids": ids[:30],
                    "decoded": decoded,
                    "exact": decoded == text
                }
            )

            if decoded != text:
                report.warn(f"Tokenizer round-trip mismatch: {text!r}")

        report.add("roundtrip_tests", roundtrip_results)

        encoded = self.tokenizer.encode(self.corpus)

        if not encoded:
            report.fail("Tokenizer produced zero tokens for the corpus.")
            return

        token_counts = Counter(encoded)

        report.add(
            "corpus_token_statistics",
            {
                "characters": len(self.corpus),
                "tokens": len(encoded),
                "compression_ratio_chars_per_token": len(self.corpus) / len(encoded),
                "unique_tokens_used": len(token_counts),
                "vocabulary_utilization": len(token_counts) / tokenizer_vocab_size,
                "most_common_tokens": token_counts.most_common(25),
            }
        )

        token_lengths: List[int] = []

        for token_id in token_counts:
            token = self.tokenizer.tokenizer.id_to_token(token_id)
            if token is not None:
                token_lengths.append(len(token))

        report.add(
            "token_length_statistics",
            {
                "mean": statistics.mean(token_lengths),
                "median": statistics.median(token_lengths),
                "min": min(token_lengths),
                "max": max(token_lengths),
            }
        )

        suspicious_tokens: List[Tuple[int, str, int]] = []

        for token_id, count in token_counts.most_common():
            token = self.tokenizer.tokenizer.id_to_token(token_id)

            if token is None:
                continue

            if self._is_suspicious(token):
                suspicious_tokens.append((token_id, token, count))

        report.add(
            "suspicious_tokens",
            suspicious_tokens[:100]
        )

        if suspicious_tokens:
            report.warn(
                f"{len(suspicious_tokens)} frequently used suspicious tokens detected."
            )

        single_character_tokens: List[Tuple[int, str, int]] = []

        for token_id, count in token_counts.items():
            token = self.tokenizer.tokenizer.id_to_token(token_id)

            if token is not None and len(token) == 1:
                single_character_tokens.append((token_id, token, count))

        report.add(
            "single_character_tokens",
            sorted(single_character_tokens, key=lambda x: x[2], reverse=True)[:100]
        )

        whitespace_tokens = []

        for token_id, count in token_counts.items():
            token = self.tokenizer.tokenizer.id_to_token(token_id)

            if token is not None and token.isspace():
                whitespace_tokens.append((token_id, repr(token), count))

        report.add(
            "whitespace_tokens",
            sorted(whitespace_tokens, key=lambda x: x[2], reverse=True)
        )

        special_character_counts = Counter(
            character
            for character in self.corpus
            if not character.isalnum() and not character.isspace()
        )

        report.add(
            "special_character_frequency",
            special_character_counts.most_common(100)
        )

        words = re.findall(r"\b[\w’'-]+\b", self.corpus)

        word_lengths = [len(word) for word in words]

        report.add(
            "word_statistics",
            {
                "word_count": len(words),
                "unique_words": len(set(words)),
                "mean_word_length": statistics.mean(word_lengths),
                "median_word_length": statistics.median(word_lengths),
                "max_word_length": max(word_lengths),
            }
        )

        probe_text = self.corpus[:1_000_000]
        probe_tokens = self.tokenizer.encode(probe_text)

        report.add(
            "tokenization_throughput",
            {
                "characters": len(probe_text),
                "tokens": len(probe_tokens),
            }
        )

    @staticmethod
    def _is_suspicious(token: str) -> bool:
        if not token:
            return True

        if len(token) == 1 and not token.isalnum() and not token.isspace():
            return True

        unicode_categories = {
            "Cf",
            "Cc",
            "Cs",
            "Co",
            "Cn",
        }

        import unicodedata

        for character in token:
            if unicodedata.category(character) in unicode_categories:
                return True

        return False


class DatasetDiagnostics:
    def __init__(self, corpus: str, tokenizer: BPETokenizer, max_context: int) -> None:
        self.corpus = corpus
        self.tokenizer = tokenizer
        self.max_context = max_context

    def run(self, report: DiagnosticReport) -> None:
        report.section("DATASET DIAGNOSTICS")

        corpus_size = len(self.corpus)

        lines = self.corpus.splitlines()

        blank_lines = sum(1 for line in lines if not line.strip())

        very_short_lines = sum(
            1 for line in lines
            if line.strip() and len(line.strip()) < 20
        )

        boilerplate_patterns = [
            "Project Gutenberg",
            "START OF THE PROJECT GUTENBERG",
            "END OF THE PROJECT GUTENBERG",
            "www.gutenberg.org",
            "gutenberg.org",
        ]

        boilerplate_hits = {
            pattern: self.corpus.lower().count(pattern.lower())
            for pattern in boilerplate_patterns
        }

        replacement_characters = self.corpus.count("\ufffd")

        report.add(
            "corpus_statistics",
            {
                "characters": corpus_size,
                "megabytes_utf8_estimate": len(self.corpus.encode("utf-8")) / 1_000_000,
                "lines": len(lines),
                "blank_lines": blank_lines,
                "very_short_lines": very_short_lines,
                "replacement_characters": replacement_characters,
                "boilerplate_hits": boilerplate_hits,
            }
        )

        if replacement_characters > 0:
            report.warn(
                f"Corpus contains {replacement_characters} Unicode replacement characters."
            )

        report.add(
            "character_distribution",
            Counter(self.corpus).most_common(100)
        )

        token_ids = self.tokenizer.encode(self.corpus)

        split_index = int(len(token_ids) * 0.9)

        train_tokens = token_ids[:split_index]
        validation_tokens = token_ids[split_index:]

        report.add(
            "train_validation_split",
            {
                "train_tokens": len(train_tokens),
                "validation_tokens": len(validation_tokens),
                "train_fraction": len(train_tokens) / len(token_ids),
                "validation_fraction": len(validation_tokens) / len(token_ids),
            }
        )

        train_counts = Counter(train_tokens)
        validation_counts = Counter(validation_tokens)

        train_vocab = set(train_counts)
        validation_vocab = set(validation_counts)

        unseen_validation_tokens = validation_vocab - train_vocab

        report.add(
            "validation_vocabulary_analysis",
            {
                "train_unique_tokens": len(train_vocab),
                "validation_unique_tokens": len(validation_vocab),
                "validation_tokens_unseen_in_train": len(unseen_validation_tokens),
                "unseen_fraction": (
                    len(unseen_validation_tokens) / len(validation_vocab)
                    if validation_vocab else 0.0
                ),
            }
        )

        windows = max(1, len(token_ids) // self.max_context)

        report.add(
            "context_window_analysis",
            {
                "max_context": self.max_context,
                "approximate_full_windows": windows,
                "tokens_per_window": self.max_context,
                "tokens_lost_to_remainder": len(token_ids) % self.max_context,
            }
        )


class ModelDiagnostics:
    def __init__(self, engine: ONNXDiagnosticEngine, config: Config) -> None:
        self.engine = engine
        self.config = config

    def run(self, report: DiagnosticReport) -> None:
        report.section("MODEL / ONNX DIAGNOSTICS")

        input_metadata = [
            {
                "name": value.name,
                "shape": value.shape,
                "type": value.type
            }
            for value in self.engine.inputs
        ]

        output_metadata = [
            {
                "name": value.name,
                "shape": value.shape,
                "type": value.type
            }
            for value in self.engine.outputs
        ]

        report.add(
            "onnx_metadata",
            {
                "providers": self.engine.providers,
                "inputs": input_metadata,
                "outputs": output_metadata,
            }
        )

        parameter_estimate = (
            self.config.vocab_size * self.config.embed_dim
            + self.config.max_context * self.config.embed_dim
            + self.config.num_layers * (
                4 * self.config.embed_dim * self.config.embed_dim
                + 2 * self.config.embed_dim
                + 2 * self.config.embed_dim * self.config.feedforward_dim
            )
            + self.config.vocab_size * self.config.embed_dim
        )

        report.add(
            "parameter_estimate",
            {
                "estimated_parameters": parameter_estimate,
                "estimated_parameters_millions": parameter_estimate / 1_000_000,
                "configuration": {
                    "vocab_size": self.config.vocab_size,
                    "embed_dim": self.config.embed_dim,
                    "num_heads": self.config.num_heads,
                    "num_layers": self.config.num_layers,
                    "feedforward_dim": self.config.feedforward_dim,
                    "max_context": self.config.max_context,
                }
            }
        )

        self._shape_tests(report)
        self._determinism_test(report)
        self._sensitivity_test(report)
        self._throughput_test(report)

    def _shape_tests(self, report: DiagnosticReport) -> None:
        tests = [1, 2, 8, 32, 128, self.config.max_context]

        results = []

        for sequence_length in tests:
            if sequence_length > self.config.max_context:
                continue

            tokens = np.random.randint(
                0,
                self.config.vocab_size,
                size=(1, sequence_length),
                dtype=np.int64
            )

            start = time.perf_counter()
            logits = self.engine.predict(tokens)
            elapsed = time.perf_counter() - start

            expected_shape = (1, sequence_length, self.config.vocab_size)

            results.append(
                {
                    "sequence_length": sequence_length,
                    "input_shape": list(tokens.shape),
                    "output_shape": list(logits.shape),
                    "expected_shape": list(expected_shape),
                    "correct": logits.shape == expected_shape,
                    "latency_ms": elapsed * 1000,
                }
            )

        report.add("shape_tests", results)

    def _determinism_test(self, report: DiagnosticReport) -> None:
        tokens = np.random.randint(
            0,
            self.config.vocab_size,
            size=(1, min(128, self.config.max_context)),
            dtype=np.int64
        )

        first = self.engine.predict(tokens)
        second = self.engine.predict(tokens)

        difference = np.max(np.abs(first - second))

        report.add(
            "determinism_test",
            {
                "maximum_difference": float(difference),
                "deterministic": bool(np.allclose(first, second, rtol=1e-5, atol=1e-6)),
            }
        )

    def _sensitivity_test(self, report: DiagnosticReport) -> None:
        length = min(128, self.config.max_context)

        tokens_a = np.random.randint(
            0,
            self.config.vocab_size,
            size=(1, length),
            dtype=np.int64
        )

        tokens_b = tokens_a.copy()
        tokens_b[0, length // 2] = (tokens_b[0, length // 2] + 1) % self.config.vocab_size

        logits_a = self.engine.predict(tokens_a)
        logits_b = self.engine.predict(tokens_b)

        difference = np.abs(logits_a - logits_b)

        report.add(
            "input_sensitivity_test",
            {
                "maximum_difference": float(np.max(difference)),
                "mean_difference": float(np.mean(difference)),
                "changed_position": length // 2,
            }
        )

    def _throughput_test(self, report: DiagnosticReport) -> None:
        sequence_length = min(self.config.max_context, 512)

        tokens = np.random.randint(
            0,
            self.config.vocab_size,
            size=(1, sequence_length),
            dtype=np.int64
        )

        warmup_runs = 3
        measured_runs = 10

        for _ in range(warmup_runs):
            self.engine.predict(tokens)

        timings = []

        for _ in range(measured_runs):
            start = time.perf_counter()
            self.engine.predict(tokens)
            timings.append(time.perf_counter() - start)

        mean_latency = statistics.mean(timings)

        report.add(
            "onnx_latency",
            {
                "sequence_length": sequence_length,
                "runs": measured_runs,
                "mean_ms": mean_latency * 1000,
                "median_ms": statistics.median(timings) * 1000,
                "min_ms": min(timings) * 1000,
                "max_ms": max(timings) * 1000,
                "tokens_per_second": sequence_length / mean_latency,
            }
        )


class GenerationDiagnostics:
    def __init__(self, engine: ONNXDiagnosticEngine, tokenizer: BPETokenizer, config: Config) -> None:
        self.engine = engine
        self.tokenizer = tokenizer
        self.config = config

    def run(self, report: DiagnosticReport) -> None:
        report.section("GENERATION DIAGNOSTICS")

        prompts = [
            "My father’s family name being Pirrip, and my christian name Philip, my infant tongue",
            "There must not be any violence, for it is a sin to be wrathful. Instead we must",
            "It was a dark and stormy night",
            "The object of my study was to understand",
            "In the beginning",
            "The old man looked at the house and",
        ]

        temperatures = [0.2, 0.5, 0.8, 1.0]

        results = []

        for prompt in prompts:
            prompt_result = {
                "prompt": prompt,
                "temperatures": {}
            }

            prompt_tokens = self.tokenizer.encode(prompt)

            for temperature in temperatures:
                generated_ids = list(prompt_tokens)

                start = time.perf_counter()

                for _ in range(100):
                    context = generated_ids[-self.config.max_context:]

                    tokens = np.asarray([context], dtype=np.int64)

                    logits = self.engine.predict(tokens)
                    next_logits = logits[0, -1]

                    next_token = self._sample(next_logits, temperature)
                    generated_ids.append(next_token)

                elapsed = time.perf_counter() - start

                generated_text = self.tokenizer.decode(generated_ids)

                prompt_result["temperatures"][str(temperature)] = {
                    "text": generated_text,
                    "generation_seconds": elapsed,
                    "tokens_per_second": 100 / elapsed,
                    "repetition_score": self._repetition_score(generated_ids),
                    "unique_token_ratio": len(set(generated_ids)) / len(generated_ids),
                }

            results.append(prompt_result)

        report.add("generation_tests", results)

    @staticmethod
    def _sample(logits: np.ndarray, temperature: float) -> int:
        if temperature <= 0:
            return int(np.argmax(logits))

        scaled = logits / temperature
        scaled -= np.max(scaled)

        probabilities = np.exp(scaled)
        probabilities /= probabilities.sum()

        return int(np.random.choice(len(probabilities), p=probabilities))

    @staticmethod
    def _repetition_score(token_ids: List[int], n: int = 3) -> float:
        if len(token_ids) < n:
            return 0.0

        grams = [
            tuple(token_ids[index:index + n])
            for index in range(len(token_ids) - n + 1)
        ]

        return 1.0 - len(set(grams)) / len(grams)


class LossDiagnostics:
    def __init__(self, engine: ONNXDiagnosticEngine, tokenizer: BPETokenizer, corpus: str, config: Config) -> None:
        self.engine = engine
        self.tokenizer = tokenizer
        self.corpus = corpus
        self.config = config

    def run(self, report: DiagnosticReport) -> None:
        report.section("EMPIRICAL LOSS DIAGNOSTICS")

        tokens = self.tokenizer.encode(self.corpus)

        if len(tokens) < self.config.max_context + 2:
            report.fail("Corpus is too small for loss diagnostics.")
            return

        train_end = int(len(tokens) * 0.9)

        train_tokens = tokens[:train_end]
        validation_tokens = tokens[train_end:]

        train_loss = self._estimate_loss(train_tokens)
        validation_loss = self._estimate_loss(validation_tokens)

        report.add(
            "estimated_loss",
            {
                "train_loss": train_loss,
                "validation_loss": validation_loss,
                "train_perplexity": math.exp(min(train_loss, 20)),
                "validation_perplexity": math.exp(min(validation_loss, 20)),
                "generalization_gap": validation_loss - train_loss,
            }
        )

    def _estimate_loss(self, tokens: List[int], max_batches: int = 32) -> float:
        losses = []

        stride = self.config.max_context

        maximum_start = len(tokens) - self.config.max_context - 1

        if maximum_start <= 0:
            return float("nan")

        starts = np.linspace(
            0,
            maximum_start,
            min(max_batches, maximum_start + 1),
            dtype=np.int64
        )

        for start in starts:
            context = tokens[int(start):int(start) + self.config.max_context]
            target = tokens[int(start) + 1:int(start) + self.config.max_context + 1]

            if len(context) != self.config.max_context or len(target) != self.config.max_context:
                continue

            logits = self.engine.predict(
                np.asarray([context], dtype=np.int64)
            )[0]

            logits = logits.astype(np.float64)

            logits -= np.max(logits, axis=-1, keepdims=True)

            log_probabilities = logits - np.log(
                np.exp(logits).sum(axis=-1, keepdims=True)
            )

            token_log_probabilities = log_probabilities[
                np.arange(len(target)),
                target
            ]

            losses.append(float(-np.mean(token_log_probabilities)))

        return statistics.mean(losses) if losses else float("nan")


def print_generation_samples(report: DiagnosticReport) -> None:
    results = report.results.get("generation_tests", [])

    for result in results:
        print()
        print("-" * 80)
        print(result["prompt"])
        print("-" * 80)

        for temperature, data in result["temperatures"].items():
            print(f"\nTemperature: {temperature}")
            print(f"Repetition score: {data['repetition_score']:.4f}")
            print(f"Unique token ratio: {data['unique_token_ratio']:.4f}")
            print(data["text"])


def save_report(report: DiagnosticReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(report.results, file, indent=2, ensure_ascii=False)


def main() -> None:
    print("=" * 80)
    print("quillGPT v0.1.3 DIAGNOSTIC SUITE")
    print("=" * 80)

    config = Config()

    report = DiagnosticReport()

    print("\nLoading corpus...")
    corpus = TextCorpus(str(CORPUS_PATH))

    print(f"Corpus loaded: {len(corpus.text):,} characters")

    print("\nLoading tokenizer...")
    tokenizer = BPETokenizer(config.vocab_size)
    tokenizer.load(TOKENIZER_PATH)

    print(
        f"Tokenizer loaded: "
        f"{tokenizer.tokenizer.get_vocab_size():,} tokens"
    )

    print("\nLoading ONNX model...")
    engine = ONNXDiagnosticEngine(MODEL_PATH)

    print(f"Model: {MODEL_PATH}")
    print(f"Providers: {engine.providers}")

    tokenizer_diagnostics = TokenizerDiagnostics(
        tokenizer=tokenizer,
        corpus=corpus.text,
        vocab_size=config.vocab_size
    )

    tokenizer_diagnostics.run(report)

    dataset_diagnostics = DatasetDiagnostics(
        corpus=corpus.text,
        tokenizer=tokenizer,
        max_context=config.max_context
    )

    dataset_diagnostics.run(report)

    model_diagnostics = ModelDiagnostics(
        engine=engine,
        config=config
    )

    model_diagnostics.run(report)

    generation_diagnostics = GenerationDiagnostics(
        engine=engine,
        tokenizer=tokenizer,
        config=config
    )

    generation_diagnostics.run(report)

    loss_diagnostics = LossDiagnostics(
        engine=engine,
        tokenizer=tokenizer,
        corpus=corpus.text,
        config=config
    )

    loss_diagnostics.run(report)

    print_generation_samples(report)

    report.print_summary()

    report_path = Path("diagnostics/quillGPT_v0.1.3_diagnostic.json")
    save_report(report, report_path)

    print()
    print(f"Full diagnostic report saved to: {report_path}")


if __name__ == "__main__":
    main()