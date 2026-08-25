"""
Byte-pair encoding tokenizer for V2.

Counts the most frequently occuring character pairs, and has various permutations with other characters in the string.
Eventually build the entire vocabulary like this.
"""

from collections import Counter
from typing import Dict, List, Tuple

class BPETokenizer:
    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size
        self.vocab: Dict[int, str] = {}
        self.token_to_id: Dict[str, int] = {}
        self.merges: Dict[Tuple[str, str], str] = {}

    def build_initial_vocabulary(self, text: str) -> List[str]:
        characters = sorted(set(text))

        self.vocab = {token_id: character for token_id, character in enumerate(characters)}
        self.token_to_id = {token: token_id for token_id, token in self.vocab.items()}

        return list(text)
    
    def count_pairs(self, tokens: List[str]) -> Counter[Tuple[str, str]]:
        pairs = Counter()

        for first, second in zip(tokens, tokens[1:]):
            pairs[(first, second)] += 1

        return pairs
    
    def merge_pair(self, tokens: List[str], pair: Tuple[str, str], merged_token: str) -> List[str]:
        merged: List[str] = []
        index = 0

        while index < len(tokens):
            if (index < len(tokens) - 1 and tokens[index] == pair[0] and tokens[index + 1] == pair[1]):
                merged.append(merged_token)
                index += 2
            else:
                merged.append(tokens[index])
                index += 1

        return merged
    
    def train(self, text: str) -> None:
        tokens = self.build_initial_vocabulary(text)

        while len(self.vocab) < self.vocab_size:
            pair_counts = self.count_pairs(tokens)

            if not pair_counts:
                break
        
            most_common_pair, frequency = pair_counts.most_common(1)[0]

            if frequency < 2:
                break

            first, second = most_common_pair
            merged_token = first + second

            self.merges[most_common_pair] = merged_token

            new_token_id = len(self.vocab)

            self.vocab[new_token_id] = merged_token
            self.token_to_id[merged_token] = new_token_id

            tokens = self.merge_pair(tokens, most_common_pair, merged_token)

    def encode(self, text: str) -> List[int]:
        tokens = list(text)

        for pair, merged_token in self.merges.items():
            tokens = self.merge_pair(tokens, pair, merged_token)
        
        return [
            self.token_to_id[token] for token in tokens

        ]
    
    def decode(self, token_ids: List[int]) -> str:
        return "".join(self.vocab[token_id] for token_id in token_ids)
    

if __name__ == "__main__":
    text = "low low lower lowest"

    tokenizer = BPETokenizer(vocab_size=20)
    tokenizer.train(text)

    print("Vocabulary:")
    for token_id, token in tokenizer.vocab.items():
        print(token_id, repr(token))

    encoded = tokenizer.encode(text)

    print("\nEncoded:")
    print(encoded)

    print("\nDecoded:")
    print(tokenizer.decode(encoded))
            