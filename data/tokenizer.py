"""
Here I will implement the tokenizer. 
This will be a character-level tokenizer for a small model.

NOTE: DEPRECATED FOR V2
"""


from typing import List, Dict

class CharacterTokenizer:
    def __init__(self, text: str) -> None:
        self.characters = sorted(set(text))

        self.char_to_id: Dict[str, int] = {
            character: index for index, character in enumerate(self.characters)
        }

        self.id_to_char: Dict[int, str] = {
            index: character for character, index in self.char_to_id.items()
        }

    @property      # This makes it into a getter, allowing to call the method without parenthesis like obj.vocab_size
    def vocab_size(self) -> int:
        return len(self.characters)
    
    def encode(self, text: str) -> List[int]:
        return [self.char_to_id[character] for character in text]
    
    def decode(self, token_ids: List[int]) -> str:
        return "".join(self.id_to_char[token_id] for token_id in token_ids)
    


