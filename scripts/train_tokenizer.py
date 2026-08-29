from data.bpe_tokenizer import BPETokenizer
from data.corpus import TextCorpus
from config import Config

print("Loading config")
config = Config()

print("Loading tokenizer")
tokenizer = BPETokenizer(config.vocab_size)

print("Loading corpus")
corpus = TextCorpus("data/raw")


print("Starting tokenizer training")
tokenizer.train(corpus.text)

try:
    tokenizer.save("bpe_tokenizer.json")
    print("Tokenizer saved")
except Exception as e:
    print(f"An error occured: {e}")