from tokenizers import Tokenizer, Encoding
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

def train_tokenizer(file_path: str):
    """
    Trains the tokenizer on our vocabulary
    """
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=50000, min_frequency=3, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    tokenizer.train([file_path], trainer)
    tokenizer.save("tokenizer.json")

def tokenize(input: str) -> list[int]:
    """
    Tokenizes some input
    """
    tokenizer = Tokenizer.from_file("tokenizer.json")
    return tokenizer.encode(input).ids

if __name__ == "__main__":
    #train_tokenizer("vocab.txt")
    print(tokenize("Hello, I am a human."))