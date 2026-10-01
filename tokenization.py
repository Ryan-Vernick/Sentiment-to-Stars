import re
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

amazon_links = r'https?://(?:[a-z0-9-]+\.)*amazon\.[a-z.]{2,6}/(?:[^/\s]+/)?(?:dp|gp/(?:product|aw/d))/[A-Z0-9]{10}(?:[/?].*)?'
edits_or_unecessary = r'-{7,}.*?-{7,}|>{2,}'
html_tags = r'<[^>]{0,8}>'
malformed_less_sign = r'&lt;'
malformed_greater_sign = r'&gt;'
a_tags = r'<a\b[^>]*>(.*?)</a>'
remove_pattern = re.compile('|'.join([amazon_links, edits_or_unecessary, malformed_less_sign, malformed_greater_sign, a_tags, html_tags]), re.IGNORECASE)
excessive_hyphens = r'-{3,6}'
tokenizer = Tokenizer.from_file("tokenizer.json")

def train_tokenizer(file_path: str):
    """
    Trains the tokenizer on our vocabulary
    """
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.post_processor = TemplateProcessing(single="$A [SEP]", special_tokens=[("[SEP]", 2),],)
    tokenizer.enable_padding(pad_id=3, pad_token="[PAD]", length=2500)
    tokenizer.enable_truncation(max_length=2500)
    trainer = BpeTrainer(vocab_size=50000, min_frequency=3, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    tokenizer.train([file_path], trainer)
    tokenizer.save("tokenizer2.json")

def tokenize(input: str) -> list[int]:
    """
    Tokenizes some input
    """
    input = re.sub(pattern=remove_pattern, repl='', string=input)
    input = re.sub(pattern=excessive_hyphens, repl='--', string=input).strip('"')
    return tokenizer.encode(input).ids

if __name__ == "__main__":
    train_tokenizer("vocab2.txt")
    print(tokenize("Simply awful product if you are familiar with Sunsweet or other domestic brands.  I don't really care if they came from Al Sharpton's basement, they taste stale, colorless, not tangy like an apricot... OMG.  I write a lot of positive food reviews here and there is nothing to say about these.  I suppose if you were a very dedicated veg that you could mix them with some normal Sunsweet apricots and get them down.  I'm going to find a way to use them as a door stop.  Save your money, there's lot's of other brands of apricots on here including Newmans Own."))