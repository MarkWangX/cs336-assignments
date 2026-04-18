
from cs336_basics.tokenizer import Tokenizer
from tqdm import tqdm
import numpy as np

vocab_path = "data/owt_vocab.pkl"
merges_path = "data/owt_merges.pkl"

tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens=["<|endoftext|>"])

input_text_path = "data/owt_train.txt"
output_text_path = "data/owt_train.bin"

with open(input_text_path, "r", encoding="utf-8") as f:
    total_lines = sum(1 for _ in f)

token_list = []
with open(input_text_path, "r", encoding="utf-8") as f:
    for token_id in tokenizer.encode_iterable(tqdm(f, total = total_lines, desc="Tokenizing dataset")):
        token_list.append(token_id)

tokens = np.array(token_list, dtype=np.uint16)
tokens.tofile(output_text_path)
