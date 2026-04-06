import os
import regex as re
from typing import BinaryIO
from collections import Counter, defaultdict
from multiprocessing import Pool
import pickle
###############################################################
# find_chunk_boundaries will receive a file and 
# return a list of int that indicates the boundaries of chunks 
###############################################################

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def multiprocess_chunk(args):       
    input_path, start, end, special_tokens = args

    with open(input_path, "rb") as f:
        # before pre-tokenization, we will first seperate chunks with special tokens
        f.seek(start)
        raw_content = f.read(end-start).decode('utf-8', errors='ignore')
        
    # Check if special tokens is none
    if special_tokens:
        special_tokens_escaped = [re.escape(token) for token in special_tokens]
        raw_chunk = re.split('|'.join(special_tokens_escaped), raw_content)
    else:
        raw_chunk = [raw_content]

    # Pre-tokenization on the chunk
    pre_tokenized_chunk = Counter()
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    
    compiled_pat = re.compile(PAT)
    for match in raw_chunk:
        slice = compiled_pat.findall(match)
        pre_tokenized_chunk.update(slice)

    return pre_tokenized_chunk

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
    # Find the boundaries of the file
    with open(input_path, "rb") as f:
        num_processes = 12
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
    
    # Assign tasks for multiprocessor 
    tasks = [
        (input_path, start, end, special_tokens)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]

    with Pool(num_processes) as pool:
        pre_tokenized_content = pool.map(multiprocess_chunk, tasks)

    # Reduce the results of different chunks
    pre_tokenized_voc = Counter()
    for local_dic in pre_tokenized_content:
        pre_tokenized_voc.update(local_dic)
    
    # Initialize Vocab and count the current vocabulary size
    vocab = {}
    for idx, token in enumerate(special_tokens):
        vocab[idx] =  token.encode("utf-8")
    cur_vocab_size = len(vocab)
    for i in range(256):
        vocab[cur_vocab_size + i] = bytes([i])
    cur_vocab_size = len(vocab)

    # Initialize merges
    merges = []

    # reconstruct the dic[bytes, int] into dic[tuple(bytes), int]
    # Eg: {low: 5, high: 3} -> {(b'l', b'o', b'w'): 5, (b'h', b'i', b'g', b'h'): 3}   
    freq_table = {
        tuple(bytes([b]) for b in word.encode("utf-8")): freq 
        for word, freq in pre_tokenized_voc.items()
    }
    
    #############################################################################
    ### we will build three references:
    ### byte_pair_counts: dictionary        byte_pair_counts[(b'l', b'o')] = 5
    ### pair_indices: dictionary            pair_indices[(b'l', b'o')] = {(b'l', b'o', b'w'), (b'l', b'o', b'w', b'e', b'r')}
    #############################################################################  
    byte_pair_counts, pair_indices = Counter(), defaultdict(set)
    for word, freq in freq_table.items():
        for pair in zip(word[:-1], word[1:]):
            byte_pair_counts[pair] += freq
            pair_indices[pair].add(word)
    
    while(cur_vocab_size < vocab_size):
        # safety check
        if not byte_pair_counts:
            break

        # find next byte pair to update
        merge_pair = max(byte_pair_counts, key=lambda k: (byte_pair_counts[k], k))
        first, second = merge_pair
        merged_byte = first + second

        byte_pair_counts.pop(merge_pair)
        words_to_process = list(pair_indices.pop(merge_pair, set()))

        for word in words_to_process:
            word_freq = freq_table.pop(word, 0)
            if(word_freq == 0): continue

            # remove all the neighber pair in the old word
            for pair in zip(word[:-1], word[1:]):
                byte_pair_counts[pair] -= word_freq
                
                if byte_pair_counts[pair] <= 0:
                    del byte_pair_counts[pair]

                if word in pair_indices.get(pair, set()):
                    pair_indices[pair].remove(word)
                    
                    if not pair_indices[pair]:
                        del pair_indices[pair]

            i = 0
            new_word = []
            while i < len(word):
                if i < len(word)-1 and word[i] == first and word[i+1] == second:
                    new_word.append(merged_byte)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word_tuple = tuple(new_word)
            freq_table[new_word_tuple] = word_freq

            for pair in zip(new_word_tuple[:-1], new_word_tuple[1:]):
                byte_pair_counts[pair] += word_freq
                pair_indices[pair].add(new_word_tuple)

        # Update merges and vocab
        merges.append(merge_pair)
        vocab[cur_vocab_size] = merged_byte
        cur_vocab_size += 1
    
    return vocab, merges

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab.copy()
        self.merges = merges
        self.special_tokens = special_tokens
        vocab_size = max(self.vocab.keys()) + 1
        existing_values = set(self.vocab.values())
        if special_tokens is not None:
            for special_token in special_tokens:
                special_token = special_token.encode("utf-8")
                if special_token not in existing_values:
                    self.vocab[vocab_size] = special_token
                    existing_values.add(special_tokens)
                    vocab_size += 1
        self.token_to_ids = {value: key for key, value in self.vocab.items()}
        self.merge_dict = {pair: i for i, pair in enumerate(self.merges)}

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)

        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def encode(self, text:str) -> list[int]:
        standard_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        
        if hasattr(self, 'special_tokens') and self.special_tokens:
            special_pattern = "|".join(re.escape(tok) for tok in self.special_tokens)
            final_pattern = f"({special_pattern}|{standard_pattern})"
        else:
            final_pattern = standard_pattern

        compiled_pat = re.compile(final_pattern)
        pre_tokens = compiled_pat.findall(text)

        for word in pre_tokens:
            if hasattr(self, 'special_tokens') and self.special_tokens and word in self.special_tokens:
                word_bytes = word.encode('utf-8')
                out_ids.append(self.token_to_ids[word_bytes])
                continue

            out_ids = []
            word_byte = [bytes([b]) for b in word.encode("utf-8")]
            while len(word_byte) > 1:
                pairs = list(zip(word_byte[:-1], word_byte[1:]))

                valid_pairs = [pair for pair in pairs if pair in self.merge_dict]
                if not valid_pairs:
                    break
                best_pairs = min(valid_pairs, key = lambda x: valid_pairs[x])
                first, second = best_pairs
                
                i = 1
                new_word = []         
                while i < len(word_byte):
                        if i < len(word_byte) - 1 and word_byte[i] == first and word_byte[i+1] == second:
                            new_word.append(first + second)
                            i += 2
                            break
                        else:
                            new_word.append(word[i])
                            i += 1
                word_byte = new_word

        for b in new_word:
            out_ids.append(self.token_to_ids[b])

        return out_ids

    def decode(self, ids: list[int]) -> str:
        tokens = []
        for id in ids:
            tokens.append(self.vocab[id])
        return tokens
  