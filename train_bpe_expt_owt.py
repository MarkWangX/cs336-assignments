from cs336_basics.tokenizer import train_bpe
import pickle
import time

input_path = "data/owt_train.txt"
vocab_size = 32000
special_tokens = ["<|endoftext|>"]

#  train bpe on TinyStories dataset and count time
start_time = time.time()
vocab, merges = train_bpe(input_path , vocab_size, special_tokens)
end_time = time.time()
duration = end_time - start_time

print(" The total time for training is: ", duration)
#######################################################################
#  Serialize the resulting vocabulary and merges todiskfor further inspection.
#######################################################################
with open("data/owt_vocab.pkl", "wb") as f:
    pickle.dump(vocab, f)
with open("data/owt_merges.pkl", "wb") as f:
    pickle.dump(merges, f)

# find the logest token
longest_token = max(vocab.values(), key=len)
print(" The longest token is: ", longest_token)