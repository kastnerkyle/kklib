import string
import numpy as np
seq_length = 15
minibatch_size = 10
embed_size = 10
hidden_size = 128
n_epochs = 1000
lr = 1e-3
grad_clip = 5.

PAD = 0
EOS = 1

symbols = ["~", "^"] + list(string.ascii_lowercase)
vocab_size = len(symbols)
symbol_to_index = {}
index_to_symbol = {}
for i, s in enumerate(symbols):
    symbol_to_index[s] = i
    index_to_symbol[i] = s

rs = np.random.RandomState(24)
def random_pair(min_length, max_length):
    random_length = rs.choice(list(range(min_length, max_length)))
    random_symbol_list = [rs.choice(symbols[2:]) for _ in range(random_length)]
    random_string = "".join(random_symbol_list) + "^"
    return random_string, random_string[:-1][::-1] + "^"

data = [random_pair(1, seq_length) for _ in range(3100)]
data_in = [d[0] for d in data]
data_out = [d[1] for d in data]
