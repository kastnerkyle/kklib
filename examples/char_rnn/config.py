seq_length = 50
minibatch_size = 50
hidden_size = 128
n_epochs = 100
n_layers = 2
lr = 2e-3
input_filename = "tiny-shakespeare.txt"
with open(input_filename, "r") as f:
    text = f.read()

chars = set(text)
chars_len = len(chars)
char_to_index = {}
index_to_char = {}
for i, c in enumerate(chars):
    char_to_index[c] = i
    index_to_char[i] = c
