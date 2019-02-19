import string
import numpy as np
truncation_length = 256
minibatch_size = 64
embed_size = 15
cell_dropout_keep_rate = 0.925
n_filters = 64
n_stacks = 3
prenet_size = 100
enc_hidden_size = 100
dec_hidden_size = 400
lr = 1E-3
grad_clip = 3.
