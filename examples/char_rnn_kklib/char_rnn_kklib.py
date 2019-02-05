from torch.autograd import Variable
import torch.nn as nn
import torch

import numpy as np
import time
import math
import os
import argparse

from kklib import run_loop

from models import RNN
from config import *

use_cuda = torch.cuda.is_available()
if use_cuda:
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

# from https://raw.githubusercontent.com/jcjohnson/torch-rnn/master/data/tiny-shakespeare.txt

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)

def chunks(l, n):
    #print(list(chunks(range(11), 3)))
    for i in range(0, len(l) - n, n):
        yield l[i:i + n]

def index_to_tensor(index):
    tensor = torch.zeros(1, 1).long()
    tensor[0,0] = index
    return Variable(tensor)


# convert all characters to indices
batches = [char_to_index[char] for char in text]

# chunk into sequences of length seq_length + 1
batches = list(chunks(batches, seq_length + 1))

# chunk sequences into batches
batches = list(chunks(batches, minibatch_size))

# convert batches to tensors and transpose
batches = [torch.LongTensor(batch).transpose_(0, 1) for batch in batches]

loss_function = nn.CrossEntropyLoss()

random_state = np.random.RandomState(1345)
model = RNN(minibatch_size, chars_len, hidden_size, chars_len, n_layers, minibatch_size, random_state).to(DEVICE)
h0, c0 = model.make_inits()
h = Variable(h0).to(DEVICE)
c = Variable(c0).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

def loop(itr, extras, stateful):
    batch_tensor = next(itr)
    if use_cuda:
        batch_tensor = batch_tensor.cuda()

    # reset the model
    model.zero_grad()

    # everything except the last
    input_variable = Variable(batch_tensor[:-1]).to(DEVICE)

    # everything except the first, flattened
    target_variable = Variable(batch_tensor[1:].contiguous().view(-1)).to(DEVICE)

    # prediction and calculate loss
    output, h1, c1 = model(input_variable, h, c)
    output = output.view(-1, output.shape[-1])
    loss = loss_function(output, target_variable)
    if extras["train"]:
        # backprop and optimize
        loss.backward()
        optimizer.step()

    loss = float(loss.detach().cpu().data.numpy())
    return [[loss], [None]]


class minitr:
    def __init__(self, list_of_data, random_state):
        self.data = list_of_data
        self.idx = 0

    def next(self):
        if self.idx >= (len(self.data) - 1):
            self.reset()
        d = self.data[self.idx]
        self.idx += 1
        return d

    def  __next__(self):
        return self.next()

    def reset(self):
        self.idx = 0
        random_state.shuffle(self.data)


train_len = int(.9 * len(batches))
random_state = np.random.RandomState(2177)
train_itr = minitr(batches[:train_len], random_state)
valid_itr = minitr(batches[train_len:], random_state)

# should the saving function be fed in, or do we feed in something so that the model can save?
# having a "standard" saver seems somewhat hard
saver_dict = {"RNN": model,
              "optimizer": optimizer}

run_loop(saver_dict,
         loop, train_itr,
         loop, valid_itr,
         #continue_training=False,
         # ~ 300 mbs per epoch
         n_steps=n_epochs * 300,
         n_train_steps_per=1000,
         #train_stateful_args=None,
         #valid_stateful_args=None,
         n_valid_steps_per=50,
         status_every_s=5,
         models_to_keep=5)
