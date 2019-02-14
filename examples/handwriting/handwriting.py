from torch.autograd import Variable
import torch.nn as nn
import torch

import numpy as np
import time
import math
import os
import argparse

from kklib import run_loop
from kklib.iterators import masked_list_multisequence_iterator
from kklib.iterators import tbptt_list_iterator
from kklib.datasets import fetch_iamondb
from kklib.datasets import rsync_fetch
from kklib.penalties import CorrelatedGMMAndBernoulliCost

from models import Model
from config import *

use_cuda = torch.cuda.is_available()
if use_cuda:
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
"""
parser.add_argument('--seq_len', dest='seq_len', default=256, type=int)
parser.add_argument('--batch_size', dest='batch_size', default=64, type=int)
parser.add_argument('--epochs', dest='epochs', default=30, type=int)
parser.add_argument('--window_mixtures', dest='window_mixtures', default=10, type=int)
parser.add_argument('--output_mixtures', dest='output_mixtures', default=20, type=int)
parser.add_argument('--lstm_layers', dest='lstm_layers', default=3, type=int)
parser.add_argument('--cell_dropout', dest='cell_dropout', default=.9, type=float)
parser.add_argument('--units_per_layer', dest='units', default=400, type=int)
parser.add_argument('--restore', dest='restore', default=None, type=str)
iamondb = fetch_iamondb()
iamondb = rsync_fetch(fetch_iamondb, "leto01")
trace_data = iamondb["data"]
char_data = iamondb["target"]
batch_size = args.batch_size
truncation_len = args.seq_len
cell_dropout_scale = args.cell_dropout
vocabulary_size = len(iamondb["vocabulary"])
itr_random_state = np.random.RandomState(2177)
itr = tbptt_list_iterator(trace_data, [char_data], batch_size, truncation_len,
                          other_one_hot_size=[vocabulary_size],
                          random_state=itr_random_state)
"""

random_state = np.random.RandomState(1345)
iamondb = fetch_iamondb()
trace_data = iamondb["data"]
char_data = iamondb["target"]
vocabulary_size = len(iamondb["vocabulary"])
itr_random_state = np.random.RandomState(2177)
train_itr = tbptt_list_iterator(trace_data[:10000], [char_data[:10000]], minibatch_size, truncation_length,
                                masked=True, random_state=itr_random_state)
valid_itr = tbptt_list_iterator(trace_data[10000:], [char_data[10000:]], minibatch_size, truncation_length,
                                masked=True, random_state=itr_random_state)
"""
for i in range(100):
    o = itr.next()
    # o[0]
    # o[1]
    # o[-1] is continue
    from IPython import embed; embed(); raise ValueError()
"""
model = Model(minibatch_size, 3, hidden_size, random_state).to(DEVICE)
loss_function = nn.CrossEntropyLoss()
h0, c0 = model.make_inits()
h = Variable(h0).to(DEVICE)
c = Variable(c0).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# should the saving function be fed in, or do we feed in something so that the model can save?
# having a "standard" saver seems somewhat hard
saver_dict = {"Model": model,
              "optimizer": optimizer}

stateful_args = [h, c]

def loop(itr, extras, stateful):
    o = next(itr)
    data = o[0]
    mask = o[1]
    #resets = o[-1]
    resets = torch.FloatTensor(o[-1]).to(DEVICE)

    # truncated tbptt
    h = resets * stateful[0]
    c = resets * stateful[1]

    # reset the model
    model.zero_grad()

    input_variable = Variable(torch.FloatTensor(data[:-1])).to(DEVICE)
    target_variable = Variable(torch.FloatTensor(data[1:])).to(DEVICE)
    input_mask = torch.LongTensor(mask[:-1]).to(DEVICE)
    #target_mask = torch.LongTensor(mask[1:]).to(DEVICE)
    target_mask = torch.FloatTensor(mask[1:]).to(DEVICE)

    # prediction and calculate loss
    mus, sigmas, corrs, log_coeffs, berns, h1, c1 = model(input_variable,
                                                          h, c,
                                                          input_mask=input_mask)

    target_bernoullis = target_variable[..., -1]
    target_coords = target_variable[..., :-1]
    step_loss = CorrelatedGMMAndBernoulliCost(mus, sigmas, corrs, log_coeffs, berns, target_coords, target_bernoullis)
    step_loss = target_mask * step_loss
    loss = torch.mean(torch.mean(step_loss, dim=-1))

    #from IPython import embed; embed(); raise ValueError()
    """
    # need to write masking losses too
    output = output.view(-1, output.shape[-1])
    target_variable = target_variable.view(-1)
    loss = loss_function(output, target_variable)
    """
    if extras["train"]:
        # backprop and optimize
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), grad_clip)
        optimizer.step()

    loss = float(loss.detach().cpu().data.numpy())
    # detach important here
    h_n = h1[-1].detach()
    c_n = c1[-1].detach()
    return [[loss], [h_n, c_n]]


run_loop(saver_dict,
         loop, train_itr,
         loop, valid_itr,
         #continue_training=False,
         n_steps=20000,
         # minibatch size is 50
         n_train_steps_per=1000,
         train_stateful_args=stateful_args,
         valid_stateful_args=stateful_args,
         n_valid_steps_per=100,
         status_every_s=5,
         models_to_keep=5)
