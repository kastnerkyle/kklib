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
model = Model(minibatch_size, vocabulary_size, embed_size, hidden_size, random_state).to(DEVICE)
a1_h_i, a1_c_i, a1_i, h1_i, c1_i, h2_i, c2_i = model.make_inits()
a_h_i = Variable(a1_h_i).to(DEVICE)
a_c_i = Variable(a1_c_i).to(DEVICE)
a_i = Variable(a1_i).to(DEVICE)
h1 = Variable(h1_i).to(DEVICE)
c1 = Variable(c1_i).to(DEVICE)
h2 = Variable(h2_i).to(DEVICE)
c2 = Variable(c2_i).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

saver_dict = {"Model": model,
              "optimizer": optimizer}

stateful_args = [a_h_i, a_c_i, a_i, h1, c1, h2, c2]

def loop(itr, extras, stateful):
    o = next(itr)
    data = o[0]
    mask = o[1]
    cond_data = o[2]
    cond_data_mask = o[3]
    resets = torch.FloatTensor(o[-1]).to(DEVICE)

    # truncated tbptt...
    a_h_i = resets * stateful[0]
    a_c_i = resets * stateful[1]
    a_i = resets * stateful[2]
    h1 = resets * stateful[3]
    c1 = resets * stateful[4]
    h2 = resets * stateful[5]
    c2 = resets * stateful[6]

    # reset the model
    model.zero_grad()

    # trim extraneous 1 dim
    cond_data = cond_data[..., 0]
    enc_input_variable = Variable(torch.LongTensor(cond_data)).to(DEVICE)
    enc_input_mask = torch.LongTensor(cond_data_mask).to(DEVICE)

    dec_input_variable = Variable(torch.FloatTensor(data[:-1])).to(DEVICE)
    dec_input_mask = torch.LongTensor(mask[:-1]).to(DEVICE)

    target_variable = Variable(torch.FloatTensor(data[1:])).to(DEVICE)
    loss_mask = torch.FloatTensor(mask[1:]).to(DEVICE)

    o = model(enc_input_variable, dec_input_variable,
              a_h_i, a_c_i, a_i,
              h1, c1, h2, c2,
              enc_input_mask=enc_input_mask, dec_input_mask=dec_input_mask)
    mus = o[0]
    sigmas = o[1]
    corrs = o[2]
    log_coeffs = o[3]
    berns = o[4]
    attn_h = o[5]
    attn_c = o[6]
    attn_a_h = o[7]
    h1 = o[8]
    c1 = o[9]
    h2 = o[10]
    c2 = o[11]

    target_bernoullis = target_variable[..., -1]
    target_coords = target_variable[..., :-1]
    step_loss = CorrelatedGMMAndBernoulliCost(mus, sigmas, corrs, log_coeffs, berns, target_coords, target_bernoullis)
    step_loss = loss_mask * step_loss
    loss = torch.mean(torch.mean(step_loss, dim=-1))

    if extras["train"]:
        # backprop and optimize
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), grad_clip)
        optimizer.step()

    loss = float(loss.detach().cpu().data.numpy())
    # detach important here
    h_n = h1[-1].detach()
    c_n = c1[-1].detach()
    attn_h_n = attn_h[-1].detach()
    attn_c_n = attn_c[-1].detach()
    attn_a_h_n = attn_a_h[-1].detach()
    h1_n = h1[-1].detach()
    c1_n = c1[-1].detach()
    h2_n = h2[-1].detach()
    c2_n = c2[-1].detach()
    return [[loss], [attn_h_n, attn_c_n, attn_a_h_n, h1_n, c1_n, h2_n, c2_n]]


run_loop(saver_dict,
         loop, train_itr,
         loop, valid_itr,
         #continue_training=False,
         n_steps=30000,
         # minibatch size is 50
         n_train_steps_per=1000,
         train_stateful_args=stateful_args,
         valid_stateful_args=stateful_args,
         n_valid_steps_per=100,
         status_every_s=5,
         models_to_keep=5)
