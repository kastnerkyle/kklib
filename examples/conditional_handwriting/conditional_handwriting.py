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
rsync_fetch(fetch_iamondb, "leto01")
iamondb = fetch_iamondb()
trace_data = iamondb["data"]
char_data = iamondb["target"]
vocabulary_size = len(iamondb["vocabulary"])
itr_random_state = np.random.RandomState(2177)
train_itr = tbptt_list_iterator(trace_data[:10000], [char_data[:10000]], minibatch_size, truncation_length,
                                masked=True, random_state=itr_random_state)
valid_itr = tbptt_list_iterator(trace_data[10000:], [char_data[10000:]], minibatch_size, truncation_length,
                                masked=True, random_state=itr_random_state)
model = Model(minibatch_size, vocabulary_size, embed_size, hidden_size, cell_dropout_keep_rate, random_state).to(DEVICE)
attn_h_i, attn_c_i, attn_k_i, attn_w_i, h1_i, c1_i, h2_i, c2_i = model.make_inits()
attn_h_i = Variable(attn_h_i).to(DEVICE)
attn_c_i = Variable(attn_c_i).to(DEVICE)
attn_k_i = Variable(attn_k_i).to(DEVICE)
attn_w_i = Variable(attn_w_i).to(DEVICE)
h1_i = Variable(h1_i).to(DEVICE)
c1_i = Variable(c1_i).to(DEVICE)
h2_i = Variable(h2_i).to(DEVICE)
c2_i = Variable(c2_i).to(DEVICE)
n_steps = 50000
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10000, 20000, 30000, 40000, 50000], gamma=0.5)

saver_dict = {"Model": model,
              "optimizer": optimizer}

stateful_args = [attn_h_i, attn_c_i, attn_k_i, attn_w_i, h1_i, c1_i, h2_i, c2_i]

def loop(itr, extras, stateful):
    o = next(itr)
    data = o[0]
    mask = o[1]
    cond_data = o[2]
    cond_data_mask = o[3]
    resets = torch.FloatTensor(o[-1]).to(DEVICE)

    # truncated tbptt...
    attn_h_i = resets * stateful[0]
    attn_c_i = resets * stateful[1]
    attn_k_i = resets * stateful[2]
    attn_w_i = resets * stateful[3]
    h1_i = resets * stateful[4]
    c1_i = resets * stateful[5]
    h2_i = resets * stateful[6]
    c2_i = resets * stateful[7]

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
              attn_h_i, attn_c_i, attn_k_i, attn_w_i,
              h1_i, c1_i, h2_i, c2_i,
              enc_input_mask=enc_input_mask, dec_input_mask=dec_input_mask)
    mus = o[0]
    sigmas = o[1]
    corrs = o[2]
    log_coeffs = o[3]
    berns = o[4]
    attn_h = o[5]
    attn_c = o[6]
    attn_k = o[7]
    attn_w = o[8]
    attn_phi = o[9]
    h1 = o[10]
    c1 = o[11]
    h2 = o[12]
    c2 = o[13]

    target_bernoullis = target_variable[..., -1]
    target_coords = target_variable[..., :-1]
    step_loss = CorrelatedGMMAndBernoulliCost(mus, sigmas, corrs, log_coeffs, berns, target_coords, target_bernoullis)
    #step_loss = loss_mask * step_loss
    loss = torch.mean(torch.mean(step_loss, dim=-1))

    if extras["train"]:
        # backprop and optimize
        loss.backward()
        #torch.nn.utils.clip_grad_value_(model.parameters(), grad_clip)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scheduler.step()
        optimizer.step()

    loss = float(loss.detach().cpu().data.numpy())
    attn_h_n = attn_h[-1].detach()
    attn_c_n = attn_c[-1].detach()
    attn_k_n = attn_k[-1].detach()
    attn_w_n = attn_w[-1].detach()
    h1_n = h1[-1].detach()
    c1_n = c1[-1].detach()
    h2_n = h2[-1].detach()
    c2_n = c2[-1].detach()
    return [[loss], [attn_h_n, attn_c_n, attn_k_n, attn_w_n, h1_n, c1_n, h2_n, c2_n]]

run_loop(saver_dict,
         loop, train_itr,
         loop, valid_itr,
         #continue_training=False,
         n_steps=n_steps,
         # minibatch size is 50
         n_train_steps_per=1000,
         train_stateful_args=stateful_args,
         valid_stateful_args=stateful_args,
         n_valid_steps_per=100,
         status_every_s=5,
         models_to_keep=5)
