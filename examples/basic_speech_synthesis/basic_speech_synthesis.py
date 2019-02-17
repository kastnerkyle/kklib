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
from kklib.iterators import wavfile_caching_mel_tbptt_iterator
from kklib.datasets import fetch_ljspeech
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
rsync_fetch(fetch_ljspeech, "leto01")
ljspeech = fetch_ljspeech()

wavfiles = ljspeech["wavfiles"]
jsonfiles = ljspeech["jsonfiles"]

# THESE HAVE TO BE THE SAME TO ENSURE SPLIT IS CORRECT
train_random_state = np.random.RandomState(3122)
valid_random_state = np.random.RandomState(3122)

train_itr = wavfile_caching_mel_tbptt_iterator(wavfiles, jsonfiles, minibatch_size, truncation_length, masked=True, stop_index=.95, shuffle=True, symbol_processing="chars_only", random_state=train_random_state)
valid_itr = wavfile_caching_mel_tbptt_iterator(wavfiles, jsonfiles, minibatch_size, truncation_length, masked=True, start_index=.95, shuffle=True, symbol_processing="chars_only", random_state=valid_random_state)

# 0 is phones
vocabulary_size = train_itr.vocabulary_sizes[0]
"""
for i in range(10000):
    print(i)
    mels, mel_mask, text, text_mask, mask, mask_mask, reset = train_itr.next_masked_batch()
print("itr")
"""

output_size = 80
model = Model(minibatch_size, vocabulary_size, embed_size, n_stacks, n_filters, prenet_size, enc_hidden_size, dec_hidden_size, output_size, cell_dropout_keep_rate, random_state).to(DEVICE)

attn_h_i, attn_c_i, attn_k_i, attn_w_i, h1_i, c1_i, h2_i, c2_i = model.make_inits()
attn_h_i = Variable(attn_h_i).to(DEVICE)
attn_c_i = Variable(attn_c_i).to(DEVICE)
attn_k_i = Variable(attn_k_i).to(DEVICE)
attn_w_i = Variable(attn_w_i).to(DEVICE)
h1_i = Variable(h1_i).to(DEVICE)
c1_i = Variable(c1_i).to(DEVICE)
h2_i = Variable(h2_i).to(DEVICE)
c2_i = Variable(c2_i).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10000, 20000, 30000, 500000], gamma=0.5)

saver_dict = {"Model": model,
              "optimizer": optimizer}

stateful_args = [attn_h_i, attn_c_i, attn_k_i, attn_w_i, h1_i, c1_i, h2_i, c2_i]

def loop(itr, extras, stateful):
    o = next(itr)
    mels, mel_mask, text, text_mask, mask, mask_mask, reset = o
    data = mels.copy()
    mask = mel_mask.copy()
    cond_data = text.copy()
    cond_data_mask = text_mask.copy()
    resets = torch.FloatTensor(reset.copy()).to(DEVICE)

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
    preds = o[0]
    attn_h = o[1]
    attn_c = o[2]
    attn_k = o[3]
    attn_w = o[4]
    attn_phi = o[5]
    h1 = o[6]
    c1 = o[7]
    h2 = o[8]
    c2 = o[9]

    targets = target_variable
    step_loss = (preds - targets) ** 2
    #step_loss = loss_mask * step_loss
    loss = torch.mean(torch.sum(step_loss, dim=-1))

    if extras["train"]:
        # backprop and optimize
        loss.backward()
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
         n_steps=1000000,
         n_train_steps_per=1000,
         train_stateful_args=stateful_args,
         valid_stateful_args=stateful_args,
         n_valid_steps_per=0,
         status_every_s=10,
         models_to_keep=5)
