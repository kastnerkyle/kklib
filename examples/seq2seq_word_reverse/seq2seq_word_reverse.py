from torch.autograd import Variable
import torch.nn as nn
import torch

import numpy as np
import time
import math
import os
import argparse

from kklib import run_loop

from models import Model
from config import *

use_cuda = torch.cuda.is_available()
if use_cuda:
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

class masked_list_multisequence_iterator(object):
    """
       assumes it is elements, sequence, features ordered

       for an example of a list of numbers, and its reverse
       [[[1, 2, 3], [4, 5, 6]], [[3, 2, 1], [6, 5, 4]]]

       batches returned as two lists of elements, (sequence, samples), one for data, one for masks
    """
    def __init__(self, list_of_data, minibatch_size, random_state=None, pad_with=0, dtype="float32", mask_dtype="float32"):
        self.data = list_of_data
        l = len(list_of_data[0])
        for i in range(len(list_of_data)):
            if len(list_of_data[i]) != l:
                raise ValueError("All list inputs in list_of_data must be the same length!")
        self.dtype = dtype
        self.mask_dtype = mask_dtype
        self.minibatch_size = minibatch_size
        self.pad_with = pad_with
        self.random_state = random_state
        self.start_idx_ = 0
        self.n_elements_ = len(list_of_data)

    def next(self):
        if self.start_idx_ >= (len(self.data[0]) - self.minibatch_size):
            self.reset()
        start_ = self.start_idx_
        end_ = self.start_idx_ + self.minibatch_size
        subs = [d[start_:end_] for d in self.data]
        maxlens = [max([len(su_i) for su_i in su]) for su in subs]
        r = []
        mask_r = []
        for n, su in enumerate(subs):
            r_i = []
            mask_r_i = []
            for su_i in su:
                if len(su_i) != maxlens[n]:
                    pad_part = [self.pad_with for _ in range(maxlens[n] - len(su_i))]
                    pad_su_i = su_i + pad_part
                    mask_su_i = [1. for _ in su_i] + [0. for _ in pad_part]
                    r_i.append(pad_su_i)
                    mask_r_i.append(mask_su_i)
                else:
                    r_i.append(su_i)
                    mask_r_i.append([1. for _ in su_i])
            np_r_i = np.array(r_i).astype(self.dtype).transpose(1, 0)
            np_mask_r_i = np.array(mask_r_i).astype(self.mask_dtype).transpose(1, 0)
            r.append(np_r_i)
            mask_r.append(np_mask_r_i)
        self.start_idx_ = end_
        return r, mask_r

    def  __next__(self):
        return self.next()

    def reset(self):
        self.start_idx_ = 0
        if self.random_state is not None:
            random_state.shuffle(self.data)


# convert all characters to indices
data_in_int = [[symbol_to_index[s] for s in s_i] for s_i in data_in]
data_out_int = [[symbol_to_index[s] for s in s_i] for s_i in data_out]

random_state = np.random.RandomState(1345)
train_itr = masked_list_multisequence_iterator([data_in_int[:3000], data_out_int[:3000]], minibatch_size, random_state=random_state, pad_with=0)
valid_itr = masked_list_multisequence_iterator([data_in_int[-100:], data_out_int[-100:]], minibatch_size, random_state=random_state, pad_with=0)
"""
for i in range(100):
    d, m_d = next(valid_itr)
    print("          ")
    print(d[0][:, 0], d[1][:, 0]) #, m_d)
    print("          ")
from IPython import embed; embed(); raise ValueError()
"""

loss_function = nn.CrossEntropyLoss()

model = Model(minibatch_size, vocab_size, embed_size, hidden_size, vocab_size, random_state).to(DEVICE)
dh0, dc0, a0 = model.make_inits()
#ehf = Variable(ehf0).to(DEVICE)
#ecf = Variable(ecf0).to(DEVICE)
#ehb = Variable(ehb0).to(DEVICE)
#ecb = Variable(ecb0).to(DEVICE)

dh = Variable(dh0).to(DEVICE)
dc = Variable(dc0).to(DEVICE)
a = Variable(a0).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

def loop(itr, extras, stateful):
    data, masks = next(itr)

    # reset the model
    model.zero_grad()

    input_variable = Variable(torch.LongTensor(data[0])).to(DEVICE)
    target_variable = Variable(torch.LongTensor(data[1])).to(DEVICE)
    input_mask = torch.LongTensor(masks[0]).to(DEVICE)
    target_mask = torch.LongTensor(masks[1]).to(DEVICE)

    # prediction and calculate loss
    output, dh1, dc1, a1, attn = model(input_variable, target_variable,
                                       dh, dc, a,
                                       input_mask=input_mask,
                                       output_mask=target_mask)
    # need to write masking losses too
    output = output.view(-1, output.shape[-1])
    target_variable = target_variable.view(-1)
    loss = loss_function(output, target_variable)
    if extras["train"]:
        # backprop and optimize
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), grad_clip)
        optimizer.step()

    loss = float(loss.detach().cpu().data.numpy())
    return [[loss], [None]]


# should the saving function be fed in, or do we feed in something so that the model can save?
# having a "standard" saver seems somewhat hard
saver_dict = {"Model": model,
              "optimizer": optimizer}

run_loop(saver_dict,
         loop, train_itr,
         loop, valid_itr,
         #continue_training=False,
         n_steps=20000,
         # minibatch size is 50
         n_train_steps_per=1000,
         #train_stateful_args=None,
         #valid_stateful_args=None,
         n_valid_steps_per=100,
         status_every_s=5,
         models_to_keep=5)
