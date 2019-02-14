import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import argparse
import sys

from kklib import get_saved_model_defs
from kklib import get_saved_model_config

use_cuda = torch.cuda.is_available()
if use_cuda:
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
parser = argparse.ArgumentParser()
parser.add_argument('direct_model', nargs=1, default=None)
parser.add_argument('--model', dest='model_path', type=str, default=None)
args = parser.parse_args()
if args.model_path == None:
    if args.direct_model == None:
        raise ValueError("Must pass first positional argument as model, or --model argument, e.g. summary/experiment-0/models/model-7")
    else:
        saved_model_path = args.direct_model[0]
else:
    saved_model_path = args.model_path

random_state = np.random.RandomState(11)
all_state_dicts = torch.load(saved_model_path)
models_file_path = get_saved_model_defs(saved_model_path)
sys.path.insert(0, models_file_path); import models; sys.path.remove(models_file_path)
config_file_path = get_saved_model_config(saved_model_path)
sys.path.insert(0, config_file_path); from config import *; sys.path.remove(config_file_path)
model = models.Model(minibatch_size, vocab_size, embed_size, hidden_size, vocab_size, random_state).to(DEVICE)
model.load_state_dict(all_state_dicts["Model"])
model = model.to(DEVICE)

sample_len = 100
h0, c0, a0 = model.make_inits()
h = Variable(h0).to(DEVICE)
c = Variable(c0).to(DEVICE)
a = Variable(a0).to(DEVICE)
#in_strings = ["abcdefg^" for i in range(minibatch_size)]
in_strings = ["icecream^" for i in range(minibatch_size)]
out_strings = ["~" for i in range(minibatch_size)]
for i in range(len(in_strings[0])):
    in_batch = [[symbol_to_index[in_strings[i][j]] for j in range(len(in_strings[i]))] for i in range(minibatch_size)]
    in_batch = torch.LongTensor(in_batch).transpose(0, 1).to(DEVICE)

    out_batch = [[symbol_to_index[out_strings[i][j]] for j in range(len(out_strings[i]))] for i in range(minibatch_size)]
    out_batch = torch.LongTensor(out_batch).transpose(0, 1).to(DEVICE)
    in_mask = 0 * in_batch + 1
    out_mask = 0 * out_batch + 1
    # slow sample using full processing each time
    output, dh1, dc1, a1, attn = model(in_batch, out_batch,
                                       h, c, a, input_mask=in_mask, output_mask=out_mask)
    out = output[-1]
    p_out = F.softmax(out, dim=-1)
    p = p_out.cpu().data.numpy()
    p = p / (p.sum(axis=-1)[:, None] + 1E-3)
    # sampling here though should be argmax for deterministic things
    sampled = [np.argmax(random_state.multinomial(1, p[i])) for i in range(len(p))]
    sampled_c = [index_to_symbol[s] for s in sampled]
    for m in range(minibatch_size):
        out_strings[m] = out_strings[m][:-1] + sampled_c[m]
        out_strings[m] += "~"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def _plot_attention(matrix, head, max_weight=None, ax=None):
       """Draw Hinton diagram for visualizing a weight matrix."""
       # https://talbaumel.github.io/blog/attention/
       ax = ax if ax is not None else plt.gca()

       if not max_weight:
           max_weight = 2**np.ceil(np.log(np.abs(matrix).max())/np.log(2))

       ax.patch.set_facecolor('gray')
       ax.set_aspect('equal', 'box')
       ax.xaxis.set_major_locator(plt.NullLocator())
       ax.yaxis.set_major_locator(plt.NullLocator())

       for (x, y), w in np.ndenumerate(matrix):
           color = 'white' if w > 0 else 'black'
           size = np.sqrt(np.abs(w))
           rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                                facecolor=color, edgecolor=color)
           ax.add_patch(rect)

       ax.autoscale_view()
       ax.invert_yaxis()
       plt.savefig("attn{}.png".format(head))
       #plt.show()

if len(attn.shape) > 3:
    for head in range(attn.shape[-1]):
        # cut off the ends due to padding with ~
        _plot_attention(attn[:-1, :-1, 0, head].detach().numpy(), head)
else:
    # cut off the ends due to padding with ~
    _plot_attention(attn[:-1, :-1, 0].detach().numpy(), 0)
