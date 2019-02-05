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
model = models.RNN(minibatch_size, chars_len, hidden_size, chars_len, n_layers, minibatch_size, random_state=random_state).to(DEVICE)
model.load_state_dict(all_state_dicts["RNN"])
model = model.to(DEVICE)

sample_len = 100
gen_strings = ["I" for i in range(minibatch_size)]
h0, c0 = model.make_inits()
h = Variable(h0).to(DEVICE)
c = Variable(c0).to(DEVICE)
for i in range(sample_len):
    batch = [char_to_index[gen_strings[i][-1]] for i in range(minibatch_size)]
    batch = torch.LongTensor(batch)[None, :].to(DEVICE)
    out, h, c = model(batch, h, c)
    out = out[0]
    p_out = F.softmax(out, dim=-1)
    p = p_out.cpu().data.numpy()
    p = p / (p.sum(axis=-1)[:, None] + 1E-3)
    sampled = [np.argmax(random_state.multinomial(1, p[i])) for i in range(len(p))]
    sampled_c = [index_to_char[s] for s in sampled]
    for i in range(minibatch_size):
        gen_strings[i] += sampled_c[i]
print(gen_strings)
from IPython import embed; embed(); raise ValueError()
