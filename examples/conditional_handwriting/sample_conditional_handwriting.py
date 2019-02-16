import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import argparse
import sys

from kklib import get_saved_model_defs
from kklib import get_saved_model_config

from kklib.datasets import fetch_iamondb

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

def numpy_softmax(x, temperature=1.):
    # should work for both 2D and 3D
    dim = x.ndim
    x = x / temperature
    e_x = np.exp((x - x.max(axis=dim - 1, keepdims=True)))
    out = e_x / e_x.sum(axis=dim - 1, keepdims=True)
    return out

def cumsum(points):
    sums = np.cumsum(points[:, :2], axis=0)
    return np.concatenate([sums, points[:, 2:]], axis=1)

def split_strokes(points):
    points = np.array(points)
    strokes = []
    b = 0
    for e in range(len(points)):
        if points[e, 2] == 1.:
            strokes += [points[b: e + 1, :2].copy()]
            b = e + 1
    strokes += [points[b: e + 1, :2].copy()]
    return strokes

def sample(mu1, mu2, std1, std2, rho, bern, random_state=None):
    if random_state is None:
        raise ValueError("Random state must be provided, currently None")
    cov = np.array([[std1 * std1, std1 * std2 * rho],
                    [std1 * std2 * rho, std2 * std2]])
    mean = np.array([mu1, mu2])

    x, y = random_state.multivariate_normal(mean, cov)
    end = random_state.binomial(1, bern)
    return np.array([x, y, end])

random_state = np.random.RandomState(11)

iamondb = fetch_iamondb()
trace_data = iamondb["data"]
char_data = iamondb["target"]
vocabulary_size = len(iamondb["vocabulary"])

all_state_dicts = torch.load(saved_model_path, map_location=DEVICE)
models_file_path = get_saved_model_defs(saved_model_path)
sys.path.insert(0, models_file_path); import models; sys.path.remove(models_file_path)
config_file_path = get_saved_model_config(saved_model_path)
sys.path.insert(0, config_file_path); from config import *; sys.path.remove(config_file_path)
model = models.Model(minibatch_size, vocabulary_size, embed_size, hidden_size, cell_dropout_keep_rate, random_state).to(DEVICE)
model.load_state_dict(all_state_dicts["Model"])
model = model.to(DEVICE)
model = model.eval()

attn_h_i, attn_c_i, attn_k_i, attn_w_i, h1_i, c1_i, h2_i, c2_i = model.make_inits()
attn_h_i = Variable(attn_h_i).to(DEVICE)
attn_c_i = Variable(attn_c_i).to(DEVICE)
attn_k_i = Variable(attn_k_i).to(DEVICE)
attn_w_i = Variable(attn_w_i).to(DEVICE)
h1_i = Variable(h1_i).to(DEVICE)
c1_i = Variable(c1_i).to(DEVICE)
h2_i = Variable(h2_i).to(DEVICE)
c2_i = Variable(c2_i).to(DEVICE)

sample_len = 500
coords = np.array([0., 0., 1.])
coords = coords[None] * np.ones((minibatch_size, 1))
coords = coords[None]
coords_mask = 0. * coords[:, :, 0] + 1.

predicted_coords = [coords]
predicted_attn_k = []
predicted_attn_w = []
predicted_attn_phi = []

teacher_force_coords = trace_data[0][:, None] * np.ones((1, minibatch_size, 1))

symbol_to_ind = iamondb["vocabulary"]
ind_to_symbol = {v:k for k, v in symbol_to_ind.items()}
chars = "today is the day"
inds = [symbol_to_ind[c] for c in chars]

#inds = char_data[0]
#chars = "".join([ind_to_symbol[i] for i in inds])

# Only 2D, so embedding works
inds = np.array(inds)[:, None] * np.ones((1, minibatch_size)).astype("int32")
inds_mask = 0 * inds + 1

for i in range(sample_len):
    print("Sample step {}".format(i))
    enc_input_variable = Variable(torch.LongTensor(inds)).to(DEVICE)
    enc_input_mask = torch.LongTensor(inds_mask).to(DEVICE)

    dec_input_variable = Variable(torch.FloatTensor(predicted_coords[-1])).to(DEVICE)
    #dec_input_variable = Variable(torch.FloatTensor(teacher_force_coords[i][None])).to(DEVICE)

    dec_input_mask = Variable(torch.FloatTensor(coords_mask)).to(DEVICE)

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

    mus = mus.detach().numpy()[-1]
    sigmas = sigmas.detach().numpy()[-1]
    log_coeffs = log_coeffs.detach().numpy()[-1]
    coeffs = numpy_softmax(log_coeffs)
    # all the above are now 3D, minibatch_size x n_components x something
    corrs = corrs.detach().numpy()[-1]
    # corrs is 2D, minibatch_size x n_components
    berns = berns.detach().numpy()[-1]
    # berns is 2D, minibatch_size x 1

    # get attn information
    from IPython import embed; embed(); raise ValueError()
    attn_k = attn_k.detach().numpy()[-1]
    attn_w = attn_w.detach().numpy()[-1]
    attn_phi = attn_phi.detach().numpy()[-1]
    predicted_attn_w.append(attn_w)
    predicted_attn_phi.append(attn_phi)

    sampled_coords = []
    for choose_i in range(coeffs.shape[0]):
        g = random_state.choice(np.arange(coeffs.shape[1]), p=coeffs[choose_i])
        coord_i = sample(mus[choose_i, g, 0], mus[choose_i, g, 1],
                         sigmas[choose_i, g, 0], sigmas[choose_i, g, 1],
                         corrs[choose_i, g],
                         berns[choose_i, 0],
                         random_state=random_state)
        sampled_coords.append(coord_i)
    coords = np.array(sampled_coords)[None]
    predicted_coords.append(coords)

    attn_h_i = attn_h[-1].detach()
    attn_c_i = attn_c[-1].detach()
    attn_k_i = attn_k[-1].detach()
    attn_w_i = attn_w[-1].detach()
    h1_i = h1[-1].detach()
    c1_i = c1[-1].detach()
    h2_i = h2[-1].detach()
    c2_i = c2[-1].detach()

# samples plot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
attn_weights = np.concatenate(predicted_coords, axis=0)
n_plots = 5
for n in range(n_plots):
    fig, ax = plt.subplots(1, 1)
    for stroke in split_strokes(cumsum(traces[:, n])):
        plt.plot(stroke[:, 0], -stroke[:, 1])
    ax.set_title("sampled {}: {}".format(n, chars))
    ax.set_aspect('equal')
    plt.savefig("plot_sample_conditional_handwriting{}.png".format(n))
print("Plotted {} examples, as 'plot_sample_handwriting*.png'".format(n_plots))

predicted_coords[-1][..., 2] = 1.
traces = np.concatenate(predicted_coords, axis=0)
n_plots = 5
for n in range(n_plots):
    fig, ax = plt.subplots(1, 1)
    for stroke in split_strokes(cumsum(traces[:, n])):
        plt.plot(stroke[:, 0], -stroke[:, 1])
    ax.set_title("sampled {}: {}".format(n, chars))
    ax.set_aspect('equal')
    plt.savefig("plot_sample_conditional_handwriting{}.png".format(n))
print("Plotted {} examples, as 'plot_sample_handwriting*.png'".format(n_plots))

'''
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
'''
