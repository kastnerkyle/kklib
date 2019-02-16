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
all_state_dicts = torch.load(saved_model_path, map_location=DEVICE)
models_file_path = get_saved_model_defs(saved_model_path)
sys.path.insert(0, models_file_path); import models; sys.path.remove(models_file_path)
config_file_path = get_saved_model_config(saved_model_path)
sys.path.insert(0, config_file_path); from config import *; sys.path.remove(config_file_path)
model = models.Model(minibatch_size, 3, hidden_size, random_state).to(DEVICE)
model.load_state_dict(all_state_dicts["Model"])
model = model.to(DEVICE)

h0, c0 = model.make_inits()
h = Variable(h0).to(DEVICE)
c = Variable(c0).to(DEVICE)

sample_len = 512
coords = np.array([0., 0., 1.])
coords = coords[None] * np.ones((minibatch_size, 1))
coords = coords[None]
coords_mask = 0. * coords[:, :, 0] + 1.
predicted_coords = [coords]
for i in range(sample_len):
    print("Sample step {}".format(i))
    input_variable = Variable(torch.FloatTensor(predicted_coords[-1])).to(DEVICE)
    input_mask = Variable(torch.FloatTensor(coords_mask)).to(DEVICE)
    mus, sigmas, corrs, log_coeffs, berns, h1, c1 = model(input_variable,
                                                          h, c,
                                                          input_mask=input_mask)
    mus = mus.detach().numpy()[-1]
    sigmas = sigmas.detach().numpy()[-1]
    log_coeffs = log_coeffs.detach().numpy()[-1]
    coeffs = numpy_softmax(log_coeffs)
    # all the above are now 3D, minibatch_size x n_components x something
    corrs = corrs.detach().numpy()[-1]
    # corrs is 2D, minibatch_size x n_components
    berns = berns.detach().numpy()[-1]
    # berns is 2D, minibatch_size x 1
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
    h = h1[-1]
    c = c1[-1]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
predicted_coords[-1][..., 2] = 1.
traces = np.concatenate(predicted_coords, axis=0)
n_plots = 5
for n in range(n_plots):
    fig, ax = plt.subplots(1, 1)
    for stroke in split_strokes(cumsum(traces[:, n])):
        plt.plot(stroke[:, 0], -stroke[:, 1])
    ax.set_title("sampled {}".format(n))
    ax.set_aspect('equal')
    plt.savefig("plot_sample_handwriting{}.png".format(n))
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
