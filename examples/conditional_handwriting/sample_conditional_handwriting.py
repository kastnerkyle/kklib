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

coords = np.array([0., 0., 1.])
coords = coords[None] * np.ones((minibatch_size, 1))
coords = coords[None]
coords_mask = 0. * coords[:, :, 0] + 1.

predicted_coords = [coords]
predicted_components = []
predicted_stops = [-1 for _ in range(minibatch_size)]

predicted_attn_k = []
predicted_attn_w = []
predicted_attn_phi = []
predicted_mus = []
predicted_sigmas = []
predicted_coeffs = []
predicted_corrs = []
predicted_berns = []

teacher_force_coords = trace_data[0][:, None] * np.ones((1, minibatch_size, 1))

symbol_to_ind = iamondb["vocabulary"]
ind_to_symbol = {v:k for k, v in symbol_to_ind.items()}
chars = "today is the day"
# add a period so that we know the ending
chars = chars + "."
inds = [symbol_to_ind[c] for c in chars]

#inds = char_data[0]
#chars = "".join([ind_to_symbol[i] for i in inds])

# Only 2D, so embedding works
inds = np.array(inds)[:, None] * np.ones((1, minibatch_size)).astype("int32")
inds_mask = 0 * inds + 1

sample_len = 30 * len(chars)
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

    mus = mus.detach().cpu().numpy()[-1]
    sigmas = sigmas.detach().cpu().numpy()[-1]
    log_coeffs = log_coeffs.detach().cpu().numpy()[-1]
    coeffs = numpy_softmax(log_coeffs)
    # all the above are now 3D, minibatch_size x n_components x something
    corrs = corrs.detach().cpu().numpy()[-1]
    # corrs is 2D, minibatch_size x n_components
    berns = berns.detach().cpu().numpy()[-1]
    # berns is 2D, minibatch_size x 1

    # get attn information for plots
    attn_k_np = attn_k.detach().cpu().numpy()[-1]
    attn_w_np = attn_w.detach().cpu().numpy()[-1]
    attn_phi_np = attn_phi.detach().cpu().numpy()[-1]
    predicted_attn_k.append(attn_k_np)
    predicted_attn_w.append(attn_w_np)
    predicted_attn_phi.append(attn_phi_np)

    stop_scale = 7.5
    for choose_i in range(coeffs.shape[0]):
        thresh = attn_phi_np[choose_i, -1] > stop_scale * np.max(attn_phi_np[choose_i, :-1])
        if thresh and predicted_stops[choose_i] < 0:
            predicted_stops[choose_i] = i

    # sample the handwriting trace
    sampled_coords = []
    sampled_components = []
    for choose_i in range(coeffs.shape[0]):
        g = random_state.choice(np.arange(coeffs.shape[1]), p=coeffs[choose_i])
        coord_i = sample(mus[choose_i, g, 0], mus[choose_i, g, 1],
                         sigmas[choose_i, g, 0], sigmas[choose_i, g, 1],
                         corrs[choose_i, g],
                         berns[choose_i, 0],
                         random_state=random_state)
        sampled_coords.append(coord_i)
        sampled_components.append(g)
    coords = np.array(sampled_coords)[None]
    components = np.array(sampled_components)[None]
    predicted_coords.append(coords)
    predicted_components.append(components)

    # store density information for plots
    predicted_mus.append(mus)
    predicted_sigmas.append(sigmas)
    predicted_coeffs.append(coeffs)
    predicted_corrs.append(corrs)
    predicted_berns.append(berns)

    attn_h_i = attn_h[-1].detach()
    attn_c_i = attn_c[-1].detach()
    attn_k_i = attn_k[-1].detach()
    attn_w_i = attn_w[-1].detach()
    h1_i = h1[-1].detach()
    c1_i = c1[-1].detach()
    h2_i = h2[-1].detach()
    c2_i = c2[-1].detach()

    if all([p >= 0 for p in predicted_stops]):
        print("All samples ended, finishing...")
        break

# samples plot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Deprecated in Matplotlib 2.2, so I'm copying it here
def bivariate_normal(X, Y, sigmax=1.0, sigmay=1.0,
                     mux=0.0, muy=0.0, sigmaxy=0.0):
    Xmu = X-mux
    Ymu = Y-muy

    rho = sigmaxy/(sigmax*sigmay)
    z = Xmu**2/sigmax**2 + Ymu**2/sigmay**2 - 2*rho*Xmu*Ymu/(sigmax*sigmay)
    denom = 2*np.pi*sigmax*sigmay*np.sqrt(1-rho**2)
    return np.exp(-z/(2*(1-rho**2))) / denom

predicted_mus = np.stack(predicted_mus) # timestep batch out_components 2
predicted_sigmas = np.stack(predicted_sigmas) # timestep batch out_components 2
predicted_coeffs = np.stack(predicted_coeffs) # timestep batch out_components
predicted_corrs = np.stack(predicted_corrs) # timestep batch out_components
predicted_berns = np.stack(predicted_berns) # timestep batch 1
attn_phi = np.stack(predicted_attn_phi) # timesteps batch input_step_length
attn_k = np.stack(predicted_attn_k) # timesteps batch attn_components

predicted_components = np.concatenate(predicted_components, axis=0)

# density plots
n_plots = 5
for n in range(n_plots):
    this_stop = predicted_stops[n]
    if this_stop == -1:
        this_stop = this_mus.shape[0]

    this_components = tc = predicted_components[:this_stop, n]

    this_mus = predicted_mus[np.arange(this_stop), n, tc]
    this_sigmas = predicted_sigmas[np.arange(this_stop), n, tc]
    this_coeffs = predicted_coeffs[np.arange(this_stop), n, tc]
    this_corrs = predicted_corrs[np.arange(this_stop), n, tc]
    this_berns = predicted_berns[np.arange(this_stop), n]
    cumulative_mus = np.cumsum(this_mus, axis=0)[:this_stop]

    minx, maxx = np.min(cumulative_mus[:, 0]), np.max(cumulative_mus[:, 0])
    miny, maxy = np.min(cumulative_mus[:, 1]), np.max(cumulative_mus[:, 1])

    # approx 400 pts in each axis resolution
    delta = abs(maxx - minx) / 400.
    x = np.arange(minx, maxx, delta)
    y = np.arange(miny, maxy, delta)
    x_grid, y_grid = np.meshgrid(x, y)
    z_grid = np.zeros_like(x_grid)
    epsilon = 1E-8

    for i in range(this_stop):
        # what
        cov = np.array([[this_sigmas[i, 0], 0.],
                        [0., this_sigmas[i, 1]]])

        gauss = bivariate_normal(x_grid, y_grid, mux=cumulative_mus[i, 0], muy=cumulative_mus[i, 1],
                                 sigmax=cov[0, 0], sigmay=cov[1, 1],
                                 sigmaxy=this_corrs[i] * cov[0, 0] * cov[1, 1])
        # needs to be rho * sigmax * sigmay
        z_grid += gauss * np.power(this_sigmas[i, 0] + this_sigmas[i, 1], 0.4) / (np.max(gauss) + epsilon)
    plt.figure()
    plt.imshow(z_grid, interpolation="bilinear", cmap=cm.jet)
    plt.axes().get_xaxis().set_visible(False)
    plt.axes().get_yaxis().set_visible(False)
    plt.axis("off")
    plt.savefig("plot_density_conditional_handwriting{}.png".format(n))

# attn plots

n_plots = 5
for n in range(n_plots):
    fig, ax = plt.subplots(1, 1)
    this_stop = predicted_stops[n]
    this_phi = attn_phi[:this_stop, n].T[::-1, :]
    plt.figure()
    plt.imshow(this_phi, interpolation='nearest', aspect='auto', cmap=cm.jet)
    plt.yticks(np.arange(0, len(chars[:-1]) + 1))
    plt.axes().set_yticklabels(list(' ' + chars[:-1][::-1]), rotation='vertical', fontsize=8)
    plt.grid(False)
    ax.set_aspect("equal")
    plt.savefig("plot_attention_conditional_handwriting{}.png".format(n))

# stroke plots
predicted_coords[-1][..., 2] = 1.
traces = np.concatenate(predicted_coords, axis=0)
for n in range(n_plots):
    fig, ax = plt.subplots(1, 1)
    this_stop = predicted_stops[n]
    for stroke in split_strokes(cumsum(traces[:this_stop, n])):
        plt.plot(stroke[:, 0], -stroke[:, 1])
    ax.set_title("sampled {}: {}".format(n, chars[:-1]))
    ax.set_aspect('equal')
    plt.savefig("plot_sample_conditional_handwriting{}.png".format(n))
print("Plotted {} examples, as 'plot_*_handwriting*.png'".format(n_plots))
