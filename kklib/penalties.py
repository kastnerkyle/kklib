import numpy as np
import torch

from .nodes import softmax

def CorrelatedGMMAndBernoulliCost(mus, sigmas, corrs, log_coeffs, bernoullis, true_values_coords, true_values_bernoullis):
    xs = true_values_coords[..., 0]
    ys = true_values_coords[..., 1]
    es = true_values_bernoullis
    batch_size = xs.shape[-1]

    # don't be clever
    buff = (1. - (corrs ** 2.)) + 1E-6
    true_0 = es[..., None]
    true_1 = xs[..., None]
    true_2 = ys[..., None]
    mu_1 = mus[..., 0]
    mu_2 = mus[..., 1]
    sigma_1 = sigmas[..., 0]
    sigma_2 = sigmas[..., 1]
    x_term = (true_1 - mu_1) / sigma_1
    y_term = (true_2 - mu_2) / sigma_2

    Z = (x_term ** 2.) + (y_term ** 2.) - 2. * corrs * x_term * y_term
    N = 1. / (2. * np.pi * sigma_1 * sigma_2 * torch.sqrt(buff)) * torch.exp(-Z / (2. * buff))
    ep = true_0 * bernoullis + (1. - true_0) * (1. - bernoullis)
    assert ep.shape[-1] == 1
    ep = ep[..., 0]
    rp = torch.sum(softmax(log_coeffs) * N, dim=-1)
    nll = -torch.log(rp + 1E-8) - torch.log(ep + 1E-8)
    return nll
