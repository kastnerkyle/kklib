from torch.autograd import Variable
import torch.nn as nn
import torch

from kklib.nodes import GLSTM
from kklib.nodes import GCorrGMMAndBernoulli

class Model(nn.Module):
    def __init__(self, minibatch_size, input_size, hidden_size, random_state):
        super(Model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.minibatch_size = minibatch_size
        self.random_state = random_state

        init = "normal"
        self.rnn = GLSTM([3], hidden_size, random_state=self.random_state, init=init)
        self.output_proj = GCorrGMMAndBernoulli([hidden_size], random_state=self.random_state, init=init)

    def forward(self, x, h, c, input_mask):
        h, c = self.rnn([x], h, c, mask=input_mask)
        mus, sigmas, corrs, log_coeffs, berns = self.output_proj([h])
        return mus, sigmas, corrs, log_coeffs, berns, h, c

    def make_inits(self):
        return self.rnn.make_inits(self.minibatch_size)
