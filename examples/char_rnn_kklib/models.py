from torch.autograd import Variable
import torch.nn as nn
import torch

from kklib.nodes import GLinear, GLSTMCell

class RNN(nn.Module):
    def __init__(self, minibatch_size, input_size, hidden_size, output_size, n_layers, batch_size, random_state):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.minibatch_size = minibatch_size
        self.random_state = random_state

        self.encoder = nn.Embedding(input_size, hidden_size)
        self.proj = GLinear([hidden_size], hidden_size, random_state=self.random_state)
        self.cell = GLSTMCell([hidden_size], hidden_size, random_state=self.random_state)

        self.output_proj = GLinear([hidden_size], output_size, random_state=self.random_state)

    def forward(self, x, h, c):
        e = self.encoder(x)
        p = self.proj([e])
        p_h = h
        p_c = c
        res = []
        for i in range(p.shape[0]):
            r, (h, c) = self.cell([p[i]], p_h, p_c)
            p_h = h
            p_c = c
            res.append(r)
        res = torch.stack(res)
        output = self.output_proj([res])
        return output, h, c

    def make_inits(self):
        return self.cell.make_inits(self.minibatch_size)
