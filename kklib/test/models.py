from torch.autograd import Variable
import torch.nn as nn
import torch

from kklib.nodes import GLinear, GLSTMCell
from kklib.nodes import GLSTM
from kklib.nodes import GBiLSTM

class Model(nn.Module):
    def __init__(self, minibatch_size, input_size, embed_size, hidden_size, output_size, random_state):
        super(Model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.output_size = output_size
        self.minibatch_size = minibatch_size
        self.random_state = random_state

        self.encoder = nn.Embedding(input_size, hidden_size)
        #self.enc_rnn = GLSTM([hidden_size], hidden_size, random_state=self.random_state)
        self.enc_rnn = GBiLSTM([hidden_size], hidden_size, random_state=self.random_state)

        #self.proj = GLinear([hidden_size], hidden_size, random_state=self.random_state)
        #self.cell = GLSTMCell([hidden_size], hidden_size, random_state=self.random_state)

        self.output_proj = GLinear([hidden_size, hidden_size], output_size, random_state=self.random_state)

    def forward(self, x, hf, cf, hb, cb, input_mask):
        e = self.encoder(x)
        #o, h, c = self.enc_rnn(e, h, c, mask=input_mask)
        #o, h, c = self.enc_rnn(e, mask=input_mask)
        of, hf, cf, ob, hb, cb = self.enc_rnn(e, hf=hf, cf=cf, hb=hb, cb=cb, mask=input_mask)
        output = self.output_proj([of, ob])
        return output, hf, cf, hb, cb

    def make_inits(self):
        return self.enc_rnn.make_inits(self.minibatch_size)
