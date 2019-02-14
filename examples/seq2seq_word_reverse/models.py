from torch.autograd import Variable
import torch.nn as nn
import torch

from kklib.nodes import GLinear, GLSTMCell
from kklib.nodes import GLSTM
from kklib.nodes import GBiLSTM
from kklib.nodes import MultiHeadGlobalAttention
from kklib.nodes import GBiLSTMMultiHeadAttentionLSTM

class Model(nn.Module):
    def __init__(self, minibatch_size, input_size, embed_size, hidden_size, output_size, random_state):
        super(Model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.output_size = output_size
        self.minibatch_size = minibatch_size
        self.random_state = random_state

        self.encoder = nn.Embedding(input_size, hidden_size)
        self.decoder = nn.Embedding(output_size, hidden_size)
        init = "normal"
        self.attn = GBiLSTMMultiHeadAttentionLSTM([hidden_size], [hidden_size], hidden_size, n_attention_heads=1, random_state=random_state, init=init)
        """
        self.enc_rnn = GBiLSTM([hidden_size], hidden_size, random_state=self.random_state)
        self.dec_rnn = GLSTM([hidden_size], hidden_size, random_state=self.random_state)

        self.attention = MultiHeadGlobalAttention([hidden_size], [hidden_size, hidden_size], [hidden_size, hidden_size],
                                                  hidden_size, random_state=random_state)
        # embed the timestep shifted targets
        self.dec_rnn = GLSTM([hidden_size], hidden_size, random_state=self.random_state)
        """
        self.output_proj = GLinear([hidden_size, hidden_size], hidden_size, random_state=self.random_state, init=init)

    def forward(self, x, y, dh, dc, a, input_mask, output_mask):
        e = self.encoder(x)
        d = self.decoder(y)
        raise ValueError("Updated API, need to fix this example")
        output, h, c, oa, all_attn_info = self.attn([e], [d], dh, dc, a, input_mask=input_mask, output_mask=output_mask)
        return output, h, c, oa, all_attn_info

    def make_inits(self):
        return self.attn.make_inits(self.minibatch_size)
