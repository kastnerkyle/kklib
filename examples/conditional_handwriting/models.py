from torch.autograd import Variable
import torch.nn as nn
import torch

from kklib.nodes import GLSTM
from kklib.nodes import GLinear
from kklib.nodes import GBiLSTM
from kklib.nodes import GGaussianAttentionLSTM
from kklib.nodes import GCorrGMMAndBernoulli

class Model(nn.Module):
    def __init__(self, minibatch_size, input_size, embed_size, hidden_size, cell_dropout_keep_rate, random_state):
        super(Model, self).__init__()
        self.input_size = input_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.minibatch_size = minibatch_size
        self.random_state = random_state

        init = "truncated_normal"
        self.encoder = nn.Embedding(input_size, embed_size)
        self.proje1 = GLinear([embed_size], hidden_size, random_state=self.random_state, init=init)

        self.projd1 = GLinear([3], hidden_size, random_state=self.random_state, init=init)
        self.attn = GGaussianAttentionLSTM([hidden_size], [hidden_size], hidden_size, n_components=10, attention_scale = 1. / 25.,
                                           cell_dropout_keep_rate=cell_dropout_keep_rate,
                                           shift_decoder_inputs=False, random_state=random_state, init=init)
        self.dec_rnn1 = GLSTM([hidden_size, hidden_size, hidden_size], hidden_size,
                              cell_dropout_keep_rate=cell_dropout_keep_rate, random_state=self.random_state, init=init)
        self.dec_rnn2 = GLSTM([hidden_size, hidden_size, hidden_size], hidden_size,
                              cell_dropout_keep_rate=cell_dropout_keep_rate, random_state=self.random_state, init=init)
        self.output_proj = GCorrGMMAndBernoulli([hidden_size], random_state=self.random_state, init=init)

    def forward(self, enc_in, dec_in, attn_h_i, attn_c_i, attn_k_i, attn_w_i, h1_i, c1_i, h2_i, c2_i, enc_input_mask, dec_input_mask):
        e = self.encoder(enc_in)
        p_e = self.proje1([e])

        p_d1 = self.projd1([dec_in])

        attn_h, attn_c, attn_k, attn_w, attn_phi = self.attn([p_e], [p_d1], attn_h_i, attn_c_i, attn_k_i, attn_w_i, input_mask=enc_input_mask, output_mask=dec_input_mask)
        h1, c1 = self.dec_rnn1([p_d1, attn_w, attn_h], h1_i, c1_i, mask=dec_input_mask)
        h2, c2 = self.dec_rnn2([p_d1, attn_w, h1], h2_i, c2_i, mask=dec_input_mask)
        mus, sigmas, corrs, log_coeffs, berns = self.output_proj([h2])
        return mus, sigmas, corrs, log_coeffs, berns, attn_h, attn_c, attn_k, attn_w, attn_phi, h1, c1, h2, c2

    def make_inits(self):
        return self.attn.make_inits(self.minibatch_size) + self.dec_rnn1.make_inits(self.minibatch_size) + self.dec_rnn2.make_inits(self.minibatch_size)
