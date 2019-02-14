from torch.autograd import Variable
import torch.nn as nn
import torch

from kklib.nodes import GLSTM
from kklib.nodes import GBiLSTM
from kklib.nodes import GBiLSTMMultiHeadAttentionLSTM
from kklib.nodes import GCorrGMMAndBernoulli

class Model(nn.Module):
    def __init__(self, minibatch_size, input_size, embed_size, hidden_size, random_state):
        super(Model, self).__init__()
        self.input_size = input_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.minibatch_size = minibatch_size
        self.random_state = random_state

        init = "truncated_normal"
        self.encoder = nn.Embedding(input_size, embed_size)
        self.enc_rnn = GBiLSTM([embed_size], hidden_size, random_state=self.random_state, init=init)
        self.attn = GBiLSTMMultiHeadAttentionLSTM([embed_size, hidden_size, hidden_size], [3], hidden_size, n_attention_heads=4, shift_decoder_inputs=False, random_state=random_state, init=init)
        self.dec_rnn1 = GLSTM([hidden_size, hidden_size], hidden_size, random_state=self.random_state, init=init)
        self.dec_rnn2 = GLSTM([hidden_size, hidden_size, hidden_size], hidden_size, random_state=self.random_state, init=init)
        self.output_proj = GCorrGMMAndBernoulli([hidden_size, hidden_size, hidden_size, hidden_size], random_state=self.random_state, init=init)

    def forward(self, x, y, a_h_i, a_c_i, a_i, h1_i, c1_i, h2_i, c2_i, enc_input_mask, dec_input_mask):
        e = self.encoder(x)
        hf, cf, hb, cb = self.enc_rnn([e], mask=enc_input_mask)
        attn_h, attn_c, attn_a_h, all_attn_info = self.attn([e, hf, hb], [y], a_h_i, a_c_i, a_i, input_mask=enc_input_mask, output_mask=dec_input_mask)
        h1, c1 = self.dec_rnn1([attn_h, attn_a_h], h1_i, c1_i, mask=dec_input_mask)
        h2, c2 = self.dec_rnn2([attn_h, attn_a_h, h1], h2_i, c2_i, mask=dec_input_mask)
        mus, sigmas, corrs, log_coeffs, berns = self.output_proj([attn_h, attn_a_h, h1, h2])
        return mus, sigmas, corrs, log_coeffs, berns, attn_h, attn_c, attn_a_h, h1, c1, h2, c2

    def make_inits(self):
        return self.attn.make_inits(self.minibatch_size) + self.dec_rnn1.make_inits(self.minibatch_size) + self.dec_rnn2.make_inits(self.minibatch_size)
