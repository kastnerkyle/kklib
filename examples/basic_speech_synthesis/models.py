from torch.autograd import Variable
import torch.nn as nn
import torch

from kklib.nodes import GLSTM
from kklib.nodes import GLinear
from kklib.nodes import GBiLSTM
from kklib.nodes import GGaussianAttentionLSTM
from kklib.nodes import GCorrGMMAndBernoulli
from kklib.nodes import GSequenceConv1dStack

class Model(nn.Module):
    def __init__(self, minibatch_size, input_size, embed_size, n_stacks, n_filters, prenet_size, enc_hidden_size, dec_hidden_size, output_size, cell_dropout_keep_rate, random_state):
        super(Model, self).__init__()
        self.input_size = input_size
        self.embed_size = embed_size
        self.n_stacks = n_stacks
        self.n_filters = n_filters
        self.prenet_size = prenet_size
        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size
        self.minibatch_size = minibatch_size
        self.random_state = random_state

        init = "truncated_normal"
        self.encoder = nn.Embedding(input_size, embed_size)
        self.conv_stack = GSequenceConv1dStack([embed_size], n_filters,
                                                n_stacks=n_stacks,
                                                kernel_sizes=[(1, 1), (3, 3), (5, 5)],
                                                residual=True,
                                                activation="relu",
                                                use_batch_norm=True,
                                                init=None,
                                                random_state=random_state)
        self.prenet1 = GLinear([output_size], prenet_size, random_state=self.random_state, init=init)
        self.prenet2 = GLinear([prenet_size], enc_hidden_size, random_state=self.random_state, init=init)
        # no cell dropout on encoder
        # could be bug in BiLSTM too...
        #self.enc_proj = GLinear([embed_size, n_filters], enc_hidden_size, random_state=self.random_state, init=init)
        self.enc_rnn = GBiLSTM([embed_size, n_filters], enc_hidden_size,
                                random_state=self.random_state, init=init)
        self.attn = GGaussianAttentionLSTM([enc_hidden_size, enc_hidden_size], [enc_hidden_size], dec_hidden_size,
                                           n_components=10,
                                           attention_scale=1.,
                                           step_op="softplus",
                                           cell_dropout_keep_rate=1., #cell_dropout_keep_rate,
                                           shift_decoder_inputs=False, random_state=random_state, init=init)
        self.dec_rnn1 = GLSTM([enc_hidden_size, 2 * enc_hidden_size, dec_hidden_size], dec_hidden_size,
                              cell_dropout_keep_rate=cell_dropout_keep_rate, random_state=self.random_state, init=init)
        self.dec_rnn2 = GLSTM([enc_hidden_size, 2 * enc_hidden_size, dec_hidden_size], dec_hidden_size,
                              cell_dropout_keep_rate=cell_dropout_keep_rate, random_state=self.random_state, init=init)
        self.output_proj = GLinear([dec_hidden_size], output_size, random_state=self.random_state, init=init)

    def forward(self, x, y, attn_h_i, attn_c_i, attn_k_i, attn_w_i, h1_i, c1_i, h2_i, c2_i, enc_input_mask, dec_input_mask):
        e = self.encoder(x)
        c = self.conv_stack([e], mask=enc_input_mask)
        hf, cf, hb, cb = self.enc_rnn([e, c], mask=enc_input_mask)
        #p_e = self.enc_proj([e, c])

        # apply the prenet
        d_y = nn.functional.dropout(y, 0.5, training=True)
        dp_y1 = nn.functional.relu(self.prenet1([d_y]))
        dp_y2 = nn.functional.dropout(dp_y1, 0.5, training=True)
        p_y2 = nn.functional.relu(self.prenet2([dp_y2]))

        attn_h, attn_c, attn_k, attn_w, attn_phi = self.attn([hf, hb], [p_y2], attn_h_i, attn_c_i, attn_k_i, attn_w_i, input_mask=enc_input_mask, output_mask=dec_input_mask)
        h1, c1 = self.dec_rnn1([p_y2, attn_w, attn_h], h1_i, c1_i, mask=dec_input_mask)
        h2, c2 = self.dec_rnn2([p_y2, attn_w, h1], h2_i, c2_i, mask=dec_input_mask)
        preds = self.output_proj([h2])
        return preds, attn_h, attn_c, attn_k, attn_w, attn_phi, h1, c1, h2, c2

    def make_inits(self):
        return self.attn.make_inits(self.minibatch_size) + self.dec_rnn1.make_inits(self.minibatch_size) + self.dec_rnn2.make_inits(self.minibatch_size)
