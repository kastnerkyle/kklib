import torch
from torch import nn
import numpy as np
import uuid
from scipy import linalg
from scipy.stats import truncnorm
from scipy.misc import factorial

from .core import get_logger

logger = get_logger()

def _get_uuid():
    return str(uuid.uuid4())


def exp(x):
    return torch.exp(x)


def softplus(x):
    return nn.functional.softplus(x)


def softmax(x):
    return nn.functional.softmax(x, dim=-1)


def sigmoid(x):
    return torch.sigmoid(x)


def tanh(x):
    return torch.tanh(x)


def relu(x):
    return nn.functional.relu(x)


def np_zeros(shape):
    """
    Builds a numpy variable filled with zeros
    Parameters
    ----------
    shape, tuple of ints
        shape of zeros to initialize
    Returns
    -------
    initialized_zeros, array-like
        Array-like of zeros the same size as shape parameter
    """
    return np.zeros(shape).astype("float32")


def np_normal(shape, random_state, scale=0.01):
    """
    Builds a numpy variable filled with normal random values
    Parameters
    ----------
    shape, tuple of ints or tuple of tuples
        shape of values to initialize
        tuple of ints should be single shape
        tuple of tuples is primarily for convnets and should be of form
        ((n_in_kernels, kernel_width, kernel_height),
         (n_out_kernels, kernel_width, kernel_height))
    random_state, numpy.random.RandomState() object
    scale, float (default 0.01)
        default of 0.01 results in normal random values with variance 0.01
    Returns
    -------
    initialized_normal, array-like
        Array-like of normal random values the same size as shape parameter
    """
    if type(shape[0]) is tuple:
        shp = (shape[1][0], shape[0][0]) + shape[1][1:]
    else:
        shp = shape
    return (scale * random_state.randn(*shp)).astype("float32")


def np_truncated_normal(shape, random_state, scale=0.075):
    """
    Builds a numpy variable filled with truncated normal random values
    Parameters
    ----------
    shape, tuple of ints or tuple of tuples
        shape of values to initialize
        tuple of ints should be single shape
        tuple of tuples is primarily for convnets and should be of form
        ((n_in_kernels, kernel_width, kernel_height),
         (n_out_kernels, kernel_width, kernel_height))
    random_state, numpy.random.RandomState() object
    scale, float (default 0.075)
        default of 0.075
    Returns
    -------
    initialized_normal, array-like
        Array-like of truncated normal random values the same size as shape parameter
    """
    if type(shape[0]) is tuple:
        shp = (shape[1][0], shape[0][0]) + shape[1][1:]
    else:
        shp = shape

    sigma = scale
    lower = -2 * sigma
    upper = 2 * sigma
    mu = 0
    N = np.prod(shp)
    samples = truncnorm.rvs(
              (lower - mu) / float(sigma), (upper - mu) / float(sigma),
              loc=mu, scale=sigma, size=N, random_state=random_state)
    return samples.reshape(shp).astype("float32")


def np_tanh_fan_normal(shape, random_state, scale=1.):
    """
    Builds a numpy variable filled with random values
    Parameters
    ----------
    shape, tuple of ints or tuple of tuples
        shape of values to initialize
        tuple of ints should be single shape
        tuple of tuples is primarily for convnets and should be of form
        ((n_in_kernels, kernel_width, kernel_height),
         (n_out_kernels, kernel_width, kernel_height))
    random_state, numpy.random.RandomState() object
    scale, float (default 1.)
        default of 1. results in normal random values
        with sqrt(2 / (fan in + fan out)) scale
    Returns
    -------
    initialized_fan, array-like
        Array-like of random values the same size as shape parameter
    References
    ----------
    Understanding the difficulty of training deep feedforward neural networks
        X. Glorot, Y. Bengio
    """
    # The . after the 2 is critical! shape has dtype int...
    if type(shape[0]) is tuple:
        kern_sum = np.prod(shape[0]) + np.prod(shape[1])
        shp = (shape[1][0], shape[0][0]) + shape[1][1:]
    else:
        kern_sum = np.sum(shape)
        shp = shape
    var = scale * np.sqrt(2. / kern_sum)
    return var * random_state.randn(*shp).astype("float32")


def np_variance_scaled_uniform(shape, random_state, scale=1.):
    """
    Builds a numpy variable filled with random values
    Parameters
    ----------
    shape, tuple of ints or tuple of tuples
        shape of values to initialize
        tuple of ints should be single shape
        tuple of tuples is primarily for convnets and should be of form
        ((n_in_kernels, kernel_width, kernel_height),
         (n_out_kernels, kernel_width, kernel_height))
    random_state, numpy.random.RandomState() object
    scale, float (default 1.)
        default of 1. results in uniform random values
        with 1 * sqrt(1 / (n_dims)) scale
    Returns
    -------
    initialized_scaled, array-like
        Array-like of random values the same size as shape parameter
    References
    ----------
    Efficient Backprop
        Y. LeCun, L. Bottou, G. Orr, K. Muller
    """
    if type(shape[0]) is tuple:
        shp = (shape[1][0], shape[0][0]) + shape[1][1:]
        kern_sum = np.prod(shape[0])
    else:
        shp = shape
        kern_sum = shape[0]
    #  Make sure bounds aren't the same
    bound = scale * np.sqrt(3. / float(kern_sum))  # sqrt(3) for std of uniform
    return random_state.uniform(low=-bound, high=bound, size=shp).astype(
        "float32")


def np_glorot_uniform(shape, random_state, scale=1.):
    """
    Builds a numpy variable filled with random values
    Parameters
    ----------
    shape, tuple of ints or tuple of tuples
        shape of values to initialize
        tuple of ints should be single shape
        tuple of tuples is primarily for convnets and should be of form
        ((n_in_kernels, kernel_width, kernel_height),
         (n_out_kernels, kernel_width, kernel_height))
    random_state, numpy.random.RandomState() object
    scale, float (default 1.)
        default of 1. results in uniform random values
        with 1. * sqrt(6 / (n_in + n_out)) scale
    Returns
    -------
    initialized_scaled, array-like
        Array-like of random values the same size as shape parameter
    """
    shp = shape
    kern_sum = sum(shp)
    bound = scale * np.sqrt(6. / float(kern_sum))
    return random_state.uniform(low=-bound, high=bound, size=shp).astype(
        "float32")


def np_ortho(shape, random_state, scale=1.):
    """
    Builds a numpy variable filled with orthonormal random values
    Parameters
    ----------
    shape, tuple of ints or tuple of tuples
        shape of values to initialize
        tuple of ints should be single shape
        tuple of tuples is primarily for convnets and should be of form
        ((n_in_kernels, kernel_width, kernel_height),
         (n_out_kernels, kernel_width, kernel_height))
    random_state, numpy.random.RandomState() object
    scale, float (default 1.)
        default of 1. results in orthonormal random values sacled by 1.
    Returns
    -------
    initialized_ortho, array-like
        Array-like of random values the same size as shape parameter
    References
    ----------
    Exact solutions to the nonlinear dynamics of learning in deep linear
    neural networks
        A. Saxe, J. McClelland, S. Ganguli
    """
    if type(shape[0]) is tuple:
        shp = (shape[1][0], shape[0][0]) + shape[1][1:]
        flat_shp = (shp[0], np.prod(shp[1:]))
    else:
        shp = shape
        flat_shp = shape
    g = random_state.randn(*flat_shp)
    U, S, VT = linalg.svd(g, full_matrices=False)
    res = U if U.shape == flat_shp else VT  # pick one with the correct shape
    res = res.reshape(shp)
    return (scale * res).astype("float32")


def make_numpy_biases(bias_dims, name=""):
    logger.info("Initializing {} with {} init".format(name, "zero"))
    #return [np.random.randn(dim,).astype("float32") for dim in bias_dims]
    return [np_zeros((dim,)) for dim in bias_dims]


def make_numpy_weights(in_dim, out_dims, random_state, init=None,
                       scale="default", name=""):
    """
    Will return as many things as are in the list of out_dims
    You *must* get a list back, even for 1 element
    blah, = make_weights(...)
    or
    [blah] = make_weights(...)
    """
    ff = [None] * len(out_dims)
    fs = [scale] * len(out_dims)
    for i, out_dim in enumerate(out_dims):
        if init is None:
            logger.info("Initializing {} with {} init".format(name, "ortho"))
            ff[i] = np_ortho
            fs[i] = 1.
            '''
            if in_dim == out_dim:
                logger.info("Initializing {} with {} init".format(name, "ortho"))
                ff[i] = np_ortho
                fs[i] = 1.
            else:
                logger.info("Initializing {} with {} init".format(name, "variance_scaled_uniform"))
                ff[i] = np_variance_scaled_uniform
                fs[i] = 1.
            '''
        elif init == "ortho":
            logger.info("Initializing {} with {} init".format(name, "ortho"))
            if in_dim != out_dim:
                raise ValueError("Unable to use ortho init for non-square matrices!")
            ff[i] = np_ortho
            fs[i] = 1.
        elif init == "glorot_uniform":
            logger.info("Initializing {} with {} init".format(name, "glorot_uniform"))
            ff[i] = np_glorot_uniform
        elif init == "normal":
            logger.info("Initializing {} with {} init".format(name, "normal"))
            ff[i] = np_normal
            fs[i] = 0.01
        elif init == "truncated_normal":
            logger.info("Initializing {} with {} init".format(name, "truncated_normal"))
            ff[i] = np_truncated_normal
            fs[i] = 0.075
        elif init == "embedding_normal":
            logger.info("Initializing {} with {} init".format(name, "embedding_normal"))
            ff[i] = np_truncated_normal
            fs[i] = 1. / np.sqrt(out_dim)
        else:
            raise ValueError("Unknown init type %s" % init)

    ws = []
    for i, out_dim in enumerate(out_dims):
        if fs[i] == "default":
            wi = ff[i]((in_dim, out_dim), random_state)
            if len(wi.shape) == 4:
                wi = wi.transpose(2, 3, 1, 0)
            ws.append(wi)
        else:
            wi = ff[i]((in_dim, out_dim), random_state, scale=fs[i])
            if len(wi.shape) == 4:
                wi = wi.transpose(2, 3, 1, 0)
            ws.append(wi)
    return ws


def make_torch_weights(in_dim, out_dims, random_state, init=None,
                       scale="default", name=""):
    npw = make_numpy_weights(in_dim, out_dims, random_state, init=init,
                             scale=scale, name=name)
    thw = [torch.FloatTensor(npwi) for npwi in npw]
    return thw


def make_torch_biases(bias_dims, name=""):
    npb = make_numpy_biases(bias_dims, name=name)
    thb = [torch.FloatTensor(npbi) for npbi in npb]
    return thb


def make_torch_zeros(in_dim, out_dims):
    assert hasattr(out_dims, "pop")
    npz = [np.zeros((in_dim, out_dims[i])) for i in range(len(out_dims))]
    thz = [torch.FloatTensor(npzi) for npzi in npz]
    return thz


class GLinear(nn.Module):
    def __init__(self, list_of_input_sizes, output_size, random_state=None, init=None, name=""):
        super(GLinear, self).__init__()
        if random_state is None:
            raise ValueError("random_state argument required for GLinear")
        self.list_of_input_sizes = list_of_input_sizes
        self.output_size = output_size
        input_dim = sum(list_of_input_sizes)
        output_dim = output_size
        self.linear = nn.Linear(input_dim, output_dim)
        # linear stores it "backwards"... outsize, insize
        if name == "":
            name = "GLinear"
        name_w = name + "_w"
        name_b = name + "_b"
        self.name = name

        w = make_torch_weights(input_dim, [output_dim], random_state=random_state, init=init, name=name_w)
        b = make_torch_biases([output_dim], name=name_b)
        # these come as a list from the initializer
        self._set_weights_and_biases(w[0], b[0])

    def _set_weights_and_biases(self, w, b):
        """
        weights of shape (in_dim, out_dim)
        bias of shape (out_dim,)
        """
        self.linear.weight.data = w.transpose(1, 0).contiguous()
        self.linear.bias.data = b.contiguous()

    def forward(self, list_of_inputs):
        if len(list_of_inputs) == 1:
             x = list_of_inputs[0]
        else:
            x = torch.cat(list_of_inputs, dim=-1)
        x_orig = x
        last_axis = x.size(-1)
        x = x.view(-1, last_axis)
        l_o = self.linear(x)
        return l_o.view(*list(x_orig.size())[:-1] + [self.output_size])


class GCorrGMMAndBernoulli(nn.Module):
    def __init__(self, list_of_input_sizes, output_size=2, n_components=20, sigma_eps=1E-4, random_state=None, init=None, name=""):
        super(GCorrGMMAndBernoulli, self).__init__()
        if random_state is None:
            raise ValueError("random_state argument required for GLinear")
        self.list_of_input_sizes = list_of_input_sizes
        self.random_state = random_state
        self.output_size = output_size
        self.n_components = n_components
        self.init = init
        self.sigma_eps = sigma_eps
        # mu of output_size for each component
        # sigma of output_size for each component
        # correlation for each component
        # mixture coeffs for each component
        # bernoulli output of dim 1
        self.full_output_size = n_components * output_size + n_components * output_size + n_components + n_components + 1
        self.proj =  GLinear(list_of_input_sizes,
                             self.full_output_size,
                             random_state=random_state,
                             init=init)

    def forward(self, list_of_inputs):
        o = self.proj(list_of_inputs)
        mus = o[..., :self.n_components * self.output_size]
        mus = mus.view(mus.shape[0], mus.shape[1], self.n_components, self.output_size)

        sigmas = o[..., self.n_components * self.output_size:2 * self.n_components * self.output_size]
        sigmas = sigmas.view(sigmas.shape[0], sigmas.shape[1], self.n_components, self.output_size)
        sigmas = softplus(sigmas) + self.sigma_eps

        corrs = o[..., 2 * self.n_components * self.output_size:2 * self.n_components * self.output_size + self.n_components]
        corrs = tanh(corrs)

        log_coeffs = o[..., 2 * self.n_components * self.output_size + self.n_components:2 * self.n_components * self.output_size + self.n_components + self.n_components]

        berns = o[..., -1][..., None]
        berns = sigmoid(berns)
        return mus, sigmas,  corrs, log_coeffs, berns


class GLSTMCell(nn.Module):
    def __init__(self, list_of_input_sizes, num_units, random_state=None, init=None, cell_dropout_keep_rate=None, name=""):
        super(GLSTMCell, self).__init__()
        if random_state is None:
            raise ValueError("random_state argument required for GLinear")

        if name == "":
            name = "GLSTMCell"
        name_w = name + "_lstm_proj_w"
        name_b = name + "_lstm_proj_b"
        name_proj = name + "_lstm_proj"
        self.name = name
        self.cell_dropout_keep_rate = cell_dropout_keep_rate
        if cell_dropout_keep_rate is None:
            cell_dropout_keep_rate = 1.
        self._cell_dropout_rate = 1. - cell_dropout_keep_rate

        if init is None:
            inp_init = None
            h_init = None
            out_init = None
        elif init == "truncated_normal":
            inp_init = "truncated_normal"
            h_init = "truncated_normal"
            out_init = "truncated_normal"
        elif init == "glorot_uniform":
            inp_init = "glorot_uniform"
            h_init = "glorot_uniform"
            out_init = "glorot_uniform"
        elif init == "normal":
            inp_init = "normal"
            h_init = "normal"
            out_init = "normal"
        else:
            raise ValueError("Unknown init argument {}".format(init))

        input_dim = sum(list_of_input_sizes)
        hidden_dim = 4 * num_units

        self.num_units = num_units
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        comb_w, = make_torch_weights(input_dim + num_units, [hidden_dim],
                                     random_state=random_state,
                                     init=inp_init, name=name_w)
        comb_b, = make_torch_biases([hidden_dim], name=name_b)
        # set initial forget gate bias high
        comb_b[2 * num_units:3 * num_units] *= 0.
        comb_b[2 * num_units:3 * num_units] += 1.

        self.lstm_proj = GLinear(list_of_input_sizes + [num_units],
                                 hidden_dim,
                                 random_state=random_state,
                                 name=name_proj, init=init)
        self.lstm_proj._set_weights_and_biases(comb_w, comb_b)
        self.cell_drop_layer = nn.Dropout(p=self._cell_dropout_rate)

    def _slice(self, arr, size, index, axis=-1):
        if axis != -1:
            raise ValueError("axis argument to _slice NYI")
        return arr[..., index * size:(index + 1) * size]

    def forward(self, list_of_inputs, previous_hidden, previous_cell, mask=None):
        l_p = self.lstm_proj(list_of_inputs + [previous_hidden])
        i = self._slice(l_p, self.num_units, 0)
        j = self._slice(l_p, self.num_units, 1)
        f = self._slice(l_p, self.num_units, 2)
        o = self._slice(l_p, self.num_units, 3)
        if self.cell_dropout_keep_rate is not None:
            pj = self.cell_drop_layer(tanh(j))
        else:
            pj = tanh(j)
        # this was f + 1. in the tensorflow versions instead of just setting in init bias, might be important?
        c = sigmoid(f) * previous_cell + sigmoid(i) * pj
        if mask is not None:
            mask = mask.type(c.dtype)
            c = mask[:, None] * c + (1 - mask[:, None]) * previous_cell
        h = sigmoid(o) * tanh(c)
        if mask is not None:
            mask = mask.type(h.dtype)
            h = mask[:, None] * h + (1 - mask[:, None]) * h
        return (h, c)

    def make_inits(self, minibatch_size):
        return make_torch_zeros(minibatch_size, [self.num_units, self.num_units])


class GLSTM(nn.Module):
    def __init__(self, list_of_input_sizes, num_units, random_state=None, init=None, proj_init=None, cell_init=None, cell_dropout_keep_rate=None, name=""):
        super(GLSTM, self).__init__()
        self.list_of_input_sizes = list_of_input_sizes
        self.num_units = num_units
        self.random_state = random_state
        self.proj_init = proj_init
        self.cell_init = cell_init
        if init is not None:
            self.proj_init = init
            self.cell_init = init
            proj_init = init
            cell_init = init
        self.cell_dropout_keep_rate = cell_dropout_keep_rate

        self.proj = GLinear(list_of_input_sizes, num_units, random_state=self.random_state, init=proj_init)
        self.cell = GLSTMCell([num_units], num_units, random_state=self.random_state, init=cell_init, cell_dropout_keep_rate=cell_dropout_keep_rate)

    def forward(self, list_of_inputs, h=None, c=None, mask=None):
        """
        h and c are previous hidden and cell states, optional
        mask is the minibatch mask, optional
        """
        p = self.proj(list_of_inputs)
        if h is None:
            p_h = 0. * p[0, :, :self.num_units]
        else:
            p_h = h
        if c is None:
            p_c = 0. * p[0, :, :self.num_units]
        else:
            p_c = c
        if mask is None:
            mask = 1. + 0. * p[:, :, 0]
            mask = mask.type(p.dtype)

        all_h = []
        all_c = []
        for i in range(p.shape[0]):
            h, c = self.cell([p[i]], p_h, p_c, mask=mask[i])
            p_h = h
            p_c = c
            all_h.append(h)
            all_c.append(c)
        all_h = torch.stack(all_h)
        all_c = torch.stack(all_c)
        return all_h, all_c

    def make_inits(self, minibatch_size):
        return self.cell.make_inits(minibatch_size)

class GBiLSTM(nn.Module):
    def __init__(self, list_of_input_sizes, num_units, random_state=None, init=None, proj_init=None, cell_init=None, cell_dropout_keep_rate=None, name=""):
        super(GBiLSTM, self).__init__()
        self.list_of_input_sizes = list_of_input_sizes
        self.num_units = num_units
        self.random_state = random_state

        self.proj_init = proj_init
        self.cell_init = cell_init
        if init is not None:
            self.proj_init = init
            self.cell_init = init
            proj_init = init
            cell_init = init
        self.cell_dropout_keep_rate = cell_dropout_keep_rate

        self.projf = GLinear(list_of_input_sizes, num_units, random_state=self.random_state, init=proj_init)
        self.projb = GLinear(list_of_input_sizes, num_units, random_state=self.random_state, init=proj_init)
        self.cellf = GLSTMCell([num_units], num_units, random_state=self.random_state, init=cell_init, cell_dropout_keep_rate=cell_dropout_keep_rate)
        self.cellb = GLSTMCell([num_units], num_units, random_state=self.random_state, init=cell_init, cell_dropout_keep_rate=cell_dropout_keep_rate)

    def forward(self, list_of_inputs, hf=None, cf=None, hb=None, cb=None, mask=None):
        # do it the fully proper way
        # may decide to just do the flip mask and flip hiddens version later
        pf = self.projf(list_of_inputs)
        pb = self.projb(list_of_inputs)
        if hf is None:
            p_hf = 0. * pf[0, :, :self.num_units]
        else:
            p_hf = hf

        if cf is None:
            p_cf = 0. * pf[0, :, :self.num_units]
        else:
            p_cf = cf

        if hb is None:
            p_hb = 0. * pb[0, :, :self.num_units]
        else:
            p_hb = hb

        if cb is None:
            p_cb = 0. * pb[0, :, :self.num_units]
        else:
            p_cb = cb

        if mask is None:
            mask = 1. + 0. * pf[:, :, 0]
            mask = mask.type(pf.dtype)

        # pf and pb are the same shape
        flip_pb = 0. * pb
        flip_idxs = []
        for n, v in enumerate((mask > 0).sum(dim=0)):
            # get the inverse indexes, combine with the forward ones
            # 4 3 2 1 0 5 6 7 etc
            # we iterate through the sequence backwards BUT IGNORE THE MASK AND KEEP AT THE END, making this quite annoying
            idxs = torch.arange(0, pf.shape[0])
            part_idxs = torch.arange(0, v) # can't use subidx cause they ref the same thing
            flip_part_idxs = torch.flip(part_idxs, dims=[0]) # can't use subidx cause they ref the same thing
            idxs[:v] *= 0
            idxs[:v] += flip_part_idxs
            # now it should be the correct indexing...
            # test with tmp array
            #tmp = torch.arange(0, pf.shape[0])
            #new_tmp = 0. * pf
            #new_tmp = new_tmp + tmp[:, None, None].type(torch.FloatTensor)
            #flip_pb[torch.arange(pb.shape[0]), n, :] = new_tmp[idxs, n, :] #pb[flip_idxs, n, :]
            flip_pb[torch.arange(pb.shape[0]), n, :] += pb[idxs, n, :] #pb[flip_idxs, n, :]
            flip_idxs.append(idxs)

        all_hf = []
        all_cf = []
        all_hb = []
        all_cb = []
        for i in range(pf.shape[0]):
            hf, cf = self.cellf([pf[i]], p_hf, p_cf, mask=mask[i])
            # mask is the same because we flipped the sequence
            hb, cb = self.cellb([flip_pb[i]], p_hb, p_cb, mask=mask[i])
            p_hf = hf
            p_cf = cf
            p_hb = hb
            p_cb = cb

            all_hf.append(hf)
            all_cf.append(cf)

            all_hb.append(hb)
            all_cb.append(cb)
        all_hf = torch.stack(all_hf)
        all_cf = torch.stack(all_cf)

        all_hb = torch.stack(all_hb)
        all_cb = torch.stack(all_cb)

        # flip it back
        final_all_hb = 0. * all_hb
        final_all_cb = 0. * all_cb
        for n in range(len(flip_idxs)):
            final_all_hb[torch.arange(pb.shape[0]), n, :] += all_hb[flip_idxs[n], n, :]
            final_all_cb[torch.arange(pb.shape[0]), n, :] += all_cb[flip_idxs[n], n, :]
        return all_hf, all_cf, final_all_hb, final_all_cb

    def make_inits(self, minibatch_size):
        return self.cellf.make_inits(minibatch_size) + self.cellb.make_inits(minibatch_size)


class MultiHeadGlobalAttention(nn.Module):
    def __init__(self, list_of_query_sizes, list_of_key_sizes, list_of_value_sizes, hidden_size, n_attention_heads=8, random_state=None, init=None, name=""):
        raise ValueError("DO IT LIKE GAUSSIAN ATTN?")
        # THIS CLASS IS NOT APPROPRIATE FOR TRANSFORMER YET
        # http://phontron.com/class/nn4nlp2017/assets/slides/nn4nlp-09-attention.pdf
        # http://nlp.seas.harvard.edu/2018/04/03/attention.html#attention
        # Typically query from decoder, key-value from encoder
        super(MultiHeadGlobalAttention, self).__init__()
        self.random_state = random_state
        query_dim = sum(list_of_query_sizes)
        self.query_dim = query_dim
        key_dim = sum(list_of_key_sizes)
        self.key_dim = key_dim
        value_dim = sum(list_of_value_sizes)
        self.value_dim = value_dim
        self.hidden_size = hidden_size
        if key_dim != value_dim:
            print("key_dim != value_dim not yet supported, got {} and {}".format(list_of_key_sizes, list_of_value_sizes))

        assert query_dim % n_attention_heads == 0
        assert key_dim % n_attention_heads == 0
        assert value_dim % n_attention_heads == 0
        self.n_attention_heads = n_attention_heads
        self.d_k = self.hidden_size // n_attention_heads
        self.key_proj = GLinear(list_of_key_sizes, hidden_size, random_state=self.random_state, init=init)
        self.value_proj = GLinear(list_of_value_sizes, hidden_size, random_state=self.random_state, init=init)
        self.query_proj = GLinear(list_of_query_sizes, hidden_size, random_state=self.random_state, init=init)
        self.out_proj = GLinear([hidden_size], hidden_size, random_state=self.random_state, init=init)

    def _mhattention(self, query, key, value, mask=None):
        # query is minibatch_size, n_attn_heads, 1, d_k
        # key and value are minibatch_size, n_attn_heads, timesteps, d_k
        # returns minibatch_size, n_attn_heads, 1, d_k
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.d_k ** .5)
        if mask is not None:
            # pre-fill with hugely negative value for softmaxing
            scores = scores.masked_fill(mask.transpose(0, 1)[:, None, None, :] == 0, -1E9)
        p_attn = nn.functional.softmax(scores, dim=-1)
        return torch.matmul(p_attn, value), p_attn

    def forward(self, list_of_queries, list_of_keys, list_of_values, mask=None):
        n_minibatch = list_of_queries[0].shape[0]
        p_q = self.query_proj(list_of_queries).view(-1, n_minibatch, self.n_attention_heads, self.d_k)
        p_k = self.key_proj(list_of_keys).view(-1, n_minibatch, self.n_attention_heads, self.d_k)
        p_v = self.value_proj(list_of_values).view(-1, n_minibatch, self.n_attention_heads, self.d_k)
        # https://blog.floydhub.com/the-transformer-in-pytorch/
        # put batch in front, then swap time to axis 2
        p_q = p_q.transpose(0, 1).transpose(1, 2)
        p_k = p_k.transpose(0, 1).transpose(1, 2)
        p_v = p_v.transpose(0, 1).transpose(1, 2)
        attn_h, attn_w = self._mhattention(p_q, p_k, p_v, mask=mask)
        comb = attn_h.transpose(1, 2).contiguous().view(n_minibatch, -1, self.hidden_size)
        # should go from minibatch_size, 1, hidden_size to minibatch, hidden_size
        comb = comb.squeeze()
        # make the attention weights time, batch, heads
        attn_w = attn_w.transpose(0, 1).transpose(0, -1).squeeze().contiguous()
        return self.out_proj([comb]), [attn_w,]


class GGaussianAttentionCell(nn.Module):
    def __init__(self, list_of_query_sizes, list_of_key_sizes, hidden_size, n_components=10, step_op="exp", attention_scale=1., softmax_energy=False, cell_dropout_keep_rate=None, random_state=None, init=None, name=""):
        # only query key
        # no query, key, value here
        super(GGaussianAttentionCell, self).__init__()
        self.random_state = random_state
        query_dim = sum(list_of_query_sizes)
        self.query_dim = query_dim
        key_dim = sum(list_of_key_sizes)
        self.key_dim = key_dim
        self.n_components = n_components

        if step_op == "exp":
            self.step_op = exp
        elif step_op == "softplus":
            self.step_op = softplus

        self.attention_scale = attention_scale
        self.softmax_energy = softmax_energy
        self.cell_dropout_keep_rate = cell_dropout_keep_rate

        self.attn_cell = GLSTMCell(list_of_query_sizes + [self.key_dim,], hidden_size, random_state=random_state, init=init,
                                   cell_dropout_keep_rate=cell_dropout_keep_rate)
        self.proj = GLinear([hidden_size], 3 * n_components, random_state=random_state, init=init)
        # convenience projection
        self.key_proj = GLinear(list_of_key_sizes, key_dim, random_state=random_state, init=init)

    def _calc_phi(self, lk_t, la_t, lb_t, lu):
        la_t = la_t[..., None]
        lb_t = lb_t[..., None]
        lk_t = lk_t[..., None]# + np.log(self.attention_scale)
        if not self.softmax_energy:
            # instant nan
            core = ((lk_t - lu) ** 2)
            phi = exp(-core * lb_t) * la_t
            # use softplus? no longer technically gaussian in that case...
            #phi = softplus(-((lk_t - lu) ** 2) * lb_t) * la_t
            #phi = softmax(-((lk_t - lu) ** 2) * lb_t * la_t)
        else:
            raise ValueError("softmax energy still needs fixing")
            phi = softmax(-((lk_t - lu) ** 2) * lb_t) * la_t
        # phi is minibatch, n_components, conditioning_timesteps
        phi = torch.sum(phi, dim=1)[:, None]
        # now phi is minibatch, 1, conditioning_timesteps
        return phi

    def forward(self, list_of_queries, list_of_keys, previous_hidden, previous_cell, previous_attention_position, previous_attention_weight,
                query_mask=None, key_mask=None):
        # returns h, c, k, w, phi

        # keys is the FULL conditioning tensor
        n_minibatch = list_of_queries[0].shape[0]
        h_o, c_o = self.attn_cell(list_of_queries + [previous_attention_weight], previous_hidden, previous_cell)
        ret = self.proj([h_o])
        a_t = ret[..., :self.n_components]
        b_t = ret[..., self.n_components:2 * self.n_components]
        k_t = ret[..., 2 * self.n_components:]

        a_t = torch.exp(a_t)
        b_t = torch.exp(b_t)

        k_tm1 = previous_attention_position
        step_size = self.attention_scale * self.step_op(k_t)
        k_t = k_tm1 + step_size

        ctx = self.key_proj(list_of_keys)
        u = torch.arange(ctx.shape[0], dtype=ctx.dtype, device=ctx.device)
        u = u[None, None, :]
        # phi is minibatch, 1, conditioning_timesteps
        phi_t = self._calc_phi(k_t, a_t, b_t, u)
        if key_mask is None:
            key_mask = 0. * ctx[..., 0] + 1.
        key_mask = key_mask.type(ctx.dtype)
        w_t_pre = phi_t * ctx.transpose(1, 2).transpose(2, 0)
        w_t_masked = w_t_pre * key_mask.transpose(1, 0)[:, None]
        w_t = torch.sum(w_t_masked, dim=-1)[:, None]
        """
        # tf version
        if conditioning_mask is not None:
            w_t_pre = phi_t * tf.transpose(ctx, (1, 2, 0))
            w_t_masked = w_t_pre * (tf.transpose(ctx_mask, (1, 0))[:, None])
            w_t = tf.reduce_sum(w_t_masked, axis=-1)[:, None]
        else:
            w_t = tf.matmul(phi_t, tf.transpose(ctx, (1, 0, 2)))
        """
        # now minibatch, input_timesteps
        phi_t = phi_t[:, 0]
        # now minibatch, context features
        w_t = w_t[:, 0]
        return h_o, c_o, k_t, w_t, phi_t

    def make_inits(self, minibatch_size):
        h_i, c_i = self.attn_cell.make_inits(minibatch_size)
        att_w_init, att_k_init = make_torch_zeros(minibatch_size, [self.query_dim, self.n_components])
        return [h_i, c_i, att_k_init, att_w_init]


class GGaussianAttentionLSTM(nn.Module):
    def __init__(self, list_of_encoder_input_sizes, list_of_decoder_input_sizes, hidden_size, n_components=10, step_op="exp", attention_scale=1., cell_dropout_keep_rate=None, shift_decoder_inputs=False, random_state=None, init=None, name=""):
        super(GGaussianAttentionLSTM, self).__init__()
        encoder_input_size = sum(list_of_encoder_input_sizes)
        self.encoder_input_size = encoder_input_size
        decoder_input_size = sum(list_of_decoder_input_sizes)
        self.decoder_input_size = decoder_input_size
        self.hidden_size = hidden_size
        self.n_components = n_components
        self.step_op = step_op
        self.attention_scale = attention_scale
        self.cell_dropout_keep_rate = cell_dropout_keep_rate
        self.shift_decoder_inputs = shift_decoder_inputs

        self.random_state = random_state
        self.init = init
        self.name = name

        self.attention_cell = GGaussianAttentionCell(list_of_decoder_input_sizes, list_of_encoder_input_sizes,
                                                     hidden_size, n_components=n_components,
                                                     step_op=step_op,
                                                     attention_scale=attention_scale,
                                                     cell_dropout_keep_rate=cell_dropout_keep_rate,
                                                     random_state=random_state, init=init)

    def forward(self, list_of_encoder_inputs, list_of_decoder_inputs, decoder_initial_hidden, decoder_initial_cell, attention_init_position, attention_init_weight, input_mask=None, output_mask=None):

        # this trick also ensures we work with input length 1 in generation, potentially...
        # THIS MIGHT HAVE TO BE CHANGED BETWEEN TRAINING AND TEST, STATE IS PAIN
        if self.shift_decoder_inputs:
            shifted = []
            for i in range(len(list_of_decoder_inputs)):
                d_ = list_of_decoder_inputs[i]
                shift_ = 0 * torch.cat([d_, d_])
                # starts with 0
                shift_[1:d_.shape[0]] = d_[:-1]
                shift_ = shift_[:d_.shape[0]]
                shifted.append(shift_)
        else:
            shifted = list_of_decoder_inputs

        # this is where teacher forcing dropout tricks could happen
        # do RNN before, that then controls the attention distribution
        all_h = []
        all_c = []
        all_a_k = []
        all_a_w = []
        all_a_phi = []
        h = decoder_initial_hidden
        c = decoder_initial_cell
        a_k = attention_init_position
        a_w = attention_init_weight

        for i in range(shifted[0].shape[0]):
            s_ = [shifted[k][i] for k in range(len(shifted))]
            #a_h, attn_info = self.attention([h], [he],  mask=input_mask)
            h_t, c_t, a_k_t, a_w_t, a_phi_t = self.attention_cell(s_, list_of_encoder_inputs,
                                                                  h, c, a_k, a_w,
                                                                  query_mask=output_mask,
                                                                  key_mask=input_mask)
            h = h_t
            c = c_t
            a_k = a_k_t
            a_w = a_w_t
            a_phi = a_phi_t

            all_h.append(h)
            all_c.append(c)
            all_a_k.append(a_k)
            all_a_w.append(a_w)
            all_a_phi.append(a_phi)
        all_h = torch.stack(all_h)
        all_c = torch.stack(all_c)
        all_a_k = torch.stack(all_a_k)
        all_a_w = torch.stack(all_a_w)
        all_a_phi = torch.stack(all_a_phi)
        return all_h, all_c, all_a_k, all_a_w, all_a_phi

    def make_inits(self, minibatch_size):
        return self.attention_cell.make_inits(minibatch_size)


class GLSTMMultiHeadAttentionLSTM(nn.Module):
    def __init__(self, list_of_encoder_input_sizes, list_of_decoder_input_sizes, hidden_size, n_attention_heads=8, cell_dropout_keep_rate=None, shift_decoder_inputs=False, random_state=None, init=None, name=""):
        super(GBiLSTMMultiHeadAttentionLSTM, self).__init__()
        encoder_input_size = sum(list_of_encoder_input_sizes)
        self.encoder_input_size = encoder_input_size
        decoder_input_size = sum(list_of_decoder_input_sizes)
        self.decoder_input_size = decoder_input_size
        self.hidden_size = hidden_size
        self.n_attention_heads = n_attention_heads
        self.cell_dropout_keep_rate = cell_dropout_keep_rate
        self.shift_decoder_inputs = shift_decoder_inputs

        self.random_state = random_state
        self.init = init
        self.name = name

        self.enc_rnn = GLSTM(list_of_encoder_input_sizes, hidden_size, random_state=self.random_state, init=init, cell_dropout_keep_rate=cell_dropout_keep_rate)

        self.dec_rnn_cell = GLSTMCell(list_of_decoder_input_sizes + [hidden_size,], hidden_size, random_state=self.random_state, init=init, cell_dropout_keep_rate=cell_dropout_keep_rate)

        self.attention = MultiHeadGlobalAttention([hidden_size], [hidden_size], [hidden_size],
                                                  hidden_size, n_attention_heads=n_attention_heads,
                                                  random_state=random_state, init=init)

    def forward(self, list_of_encoder_inputs, list_of_decoder_inputs, decoder_initial_hidden, decoder_initial_cell, attention_init, input_mask=None, output_mask=None):

        he, ce = self.enc_rnn(list_of_encoder_inputs, mask=input_mask)

        # this trick also ensures we work with input length 1 in generation, potentially...
        # THIS MIGHT HAVE TO BE CHANGED BETWEEN TRAINING AND TEST, STATE IS PAIN
        if self.shift_decoder_inputs:
            shifted = []
            for i in range(len(list_of_decoder_inputs)):
                d_ = list_of_decoder_inputs[i]
                shift_ = 0 * torch.cat([d_, d_])
                # starts with 0
                shift_[1:d_.shape[0]] = d_[:-1]
                shift_ = shift_[:d_.shape[0]]
                shifted.append(shift_)
        else:
            shifted = list_of_decoder_inputs
        # this is where teacher forcing dropout tricks could happen
        # do RNN before, that then controls the attention distribution
        all_h = []
        all_c = []
        all_a_h = []
        all_attn_info = []
        h = decoder_initial_hidden
        c = decoder_initial_cell
        a_h = attention_init

        for i in range(shifted[0].shape[0]):
            s_ = [shifted[k][i] for k in range(len(shifted))]
            h, c = self.dec_rnn_cell(s_ + [a_h], h, c, mask=output_mask[i])
            a_h, attn_info = self.attention([h], [he], [he], mask=input_mask)
            all_h.append(h)
            all_c.append(c)
            all_a_h.append(a_h)
            # assume we know the attention info outputs
            all_attn_info.append(attn_info[0])
        all_a_h = torch.stack(all_a_h)
        all_attn_info = torch.stack(all_attn_info)
        all_h = torch.stack(all_h)
        all_c = torch.stack(all_c)
        return all_h, all_c, all_a_h, all_attn_info

    def make_inits(self, minibatch_size):
        i_h, i_c = self.dec_rnn_cell.make_inits(minibatch_size)
        return [i_h, i_c, 0. * i_h]


class GBiLSTMMultiHeadAttentionLSTM(nn.Module):
    def __init__(self, list_of_encoder_input_sizes, list_of_decoder_input_sizes, hidden_size, n_attention_heads=8, cell_dropout_keep_rate=None, shift_decoder_inputs=False, random_state=None, init=None, name=""):
        super(GBiLSTMMultiHeadAttentionLSTM, self).__init__()
        encoder_input_size = sum(list_of_encoder_input_sizes)
        self.encoder_input_size = encoder_input_size
        decoder_input_size = sum(list_of_decoder_input_sizes)
        self.decoder_input_size = decoder_input_size
        self.hidden_size = hidden_size
        self.n_attention_heads = n_attention_heads
        self.cell_dropout_keep_rate = cell_dropout_keep_rate
        self.shift_decoder_inputs = shift_decoder_inputs

        self.random_state = random_state
        self.init = init
        self.name = name

        self.enc_rnn = GBiLSTM(list_of_encoder_input_sizes, hidden_size, random_state=self.random_state, init=init, cell_dropout_keep_rate=cell_dropout_keep_rate)

        self.dec_rnn_cell = GLSTMCell(list_of_decoder_input_sizes + [hidden_size,], hidden_size, random_state=self.random_state, init=init, cell_dropout_keep_rate=cell_dropout_keep_rate)

        self.attention = MultiHeadGlobalAttention([hidden_size], [hidden_size, hidden_size], [hidden_size, hidden_size],
                                                  hidden_size, n_attention_heads=n_attention_heads,
                                                  random_state=random_state, init=init)
        self.output_proj = GLinear([hidden_size, hidden_size], hidden_size, random_state=self.random_state, init=init)

    def forward(self, list_of_encoder_inputs, list_of_decoder_inputs, decoder_initial_hidden, decoder_initial_cell, attention_init, input_mask=None, output_mask=None):

        hf, cf, hb, cb = self.enc_rnn(list_of_encoder_inputs, hf=None, cf=None, hb=None, cb=None, mask=input_mask)

        # this trick also ensures we work with input length 1 in generation, potentially...
        # THIS MIGHT HAVE TO BE CHANGED BETWEEN TRAINING AND TEST, STATE IS PAIN
        if self.shift_decoder_inputs:
            shifted = []
            for i in range(len(list_of_decoder_inputs)):
                d_ = list_of_decoder_inputs[i]
                shift_ = 0 * torch.cat([d_, d_])
                # starts with 0
                shift_[1:d_.shape[0]] = d_[:-1]
                shift_ = shift_[:d_.shape[0]]
                shifted.append(shift_)
        else:
            shifted = list_of_decoder_inputs
        # this is where teacher forcing dropout tricks could happen
        # do RNN before, that then controls the attention distribution
        all_h = []
        all_c = []
        all_a_h = []
        all_attn_info = []
        h = decoder_initial_hidden
        c = decoder_initial_cell
        a_h = attention_init

        for i in range(shifted[0].shape[0]):
            s_ = [shifted[k][i] for k in range(len(shifted))]
            h, c = self.dec_rnn_cell(s_ + [a_h], h, c, mask=output_mask[i])
            a_h, attn_info = self.attention([h], [hf, hb], [hf, hb], mask=input_mask)
            all_h.append(h)
            all_c.append(c)
            all_a_h.append(a_h)
            # assume we know the attention info outputs
            all_attn_info.append(attn_info[0])
        all_a_h = torch.stack(all_a_h)
        all_attn_info = torch.stack(all_attn_info)
        all_h = torch.stack(all_h)
        all_c = torch.stack(all_c)
        return all_h, all_c, all_a_h, all_attn_info

    def make_inits(self, minibatch_size):
        i_h, i_c = self.dec_rnn_cell.make_inits(minibatch_size)
        return [i_h, i_c, 0. * i_h]
