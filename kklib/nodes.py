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


def sigmoid(x):
    return torch.sigmoid(x)


def Sigmoid(x):
    return sigmoid(x)


def tanh(x):
    return torch.tanh(x)


def Tanh(x):
    return tanh(x)


def relu(x):
    return nn.functional.relu(x)


def ReLU(x):
    return relu(x)


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
    def __init__(self, list_of_input_sizes, output_size, random_state=None, name=""):
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

        w = make_torch_weights(input_dim, [output_dim], random_state, name=name_w)
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
                                 name=name_proj)
        self.lstm_proj._set_weights_and_biases(comb_w, comb_b)

    def _slice(self, arr, size, index, axis=-1):
        if axis != -1:
            raise ValueError("axis argument to _slice NYI")
        return arr[..., index * size:(index + 1) * size]

    def forward(self, list_of_inputs, previous_hidden, previous_cell, input_mask=None):
        l_p = self.lstm_proj(list_of_inputs + [previous_hidden])
        i = self._slice(l_p, self.num_units, 0)
        j = self._slice(l_p, self.num_units, 1)
        f = self._slice(l_p, self.num_units, 2)
        o = self._slice(l_p, self.num_units, 3)
        if self.cell_dropout_keep_rate is not None:
            raise ValueError("cell_dropout NYI")
        else:
            pj = tanh(j)
        # this was f + 1. in the tensorflow versions
        c = sigmoid(f) * previous_cell + sigmoid(i) * pj
        if input_mask is not None:
            #c = input_mask[:, None] * c + (1. - input_mask[:, None]) * previous_cell
            raise ValueError("input_mask NYI")
        h = sigmoid(o) * tanh(c)
        if input_mask is not None:
            #h = input_mask[:, None] * h + (1. - input_mask[:, None]) * h
            raise ValueError("input_mask NYI")
        return h, (h, c)

    def make_inits(self, minibatch_size):
        return make_torch_zeros(minibatch_size, [self.num_units, self.num_units])


