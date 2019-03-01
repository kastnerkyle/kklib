# Author: Kyle Kastner
# License: BSD 3-Clause
import os
import copy
import numpy as np
from scipy.io import wavfile
import scipy.signal as sg
from scipy import linalg, fftpack
from numpy.lib.stride_tricks import as_strided

def _raised_cosine_window(window_length, periodic, a, b):
    even = 1 - window_length % 2
    periodic = 1. if True else False
    n = np.float64(window_length + periodic * even - 1)
    count = np.arange(window_length).astype(np.float64)
    cos_arg = 2 * np.pi * count / n
    return a - b * np.cos(cos_arg)


def soundsc(X, gain_scale=.9, copy=True):
    X = np.array(X, copy=copy)
    X = (X - X.min()) / (X.max() - X.min())
    X = 2 * X - 1
    X = gain_scale * X
    X = X * 2 ** 15
    return X.astype('int16')


def halfoverlap(X, window_size):
    if window_size % 2 != 0:
        raise ValueError("Window size must be even!")
    window_step = window_size // 2
    # Make sure there are an even number of windows before stridetricks
    append = np.zeros((window_size - len(X) % window_size))
    X = np.hstack((X, append))
    num_frames = len(X) // window_step - 1
    row_stride = X.itemsize * window_step
    col_stride = X.itemsize
    X_strided = as_strided(X, shape=(num_frames, window_size),
                           strides=(row_stride, col_stride))
    return X_strided


def overlap(X, window_size, window_step, window=None, copy=True):
    if not hasattr(X, "shape") or len(X.shape) != 1:
        raise ValueError("X must be passed as 1D np array")
    if copy:
        X = np.array(X)
        X = X.copy()
    if window_size % 2 != 0:
        raise ValueError("Window size must be even!")
    # Make sure there are an even number of windows before stridetricks
    # need to window in here?
    append = np.zeros((window_size - len(X) % window_size))
    X = np.hstack((X, append))
    overlap_sz = window_size - window_step
    new_shape = X.shape[:-1] + ((X.shape[-1] - overlap_sz) // window_step, window_size)
    new_strides = X.strides[:-1] + (window_step * X.strides[-1],) + X.strides[-1:]
    X_strided = as_strided(X, shape=new_shape, strides=new_strides)
    return X_strided


def stft(X, windowsize=None, fftsize=None, step="half", mean_normalize=True, real=False,
         window_type="hann", periodic=True, compute_onesided=True):
    if real:
        raise ValueError("real=True needs debug")
        local_fft = fftpack.rfft
        cut = None
    else:
        local_fft = fftpack.fft
        cut = None

    if fftsize == None:
        assert windowsize is not None
        enclosing_fftsize = int(2 ** np.ceil(np.log(windowsize) / np.log(2.0)))
        fftsize = enclosing_fftsize
    else:
        windowsize = fftsize

    if compute_onesided or real:
        cut = fftsize // 2 + 1

    if mean_normalize:
        X -= X.mean()

    if step == "half":
        X = halfoverlap(X, windowsize)
    else:
        X = overlap(X, windowsize, step)

    size = fftsize
    if window_type == "hann" and periodic:
        win = _raised_cosine_window(size, True, 0.5, 0.5)
    else:
        raise ValueError("No other windows currently supported")
        #win = 0.54 - .46 * np.cos(2 * np.pi * np.arange(size) / (size - 1))
    #win = 0.54 - .46 * np.cos(2 * np.pi * np.arange(size) / (size - 1))
    X = X * win[None]
    X = local_fft(X.astype(np.float64))[:, :cut]
    return X


def calc_offset(x1, x2):
    x1 = x1 - x1.mean()
    x2 = x2 - x2.mean()
    half = len(x2) // 2
    #xcorrs = sg.fftconvolve(x1.astype("float32"), x2[::-1].astype("float32")) / (np.linalg.norm(x1) * np.linalg.norm(x2))
    # on shorter sequences conv can be faster
    xcorrs = np.convolve(x1.astype("float32"), x2[::-1].astype("float32")) / (np.linalg.norm(x1) * np.linalg.norm(x2))

    # don't take from edges
    xcorrs[:half] = -1E30
    xcorrs[-half:] = -1E30
    offset = xcorrs.argmax() - len(x1)
    return offset

# matlab / octave hann divides by windowsize not windowsize - 1 as in numpy
def matlab_hann(windowsize):
    return np.array([.5 * (1 - np.cos((2 * np.pi * n) / (windowsize))) for n in range(windowsize)])


def make_windual(win, step):
    # http://splab.cz/wp-content/uploads/2013/11/Gabor-dual-windows-using-convex-optimization.pdf
    # calculate the canonical dual window
    extra = np.zeros((step * (int(len(win) // step) + 1) - len(win),))
    windual = np.concatenate([win, extra])
    # matlab vs numpy order
    windual = windual.reshape((-1, step)).T
    windual = windual / np.sum(np.abs(windual) ** 2, axis=1, keepdims=True)
    windual = windual.T.ravel()[:len(win)]
    return windual

def xcorr_istft(X_s, step, min_win_sum=1E-20):
    size = int(X_s.shape[1] // 2)
    wave = np.zeros((X_s.shape[0] * step + size))
    wave = wave.astype('float64')
    total_windowing_sum = np.zeros((X_s.shape[0] * step + size))
    win = make_windual(matlab_hann(size), step)
    #win = matlab_hann(size)

    est_start = int(size // 2) - 1
    est_end = est_start + size
    for i in range(X_s.shape[0]):
        wave_start = int(step * i)
        wave_end = wave_start + size
        spectral_slice = X_s[i]
        wave_est = np.fft.ifft(spectral_slice).real
        if i > 0:
            offset_size = size - step
            offset = calc_offset(wave[wave_start:wave_start + offset_size],
                               wave_est[est_start:est_start + offset_size])
        else:
            offset = 0
        wave[wave_start:wave_end] += win * wave_est[
            est_start - offset:est_end - offset]
        total_windowing_sum[wave_start:wave_end] += win
    wave = wave / (total_windowing_sum + min_win_sum)
    return wave

def linear(x):
    return x

def relu(x):
    return x * (x > 0)

def lrelu(x, alpha=.01):
    return x * (x > 0) + alpha * x * (x <= 0)

def softplus(x):
    beta = 1.
    bx = beta * x
    y = (np.fmax(bx, 0) +
         np.log1p(np.exp(-np.fabs(bx)))) * (1. / beta)
    return y

activation_map = {"linear": linear,
                  "relu": relu,
                  "lrelu": lrelu,
                  "softplus": softplus}

def power_invert_spectrogram_orig(X_s, fftsize, step, add_power=None, n_iter=10, activation_name="linear", verbose=False):
    X_best = copy.deepcopy(X_s)
    activation = activation_map[activation_name]
    if activation_name != "linear":
        print("This function is only for checking the original griffin lim against variants - use power_inver_spectrogram_gla instead!")
    if add_power != None:
        print("This function is only for checking the original griffin lim against variants - use power_inver_spectrogram_gla instead!")
    for i in range(n_iter):
        if verbose:
            print("orig|add_pwr:{}|fn:{} runnning iter {}".format(add_power, activation.func_name, i))
        X_t = xcorr_istft(X_best, step)
        # http://contents.acoust.ias.sci.waseda.ac.jp/publications/IWAENC/2018/IWAENC-yatabe-2018Sep.pdf
        X_t = activation(X_t)
        est = stft(X_t, fftsize=fftsize, step=step, compute_onesided=False)
        X_best = X_s * (est / np.abs(est))
    X_t = xcorr_istft(X_best, step)
    return np.real(X_t)

def power_invert_spectrogram_gla(X, fftsize, step, add_power=1., n_iter=10, activation_name="linear", verbose=False):
    A = copy.deepcopy(X)
    X = copy.deepcopy(X)

    activation = activation_map[activation_name]

    def pc2(X_, cmplx=False):
        if cmplx:
            return A ** (1. + add_power) * (X_ / np.abs(X_))
        else:
            #return A ** (1. + add_power) * np.sign(X_)
            # A and X_ are the same in the first iteration
            # which is where we set cmplx to False
            return A

    def pc1(X_):
        p1 = xcorr_istft(X_, step)
        # http://contents.acoust.ias.sci.waseda.ac.jp/publications/IWAENC/2018/IWAENC-yatabe-2018Sep.pdf
        p1 = activation(p1)
        p2 = stft(p1, fftsize=fftsize, step=step, compute_onesided=False)
        return p2

    for i in range(n_iter):
        if verbose:
            print("gla|add_pwr:{}|fn:{} runnning iter {}".format(add_power, activation.func_name, i))
        X = pc1(pc2(X, True if i > 0 else False))
    X_t = xcorr_istft(X, step)
    return np.real(X_t)

def power_invert_spectrogram_fgla(X, fftsize, step, alpha=.99, add_power=1., n_iter=10, activation_name="linear", verbose=False):
    A = copy.deepcopy(X)
    X = copy.deepcopy(X)

    activation = activation_map[activation_name]
    if activation_name != "linear":
        print("This function only performs well with linear activation, use activation {} at your own risk!".format(activation_name))

    def pc2(X_, cmplx=False):
        if cmplx:
            return A ** (1. + add_power) * (X_ / np.abs(X_))
        else:
            #return A ** (1. + add_power) * np.sign(X_)
            # A and X_ are the same in the first iteration
            # which is where we set cmplx to False
            return A

    def pc1(X_):
        p1 = xcorr_istft(X_, step)
        # http://contents.acoust.ias.sci.waseda.ac.jp/publications/IWAENC/2018/IWAENC-yatabe-2018Sep.pdf
        p1 = activation(p1)
        p2 = stft(p1, fftsize=fftsize, step=step, compute_onesided=False)
        return p2
    Y = X
    for i in range(n_iter):
        if verbose:
            print("fgla|alpha:{}|add_pwr:{}|fn:{} runnning iter {}".format(alpha, add_power, activation.func_name, i))
        Xold = X.copy()
        X = pc1(pc2(Y, True if i > 0 else False))
        Y = X + alpha * (X - Xold)
    X_t = xcorr_istft(X, step)
    return np.real(X_t)

def power_invert_spectrogram_admmgla(X, fftsize, step, rho=0., add_power=1., n_iter=10, activation_name="linear", verbose=False):
    # admm one doesn't work well at smaller step sizes
    A = copy.deepcopy(X)
    X = copy.deepcopy(X)

    activation = activation_map[activation_name]

    def pc2(X_, cmplx=False):
        if cmplx:
            return A ** (1. + add_power) * (X_ / np.abs(X_))
        else:
            #return A ** (1. + add_power) * np.sign(X_)
            # A and X_ are the same in the first iteration
            # which is where we set cmplx to False
            return A

    def pc1(X_):
        p1 = xcorr_istft(X_, step)
        p1 = activation(p1)
        p2 = stft(p1, fftsize=fftsize, step=step, compute_onesided=False)
        return p2

    Z = X.copy()
    U = 0. * X
    for i in range(n_iter):
        if verbose:
            print("admmgla|rho:{}|add_pwr:{}|fn:{} runnning iter {}".format(rho, add_power, activation.func_name, i))
        X = pc2(Z - U, True if i > 0 else False)
        Y = X + U
        Z = (rho * Y + pc1(Y)) / (1. + rho)
        U = U + X - Z
    X_t = xcorr_istft(X, step)
    return np.real(X_t)

if __name__ == "__main__":
    fs, d = wavfile.read("z_input_dirty.wav")
    d = d.astype("float32").copy()
    fftsize = 512
    step = 32
    gl_steps = 5

    rw_s = np.abs(stft(d, fftsize=fftsize, step=step, real=False, compute_onesided=False))

    #acts = ["linear", "softplus", "lrelu", "relu"]
    #pwrs = [.75, 1., 1.25]

    #acts = ["linear"]
    #pwrs = [0., .25, .5, .75, 1., 1.25, 1.5, 2.0]

    #acts = ["linear", "softplus"]
    #pwrs = [1., 1.5, 2.]

    acts = ["linear"]
    pwrs = [1.]

    for act in acts:
        for pwr in pwrs:
            rw = power_invert_spectrogram_gla(rw_s, fftsize, step, add_power=pwr, n_iter=gl_steps, activation_name=act, verbose=True)
            wavfile.write("output_gla_{}_pwr{}_iter{}.wav".format(act, pwr, gl_steps), fs, soundsc(rw))
            rw = power_invert_spectrogram_fgla(rw_s, fftsize, step, add_power=pwr, n_iter=gl_steps, activation_name=act, verbose=True)
            wavfile.write("output_fgla_{}_pwr{}_iter{}.wav".format(act, pwr, gl_steps), fs, soundsc(rw))
            rw = power_invert_spectrogram_admmgla(rw_s, fftsize, step, add_power=pwr, n_iter=gl_steps, activation_name=act, verbose=True)
            wavfile.write("output_admmgla_{}_pwr{}_iter{}.wav".format(act, pwr, gl_steps), fs, soundsc(rw))
            rw = power_invert_spectrogram_orig(rw_s, fftsize, step, add_power=pwr, n_iter=gl_steps, activation_name=act, verbose=True)
            wavfile.write("output_orig_{}_pwr{}_iter{}.wav".format(act, pwr, gl_steps), fs, soundsc(rw))
