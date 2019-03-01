from .audio_tools import stft
from .audio_tools import linear_to_mel_weight_matrix
from .audio_tools import soundsc
from .power_gl import power_invert_spectrogram_fgla

from ..autoptim import minimize
from functools import partial
import torch
import numpy as np

def th_mel_transform(window_size, window_step, n_mel, sample_rate, logscaled, waveform):
    # th is slightly worse than tf, either due to impl details of mel
    # (I matched tf as close as possible, however)
    # or due to implementation details in torch stft
    # 2 * window_size here because pytorch doesn't handle real values 
    # only in the "normal" way, instead does the 1/2 bins + 1 thing
    # default window in pytorch is rectangular? set it manually
    freq_r_i = torch.stft(waveform, 2 * window_size, hop_length=window_step,
                          window=torch.hann_window(2 * window_size).type(waveform.dtype))
    # skip DC component
    magnitudes = torch.sqrt(freq_r_i[1:, :, 0] ** 2 + freq_r_i[1:, :, 1] ** 2)
    magnitudes = magnitudes.transpose(1, 0)
    mels = linear_to_mel_weight_matrix(
        magnitudes.shape[1],
        sample_rate,
        lower_edge_hertz=125.,
        upper_edge_hertz=7800.,
        n_filts=n_mel, dtype=np.float64)
    # do it this way in case on cpu
    mels = torch.DoubleTensor(mels).type(waveform.dtype)
    mel_res = torch.tensordot(magnitudes, mels, 1)
    if logscaled:
        return torch.log1p(mel_res)
    else:
        return mel_res

def th_sonify(spectrogram, samples, transform_fn,
              sonify_steps=100, sonify_tol=1E-16, sonify_disp=True,
              logscaled=True, random_state=None):
    noise = random_state.randn(samples) * 1E-6

    def cost(theta):
        x = transform_fn(theta)
        y = spectrogram.type(x.dtype)
        if logscaled:
            x = torch.expm1(x)
            y = torch.expm1(y)

        def normalize(a):
            return a / torch.sqrt(torch.sum(a ** 2, dim=0) + 1E-12)

        x = normalize(x)
        y = normalize(y)
        min_shp = min(x.shape[0], y.shape[0])
        return torch.mean((x[:min_shp] - y[:min_shp]) ** 2)
    res, _ = minimize(cost, noise,
                      method='L-BFGS-B',
                      tol=sonify_tol,
                      options={"maxiter": sonify_steps,
                               "disp": sonify_disp}

                      )
    return res
"""
n_mel = output_size = 80
sonify_steps = 100
gl_steps = 100
window_size = 512
step = 128
sample_rate = 22050
"""

def mel2wav(spectrogram,
            window_size=512, window_step=128, n_mel=80, sample_rate=22050,
            sonify_steps=100,
            sonify_tol=1E-16,
            sonify_disp=True,
            gl_window_size=512,
            gl_step_size=32,
            gl_steps=10,
            gl_power=1.,
            logscaled=True, random_state=None, DEVICE="cpu"):
    if random_state is None:
        random_state = np.random.RandomState(1122)
    spectrogram = torch.DoubleTensor(spectrogram).to(DEVICE)
    this_func = partial(th_mel_transform, window_size, window_step, n_mel, sample_rate, logscaled)
    reconstructed_waveform = th_sonify(spectrogram, len(spectrogram) * window_step, this_func,
                                       sonify_steps=sonify_steps, sonify_tol=sonify_tol, sonify_disp=sonify_disp,
                                       logscaled=logscaled, random_state=random_state)
    fftsize = gl_window_size
    step = gl_step_size

    rw_s = np.abs(stft(reconstructed_waveform, fftsize=fftsize, step=step, real=False, compute_onesided=False))
    rw = power_invert_spectrogram_fgla(rw_s, fftsize, step, add_power=gl_power, n_iter=gl_steps, verbose=True)
    return reconstructed_waveform, rw
