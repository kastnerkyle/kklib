import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import argparse
import sys

from kklib import get_saved_model_defs
from kklib import get_saved_model_config

from kklib.datasets import fetch_ljspeech
from kklib.iterators import wavfile_caching_mel_tbptt_iterator

from scipy.io import wavfile
import scipy
from kklib.tools.audio import soundsc
from kklib.tools.audio import stft
from kklib.tools.audio import linear_to_mel_weight_matrix
from kklib.tools.audio import iterate_invert_spectrogram

import tensorflow as tf

use_cuda = torch.cuda.is_available()
if use_cuda:
    DEVICE = "cuda"
else:
    DEVICE = "cpu"
parser = argparse.ArgumentParser()
parser.add_argument('direct_model', nargs=1, default=None)
parser.add_argument('--model', dest='model_path', type=str, default=None)
args = parser.parse_args()
if args.model_path == None:
    if args.direct_model == None:
        raise ValueError("Must pass first positional argument as model, or --model argument, e.g. summary/experiment-0/models/model-7")
    else:
        saved_model_path = args.direct_model[0]
else:
    saved_model_path = args.model_path


random_state = np.random.RandomState(11)
ljspeech = fetch_ljspeech()

wavfiles = ljspeech["wavfiles"]
jsonfiles = ljspeech["jsonfiles"]

# THESE HAVE TO BE THE SAME TO ENSURE SPLIT IS CORRECT
train_random_state = np.random.RandomState(3122)
valid_random_state = np.random.RandomState(3122)

all_state_dicts = torch.load(saved_model_path, map_location=DEVICE)
models_file_path = get_saved_model_defs(saved_model_path)
sys.path.insert(0, models_file_path); import models; sys.path.remove(models_file_path)
config_file_path = get_saved_model_config(saved_model_path)
sys.path.insert(0, config_file_path); from config import *; sys.path.remove(config_file_path)

train_itr = wavfile_caching_mel_tbptt_iterator(wavfiles, jsonfiles, minibatch_size, truncation_length, masked=True, stop_index=.95, shuffle=True, symbol_processing="chars_only", random_state=train_random_state)
valid_itr = wavfile_caching_mel_tbptt_iterator(wavfiles, jsonfiles, minibatch_size, truncation_length, masked=True, start_index=.95, shuffle=True, symbol_processing="chars_only", random_state=valid_random_state)

vocabulary_size = train_itr.vocabulary_sizes[0]
n_mel = output_size = 80
sonify_steps = 100
gl_steps = 100
window_size = 512
step = 128
sample_rate = 22050

model = models.Model(minibatch_size, vocabulary_size, embed_size, n_stacks, n_filters, prenet_size, enc_hidden_size, dec_hidden_size, output_size, cell_dropout_keep_rate, random_state).to(DEVICE)
model.load_state_dict(all_state_dicts["Model"])
model = model.to(DEVICE)
# eval?
model = model.eval()

attn_h_i, attn_c_i, attn_k_i, attn_w_i, h1_i, c1_i, h2_i, c2_i = model.make_inits()
attn_h_i = Variable(attn_h_i).to(DEVICE)
attn_c_i = Variable(attn_c_i).to(DEVICE)
attn_k_i = Variable(attn_k_i).to(DEVICE)
attn_w_i = Variable(attn_w_i).to(DEVICE)
h1_i = Variable(h1_i).to(DEVICE)
c1_i = Variable(c1_i).to(DEVICE)
h2_i = Variable(h2_i).to(DEVICE)
c2_i = Variable(c2_i).to(DEVICE)

mels = np.zeros((1, minibatch_size, output_size))
mels_mask = 0. * mels[:, :, 0] + 1.

predicted_mels = [mels]
predicted_stops = [-1 for _ in range(minibatch_size)]

predicted_attn_k = []
predicted_attn_w = []
predicted_attn_phi = []

teacher_forced = False
if not teacher_forced:
    #chars = "today is the day"
    chars = "the Roman letter was used side by side with the Gothic"
    # add a period so that we know the ending
    chars = chars + "."
    inds, inds_mask = train_itr.transform_txt(chars)
else:
    #d = np.load("/Tmp/kastner/_data_cache/Tmp-kastner-lj_speech-LJSpeech-1.0-wavs/LJ001-0001-txt-cleanenglish_cleanersenglish_phone_cleaners-logmel-wsz512-wst128-leh125-ueh7800-nmel80.npz")
    #d = np.load("/Tmp/kastner/_data_cache/Tmp-kastner-lj_speech-LJSpeech-1.0-wavs/LJ002-0024-txt-cleanenglish_cleanersenglish_phone_cleaners-logmel-wsz512-wst128-leh125-ueh7800-nmel80.npz")
    d = np.load("/Tmp/kastner/_data_cache/Tmp-kastner-lj_speech-LJSpeech-1.0-wavs/LJ001-0061-txt-cleanenglish_cleanersenglish_phone_cleaners-logmel-wsz512-wst128-leh125-ueh7800-nmel80.npz")

    teacher_mels = d["log_mels"]
    teacher_mels = (teacher_mels - train_itr._mean) / train_itr._std
    teacher_mels = teacher_mels[:, None] * np.ones((1, minibatch_size, 1)).astype("int32")

    chars = str(d["clean_transcript"])
    if chars[-1] != ".":
        chars = chars + "."
    inds, inds_mask = train_itr.transform_txt(chars)
#inds = char_data[0]
#chars = "".join([ind_to_symbol[i] for i in inds])

# Only 2D, so embedding works
inds = np.array(inds)[:, None] * np.ones((1, minibatch_size)).astype("int32")
inds_mask = 0 * inds + 1

sample_len = 30 * len(chars)
sample_len = 1550
if teacher_forced:
    sample_len = len(teacher_mels) - 1
for i in range(sample_len):
    print("Sample step {}".format(i))
    enc_input_variable = Variable(torch.LongTensor(inds)).to(DEVICE)
    enc_input_mask = torch.LongTensor(inds_mask).to(DEVICE)

    dec_input_variable = Variable(torch.FloatTensor(predicted_mels[-1])).to(DEVICE)
    #dec_input_variable = Variable(torch.FloatTensor(teacher_force_coords[i][None])).to(DEVICE)

    dec_input_mask = Variable(torch.FloatTensor(mels_mask)).to(DEVICE)

    o = model(enc_input_variable, dec_input_variable,
              attn_h_i, attn_c_i, attn_k_i, attn_w_i,
              h1_i, c1_i, h2_i, c2_i,
              enc_input_mask=enc_input_mask, dec_input_mask=dec_input_mask)
    preds = o[0]
    attn_h = o[1]
    attn_c = o[2]
    attn_k = o[3]
    attn_w = o[4]
    attn_phi = o[5]
    h1 = o[6]
    c1 = o[7]
    h2 = o[8]
    c2 = o[9]

    # get attn information for plots
    attn_k_np = attn_k.detach().cpu().numpy()[-1]
    attn_w_np = attn_w.detach().cpu().numpy()[-1]
    attn_phi_np = attn_phi.detach().cpu().numpy()[-1]
    predicted_attn_k.append(attn_k_np)
    predicted_attn_w.append(attn_w_np)
    predicted_attn_phi.append(attn_phi_np)

    # get audio info
    preds = preds.detach().cpu().numpy()[-1]
    if teacher_forced:
        predicted_mels.append(teacher_mels[i][None])
    else:
        predicted_mels.append(preds[None])

    """
    # from tfbldr
    last_sym = int(text_mask[:, mbi].sum()) - 1
    if np.argmax(att_phi_np[0, mbi]) >= last_sym or np.argmax(att_phi_np[0, mbi]) == text_mask.shape[0]:
        if is_finished_sampling[mbi] == False:
            is_finished_sampling[mbi] = True
            finished_step[mbi] = ii
    """
    stop_scale = 7.5
    for choose_i in range(preds.shape[0]):
        thresh = attn_phi_np[choose_i, -1] > stop_scale * np.max(attn_phi_np[choose_i, :-1])
        if thresh and predicted_stops[choose_i] < 0:
            predicted_stops[choose_i] = i

    attn_h_i = attn_h[-1].detach()
    attn_c_i = attn_c[-1].detach()
    attn_k_i = attn_k[-1].detach()
    attn_w_i = attn_w[-1].detach()
    h1_i = h1[-1].detach()
    c1_i = c1[-1].detach()
    h2_i = h2[-1].detach()
    c2_i = c2[-1].detach()

    if all([p >= 0 for p in predicted_stops]):
        print("All samples ended, finishing...")
        break

# samples plot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
attn_phi = np.stack(predicted_attn_phi) # timesteps batch input_step_length
attn_k = np.stack(predicted_attn_k) # timesteps batch attn_components
predicted_mels = np.concatenate(predicted_mels, axis=0)

# attn plots
n_plots = 5
for n in range(n_plots):
    fig, ax = plt.subplots(1, 1)
    this_stop = predicted_stops[n]
    this_phi = attn_phi[:this_stop, n].T[::-1, :]
    plt.figure()
    plt.imshow(this_phi, interpolation='nearest', aspect='auto', cmap=cm.jet)
    plt.yticks(np.arange(0, len(chars[:-1]) + 1))
    plt.axes().set_yticklabels(list(' ' + chars[:-1][::-1]), rotation='vertical', fontsize=8)
    plt.grid(False)
    ax.set_aspect("equal")
    plt.savefig("plot_attention_basic_speech_synthesis{}.png".format(n))

# mel spectrogram plots?
def implot(arr, axarr, scale=None, title="", cmap=None):
    # plotting part
    #mag = 20. * np.log10(np.abs(arr))
    mag = arr
    # Transpose so time is X axis, and invert y axis so
    # frequency is low at bottom
    mag = mag.T#[::-1, :]

    if cmap != None:
        axarr.imshow(mag, cmap=cmap, origin="lower")
    else:
        axarr.imshow(mag, origin="lower")

    plt.axis("off")
    x1 = mag.shape[0]
    y1 = mag.shape[1]
    if scale == "specgram":
        y1 = int(y1 * .20)

    def autoaspect(x_range, y_range):
        """
        The aspect to make a plot square with ax.set_aspect in Matplotlib
        """
        mx = max(x_range, y_range)
        mn = min(x_range, y_range)
        if x_range <= y_range:
            return mx / float(mn)
        else:
            return mn / float(mx)
    asp = autoaspect(x1, y1)
    axarr.set_aspect(asp)
    plt.title(title)

def tflogmel(waveform):
    z = tf.contrib.signal.stft(waveform, window_size, step)
    magnitudes = tf.abs(z)
    filterbank = tf.contrib.signal.linear_to_mel_weight_matrix(
        num_mel_bins=n_mel,
        num_spectrogram_bins=magnitudes.shape[-1].value,
        sample_rate=sample_rate,
        lower_edge_hertz=125.,
        upper_edge_hertz=7800.)
    melspectrogram = tf.tensordot(magnitudes, filterbank, 1)
    return tf.log1p(melspectrogram)

def tfsonify(spectrogram, samples, transform_op_fn, logscaled=True):
    graph = tf.Graph()
    with graph.as_default():

        noise = tf.Variable(tf.random_normal([samples], stddev=1e-6))

        x = transform_op_fn(noise)
        y = spectrogram

        if logscaled:
            x = tf.expm1(x)
            y = tf.expm1(y)

        # tf.nn.normalize arguments changed between versions...
        def normalize(a):
            return a / tf.sqrt(tf.maximum(tf.reduce_sum(a ** 2, axis=0), 1E-12))

        x = normalize(x)
        y = normalize(y)
        tf.losses.mean_squared_error(x, y[-tf.shape(x)[0]:])

        optimizer = tf.contrib.opt.ScipyOptimizerInterface(
            loss=tf.losses.get_total_loss(),
            var_list=[noise],
            tol=1E-16,
            method='L-BFGS-B',
            options={
                'maxiter': sonify_steps,
                'disp': True
            })

    # THIS REALLY SHOULDN'T RUN ON GPU BUT SEEMS TO?
    config = tf.ConfigProto(
        device_count={'CPU' : 1, 'GPU' : 0},
        allow_soft_placement=True,
        log_device_placement=False
        )
    with tf.Session(config=config, graph=graph) as session:
        session.run(tf.global_variables_initializer())
        optimizer.minimize(session)
        waveform = session.run(noise)
    return waveform

def logmel(waveform):
    res = np.abs(stft(waveform, windowsize=window_size, step=step, real=False, compute_onesided=True))
    mels = linear_to_mel_weight_matrix(
        res.shape[1],
        sample_rate,
        lower_edge_hertz=125.,
        upper_edge_hertz=7800.,
        n_filts=n_mel, dtype=np.float64)
    mel_res = np.dot(res, mels)
    return np.log1p(mel_res)

def sonify(spectrogram, samples, transform_fn, logscaled=True):
    # this doesn't really work at all without grads
    noise = random_state.randn(samples) * 1E-6

    def cost(theta):
        x = transform_fn(theta)
        y = spectrogram
        if logscaled:
            x = np.expm1(x)
            y = np.expm1(y)
        def normalize(a):
            return a / np.sqrt(np.maximum(np.sum(a ** 2, axis=0), 1E-12))

        x = normalize(x)
        y = normalize(y)
        return np.mean((x - y[-x.shape[0]:]) ** 2)
    res = scipy.optimize.fmin_l_bfgs_b(cost, noise, approx_grad=True,
                                       #tol=1E7,
                                       factr=10,
                                       # factr * numpy.finfo(float).eps
                                       maxiter=sonify_step,
                                       iprint=1)
    return res[0]

for n in range(n_plots):
    spectrogram = predicted_mels[:, n] * train_itr._std + train_itr._mean
    spectrogram = spectrogram[:this_stop]
    this_stop = predicted_stops[n]
    title = "sampled {}: {}".format(n, chars[:-1])
    f, axarr = plt.subplots(1, 1)
    implot(spectrogram, axarr, scale="specgram", title=title)
    plt.savefig("plot_sample_basic_speech_synthesis{}.png".format(n))
    plt.close()
    reconstructed_waveform = tfsonify(spectrogram, len(spectrogram) * step, tflogmel)
    wavfile.write("sonify_basic_speech_synthesis{}.wav".format(n), sample_rate, soundsc(reconstructed_waveform))
    print("wrote plots and samples {}".format(n))

    """
    # impossibly slow without grad fn
    #reconstructed_waveform = sonify(spectrogram, len(spectrogram) * step, logmel)
    # do it overlap_add style, maybe it can work
    z = np.zeros((len(spectrogram) * step,))
    pos = 0
    for i in range(len(spectrogram)):
        # boundary conds might be killer
        reconstructed_waveform = sonify(spectrogram[i][None], window_size, logmel)
        piece = z[pos:min(pos + window_size, len(z))]
        win = np.hanning(len(piece))
        z[pos:min(pos + window_size, len(z))] += win * reconstructed_waveform[:len(piece)]
        pos += step
    wavfile.write("sonify_basic_speech_synthesis{}.wav".format(n), sample_rate, soundsc(z))
    print("wrote plots and samples {}".format(n))
    """
