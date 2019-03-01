import numpy as np

from .tools.audio import stft
from .tools.audio import linear_to_mel_weight_matrix
from .tools.audio import soundsc
from .tools.text import pronounce_chars
from .tools.text import text_to_sequence
from .tools.text import sequence_to_text
from .tools.text import cleaners
from .tools.text import get_vocabulary_sizes

from .core import get_logger
from .core import get_cache_dir

from scipy.io import wavfile
import numpy as np
import copy
import os
import json

logger = get_logger()

class masked_list_multisequence_iterator(object):
    """
       assumes it is elements, sequence, features ordered

       for an example of a list of numbers, and its reverse
       [[[1, 2, 3], [4, 5, 6]], [[3, 2, 1], [6, 5, 4]]]

       batches returned as two lists of elements, (sequence, samples), one for data, one for masks

       shuffles every reset if random state is provided
    """
    def __init__(self, list_of_data, minibatch_size, random_state=None, pad_with=0, dtype="float32", mask_dtype="float32"):
        self.data = list_of_data
        l = len(list_of_data[0])
        for i in range(len(list_of_data)):
            if len(list_of_data[i]) != l:
                raise ValueError("All list inputs in list_of_data must be the same length!")
        self.dtype = dtype
        self.mask_dtype = mask_dtype
        self.minibatch_size = minibatch_size
        self.pad_with = pad_with
        self.random_state = random_state
        self.start_idx_ = 0
        self.n_elements_ = len(list_of_data)

    def next(self):
        if self.start_idx_ >= (len(self.data[0]) - self.minibatch_size):
            self.reset()
        start_ = self.start_idx_
        end_ = self.start_idx_ + self.minibatch_size
        subs = [d[start_:end_] for d in self.data]
        maxlens = [max([len(su_i) for su_i in su]) for su in subs]
        r = []
        mask_r = []
        for n, su in enumerate(subs):
            r_i = []
            mask_r_i = []
            for su_i in su:
                if len(su_i) != maxlens[n]:
                    pad_part = [self.pad_with for _ in range(maxlens[n] - len(su_i))]
                    pad_su_i = su_i + pad_part
                    mask_su_i = [1. for _ in su_i] + [0. for _ in pad_part]
                    r_i.append(pad_su_i)
                    mask_r_i.append(mask_su_i)
                else:
                    r_i.append(su_i)
                    mask_r_i.append([1. for _ in su_i])
            np_r_i = np.array(r_i).astype(self.dtype).transpose(1, 0)
            np_mask_r_i = np.array(mask_r_i).astype(self.mask_dtype).transpose(1, 0)
            r.append(np_r_i)
            mask_r.append(np_mask_r_i)
        self.start_idx_ = end_
        return r, mask_r

    def  __next__(self):
        return self.next()

    def reset(self):
        self.start_idx_ = 0
        if self.random_state is not None:
            self.random_state.shuffle(self.data)


def make_mask(arr):
    mask = np.ones_like(arr[:, :, 0])
    last_step = arr.shape[0] * arr[0, :, 0]
    for mbi in range(arr.shape[1]):
        for step in range(arr.shape[0]):
            if arr[step:, mbi].min() == 0. and arr[step:, mbi].max() == 0.:
                last_step[mbi] = step
                mask[step:, mbi] = 0.
                break
    return mask


class tbptt_list_iterator(object):
    def __init__(self, tbptt_seqs, list_of_other_seqs, batch_size,
                 truncation_length,
                 tbptt_one_hot_size=None, other_one_hot_size=None,
                 masked=True,
                 random_state=None):
        """
        skips sequences shorter than truncation_len
        also cuts the tail off

        tbptt_one_hot_size
        should be either None, or the one hot size desired

        other_one_hot_size
        should either be None (if not doing one-hot) or a list the same length
        as the respective argument with integer one hot size, or None
        for no one_hot transformation, example:

        list_of_other_seqs = [my_char_data, my_vector_data]
        other_one_hot_size = [127, None]
        """
        self.tbptt_seqs = tbptt_seqs
        self.list_of_other_seqs = list_of_other_seqs
        self.batch_size = batch_size
        self.truncation_length = truncation_length
        self.masked = masked

        self.random_state = random_state
        if random_state is None:
            raise ValueError("Must pass random state for random selection")

        self.tbptt_one_hot_size = tbptt_one_hot_size

        self.other_one_hot_size = other_one_hot_size
        if other_one_hot_size is not None:
            assert len(other_one_hot_size) == len(list_of_other_seqs)

        tbptt_seqs_length = [n for n, i in enumerate(tbptt_seqs)][-1] + 1
        self.indices_lookup_ = {}
        s = 0
        for n, ts in enumerate(tbptt_seqs):
            if len(ts) >= truncation_length + 1:
                self.indices_lookup_[s] = n
                s += 1

        # this one has things removed
        self.tbptt_seqs_length_ = len(self.indices_lookup_)

        other_seqs_lengths = []
        for other_seqs in list_of_other_seqs:
            r = [n for n, i in enumerate(other_seqs)]
            l = r[-1] + 1
            other_seqs_lengths.append(l)
        self.other_seqs_lengths_ = other_seqs_lengths

        other_seqs_max_lengths = []
        for other_seqs in list_of_other_seqs:
            max_l = -1
            for os in other_seqs:
                max_l = len(os) if len(os) > max_l else max_l
            other_seqs_max_lengths.append(max_l)
        self.other_seqs_max_lengths_ = other_seqs_max_lengths

        # make sure all sequences have the minimum number of elements
        base = self.tbptt_seqs_length_
        for sl in self.other_seqs_lengths_:
            assert sl >= base

        # set up the matrices to slice one_hot indexes out of
        # todo: setup slice functions? or just keep handling in next_batch
        if tbptt_one_hot_size is None:
            self._tbptt_oh_slicer = None
        else:
            self._tbptt_oh_slicer = np.eye(tbptt_one_hot_size)

        if other_one_hot_size is None:
            self._other_oh_slicers = [None] * len(other_seqs_lengths)
        else:
            self._other_oh_slicers = []
            for ooh in other_one_hot_size:
                if ooh is None:
                    self._other_oh_slicers.append(None)
                else:
                    self._other_oh_slicers.append(np.eye(ooh, dtype=np.float32))
        # set up the indices selected for the first batch
        self.indices_ = np.array([self.indices_lookup_[si]
                                  for si in self.random_state.choice(self.tbptt_seqs_length_,
                                      size=(batch_size,), replace=False)])
        # set up the batch offset indicators for tracking where we are
        self.batches_ = np.zeros((batch_size,), dtype=np.int32)

    def next(self):
        if self.masked:
            return self.next_masked_batch()
        else:
            return self.next_batch()

    def __next__(self):
        return self.next()

    def next_batch(self):
        # whether the result is "fresh" or continuation
        reset_states = np.ones((self.batch_size, 1), dtype=np.float32)
        for i in range(self.batch_size):
            # cuts off the end of every long sequence! tricky logic
            if self.batches_[i] + self.truncation_length + 1 > self.tbptt_seqs[self.indices_[i]].shape[0]:
                ni = self.indices_lookup_[self.random_state.randint(0, self.tbptt_seqs_length_ - 1)]
                self.indices_[i] = ni
                self.batches_[i] = 0
                reset_states[i] = 0.

        # could slice before one hot to be slightly more efficient but eh
        items = [self.tbptt_seqs[ii] for ii in self.indices_]
        if self._tbptt_oh_slicer is None:
            truncation_items = items
        else:
            truncation_items = [self._tbptt_oh_slicer[ai] for ai in items]

        other_items = []
        for oi in range(len(self.list_of_other_seqs)):
            items = [self.list_of_other_seqs[oi][ii] for ii in self.indices_]
            if self._other_oh_slicers[oi] is None:
                # needs to be at least 2D
                other_items.append([it[:, None] if len(it.shape) == 1 else it for it in items])
            else:
                other_items.append([self._other_oh_slicers[oi][ai] for ai in items])

        # make storage
        tbptt_arr = np.zeros((self.truncation_length + 1, self.batch_size, truncation_items[0].shape[-1]), dtype=np.float32)
        other_arrs = [np.zeros((self.other_seqs_max_lengths_[ni], self.batch_size, other_arr[0].shape[-1]), dtype=np.float32)
                      for ni, other_arr in enumerate(other_items)]
        for i in range(self.batch_size):
            ns = truncation_items[i][self.batches_[i]:self.batches_[i] + self.truncation_length + 1]
            # dropped sequences shorter than truncation_len already
            tbptt_arr[:, i, :] = ns
            for na, oa in enumerate(other_arrs):
                oa[:len(other_items[na][i]), i, :] = other_items[na][i]
            self.batches_[i] += self.truncation_length
        return [tbptt_arr,] + other_arrs + [reset_states,]

    def next_masked_batch(self):
        r = self.next_batch()
        # reset is the last element
        end_result = []
        for ri in r[:-1]:
            ri_mask = make_mask(ri)
            end_result.append(ri)
            end_result.append(ri_mask)
        end_result.append(r[-1])
        return end_result
        

# As originally seen in sklearn.utils.extmath
# Credit to the sklearn team
def _incremental_mean_and_var(X, last_mean=.0, last_variance=None,
                              last_sample_count=0):
    """Calculate mean update and a Youngs and Cramer variance update.
    last_mean and last_variance are statistics computed at the last step by the
    function. Both must be initialized to 0.0. In case no scaling is required
    last_variance can be None. The mean is always required and returned because
    necessary for the calculation of the variance. last_n_samples_seen is the
    number of samples encountered until now.
    From the paper "Algorithms for computing the sample variance: analysis and
    recommendations", by Chan, Golub, and LeVeque.
    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Data to use for variance update
    last_mean : array-like, shape: (n_features,)
    last_variance : array-like, shape: (n_features,)
    last_sample_count : int
    Returns
    -------
    updated_mean : array, shape (n_features,)
    updated_variance : array, shape (n_features,)
        If None, only mean is computed
    updated_sample_count : int
    References
    ----------
    T. Chan, G. Golub, R. LeVeque. Algorithms for computing the sample
        variance: recommendations, The American Statistician, Vol. 37, No. 3,
        pp. 242-247
    Also, see the sparse implementation of this in
    `utils.sparsefuncs.incr_mean_variance_axis` and
    `utils.sparsefuncs_fast.incr_mean_variance_axis0`
    """
    # old = stats until now
    # new = the current increment
    # updated = the aggregated stats
    last_sum = last_mean * last_sample_count
    new_sum = X.sum(axis=0)

    new_sample_count = X.shape[0]
    updated_sample_count = last_sample_count + new_sample_count

    updated_mean = (last_sum + new_sum) / updated_sample_count

    if last_variance is None:
        updated_variance = None
    else:
        new_unnormalized_variance = X.var(axis=0) * new_sample_count
        if last_sample_count == 0:  # Avoid division by 0
            updated_unnormalized_variance = new_unnormalized_variance
        else:
            last_over_new_count = last_sample_count / new_sample_count
            last_unnormalized_variance = last_variance * last_sample_count
            updated_unnormalized_variance = (
                last_unnormalized_variance +
                new_unnormalized_variance +
                last_over_new_count / updated_sample_count *
                (last_sum / last_over_new_count - new_sum) ** 2)
        updated_variance = updated_unnormalized_variance / updated_sample_count
    return updated_mean, updated_variance, updated_sample_count


class wavfile_caching_mel_tbptt_iterator(object):
    def __init__(self, wavfile_list, txtfile_list,
                 batch_size,
                 truncation_length,
                 masked=True,
                 audio_processing="default",
                 symbol_processing="blended",
                 wav_scale = 2 ** 15,
                 window_size=512,
                 window_step=128,
                 n_mel_filters=80,
                 return_normalized=True,
                 lower_edge_hertz=125.0,
                 upper_edge_hertz=7800.0,
                 start_index=0,
                 stop_index=None,
                 cache_dir_base="default",
                 shuffle=False, random_state=None):
         if self.cache_dir_base == "default":
             self.cache_dir_base = get_cache_dir()
         self.masked = masked
         self.wavfile_list = wavfile_list
         self.wav_scale = wav_scale
         self.txtfile_list = txtfile_list
         self.batch_size = batch_size
         self.truncation_length = truncation_length
         self.random_state = random_state
         self.shuffle = shuffle
         self.cache_dir_base = cache_dir_base
         self.return_normalized = return_normalized
         self.lower_edge_hertz = lower_edge_hertz
         self.upper_edge_hertz = upper_edge_hertz

         self.audio_processing = audio_processing
         self.symbol_processing = symbol_processing
         symbol_opts = ["blended_pref", "blended", "chars_only", "phones_only", "both"]
         if symbol_processing not in symbol_opts:
             raise ValueError("symbol_processing set to invalid argument {}, should be one of {}".format(symbol_processing, symbol_opts))

         if audio_processing != "default":
             raise ValueError("Non-default settings not supported yet")
         clean_names = ["english_cleaners", "english_phone_cleaners"]
         self.clean_names = clean_names
         self.vocabulary_sizes = get_vocabulary_sizes(clean_names)
         self._special_chars = "!,:?"
         self.window_size = window_size
         self.window_step = window_step
         self.n_mel_filters = n_mel_filters
         self.start_index = start_index
         self.stop_index = stop_index

         if shuffle and self.random_state == None:
             raise ValueError("Must pass random_state in")
         if txtfile_list is not None:
             # try to match every txt file and every wav file by name
             wv_names_and_bases = sorted([(wv.split(os.sep)[-1], str(os.sep).join(wv.split(os.sep)[:-1])) for wv in self.wavfile_list])
             tx_names_and_bases = sorted([(tx.split(os.sep)[-1], str(os.sep).join(tx.split(os.sep)[:-1])) for tx in self.txtfile_list])
             wv_i = 0
             tx_i = 0
             wv_match = []
             tx_match = []
             wv_lu = {}
             tx_lu = {}
             for txnb in tx_names_and_bases:
                 if "." in txnb[0]:
                     tx_part = ".".join(txnb[0].split(".")[:1])
                 else:
                     # support txt files with no ext
                     tx_part = txnb[0]
                 tx_lu[tx_part] = txnb[1] + os.sep + txnb[0]

             for wvnb in wv_names_and_bases:
                 wv_part = ".".join(wvnb[0].split(".")[:1])
                 wv_lu[wv_part] = wvnb[1] + os.sep + wvnb[0]

             # set of in common keys
             shared_k = sorted([k for k in wv_lu.keys() if k in tx_lu])

             if self.symbol_processing == "blended_pref":
                 # no pruning needed for preferential blending
                 pass
             elif self.symbol_processing == "blended":
                 # no pruning needed for blending
                 pass
             elif self.symbol_processing == "chars_only":
                 # all txt files will have chars
                 pass
             elif self.symbol_processing in ["phones_only", "both"]:
                 # not all files will have valid phones, need to prune the set of files up front to avoid complex issues later
                 print("Pruning files to only phone results...")
                 to_remove = []
                 for n, sk in enumerate(shared_k):
                     txtpath = tx_lu[sk]
                     if not txtpath.endswith(".json"):
                         raise ValueError("Expected .json file, path given was {}".format(txtpath))
                     with open(txtpath, "rb") as f:
                         tj = json.load(f)
                     no_phones = [False if "phones" in word else True for word in tj["words"]]
                     if any(no_phones):
                         to_remove.append(sk)
                     if n % 1000 == 0:
                         print("File {} of {} inspected".format(n + 1, len(shared_k)))
                 for tr in to_remove:
                     del wv_lu[tr]
                     if tr in tx_lu:
                         del tx_lu[tr]
                 shared_k = sorted([k for k in wv_lu.keys() if k in tx_lu])
             else:
                 raise ValueError("Unknown value for self.symbol_processing {}".format(self.symbol_processing))

             for k in shared_k:
                 wv_match.append(wv_lu[k])
                 tx_match.append(tx_lu[k])
             self.wavfile_list = wv_match
             self.txtfile_list = tx_match
         self.cache = self.cache_dir_base + os.sep + "-".join(self.wavfile_list[0].split(os.sep)[1:-1])
         if not os.path.exists(self.cache):
             os.makedirs(self.cache)

         if 0 < self.start_index < 1:
             self.start_index = int(len(self.wavfile_list) * self.start_index)
         elif self.start_index >= 1:
             self.start_index = int(self.start_index)
             if self.start_index >= len(self.wavfile_list):
                 raise ValueError("start_index {} >= length of wavfile list {}".format(self.start_index, len(self.wavfile_list)))
         elif self.start_index == 0:
             self.start_index = int(self.start_index)
         else:
             raise ValueError("Invalid value for start_index : {}".format(self.start_index))

         if self.stop_index == None:
             self.stop_index = len(self.wavfile_list)
         elif 0 < self.stop_index < 1:
             self.stop_index = int(len(self.wavfile_list) * self.stop_index)
         elif self.stop_index >= 1:
             self.stop_index = int(self.stop_index)
             if self.stop_index >= len(self.wavfile_list):
                 raise ValueError("stop_index {} >= length of wavfile list {}".format(self.stop_index, len(self.wavfile_list)))
         else:
             raise ValueError("Invalid value for stop_index : {}".format(self.stop_index))

         # could match sizes here...
         self.wavfile_sizes_mbytes = [os.stat(wf).st_size // 1024 for wf in self.wavfile_list]

         if return_normalized:
             self.return_normalized = False

             # reset random seed here
             cur_random = self.random_state.get_state()

             # set up for train / test splits
             self.all_indices_ = np.arange(len(self.wavfile_list))
             self.random_state.shuffle(self.all_indices_)
             self.all_indices_ = sorted(self.all_indices_[self.start_index:self.stop_index])

             self.current_indices_ = [self.random_state.choice(self.all_indices_) for i in range(self.batch_size)]
             self.current_offset_ = [0] * self.batch_size
             self.current_read_ = [self.cache_read_wav_and_txt_features(self.wavfile_list[i], self.txtfile_list[i]) for i in self.current_indices_]
             self.to_reset_ = [0] * self.batch_size

             mean, std = self.cache_calculate_mean_and_std_normalization()
             self._mean = mean
             self._std = std

             self.random_state = np.random.RandomState()
             self.random_state.set_state(cur_random)
             self.return_normalized = True

         # set up for train / test splits
         self.all_indices_ = np.arange(len(self.wavfile_list))
         self.random_state.shuffle(self.all_indices_)
         self.all_indices_ = sorted(self.all_indices_[self.start_index:self.stop_index])

         self.current_indices_ = [self.random_state.choice(self.all_indices_) for i in range(self.batch_size)]
         self.current_offset_ = [0] * self.batch_size
         self.current_read_ = [self.cache_read_wav_and_txt_features(self.wavfile_list[i], self.txtfile_list[i]) for i in self.current_indices_]
         self.to_reset_ = [0] * self.batch_size

    def __next__(self):
        return self.next()

    def next(self):
        if self.masked:
            return self.next_masked_batch()
        else:
            return self.next_batch()

    def next_batch(self):
        mel_batch = np.zeros((self.truncation_length, self.batch_size, self.n_mel_filters))
        resets = np.ones((self.batch_size, 1))
        texts = []
        masks = []
        for bi in range(self.batch_size):
            wf, txf, mf  = self.current_read_[bi]
            if self.to_reset_[bi] == 1:
                self.to_reset_[bi] = 0
                resets[bi] = 0.
                # get a new sample
                while True:
                    self.current_indices_[bi] = self.random_state.choice(self.all_indices_)
                    self.current_offset_[bi] = 0
                    try:
                        self.current_read_[bi] = self.cache_read_wav_and_txt_features(self.wavfile_list[self.current_indices_[bi]], self.txtfile_list[self.current_indices_[bi]])
                    except:

                        logger.info("FILE / TEXT READ ERROR {}:{}".format(self.wavfile_list[self.current_indices_[bi]], self.txtfile_list[self.current_indices_[bi]]))
                        try:
                            self.current_read_[bi] = self.cache_read_wav_and_txt_features(self.wavfile_list[self.current_indices_[bi]], self.txtfile_list[self.current_indices_[bi]], force_refresh=True)
                            logger.info("CORRECTED FILE / TEXT READ ERROR VIA CACHE REFRESH")
                        except:
                            logger.info("STILL FILE / TEXT READ ERROR AFTER REFRESH {}:{}".format(self.wavfile_list[self.current_indices_[bi]], self.txtfile_list[self.current_indices_[bi]]))
                            continue
                    wf, txf, mf = self.current_read_[bi]
                    if len(wf) > self.truncation_length:
                        break

            trunc = self.current_offset_[bi] + self.truncation_length
            if trunc >= len(wf):
                self.to_reset_[bi] = 1
            wf_sub = wf[self.current_offset_[bi]:trunc]
            self.current_offset_[bi] = trunc
            mel_batch[:len(wf_sub), bi] = wf_sub
            texts.append(txf)
            masks.append(mf)

        if self.symbol_processing == "both":
            tlen = max([len(t) for t in texts])
            tlen2 = max([len(t) for t in texts])
            text_batch = np.zeros((tlen, self.batch_size, 1))
            text_batch2 = np.zeros((tlen2, self.batch_size, 1))
            text_lengths = []
            text_lengths2 = []
            for bi in range(len(texts)):
                txt = texts[bi]
                # masks are overloaded to be phones / other text repr
                txt2 = masks[bi]
                text_lengths.append(len(txt))
                text_lengths2.append(len(txt2))
                text_batch[:len(txt), bi, 0] = txt
                text_batch2[:len(txt2), bi, 0] = txt2
            return mel_batch, text_batch, text_batch2, text_lengths, text_lengths2, resets
        else:
            mlen = max([len(t) for t in texts])
            text_batch = np.zeros((mlen, self.batch_size, 1))
            type_mask_batch = np.zeros((mlen, self.batch_size, 1))
            text_lengths = []
            for bi in range(len(texts)):
                txt = texts[bi]
                mask = masks[bi]
                text_lengths.append(len(txt))
                text_batch[:len(txt), bi, 0] = txt
                type_mask_batch[:len(mask), bi, 0] = mask
            return mel_batch, text_batch, type_mask_batch, text_lengths, resets

    def next_masked_batch(self):
        if self.symbol_processing == "both":
            m, t, t2, tl, tl2, r = self.next_batch()
            m_mask = np.ones_like(m[..., 0])
            # not ideal, in theory could also hit on 0 mels but we aren't using this for now
            # should find contiguous chunk starting from the end
            m_mask[np.sum(m, axis=-1) == 0] = 0.

            t_mask = np.zeros_like(t[..., 0])
            t2_mask = np.zeros_like(t2[..., 0])
            # was [:tli], making mask of all 1s...
            for mbi, tli in enumerate(tl):
                t_mask[:tli, mbi] = 1.
            for mbi, tli in enumerate(tl2):
                t2_mask[:tli, mbi] = 1.
            return m, m_mask, t, t_mask, t2, t2_mask, r
        else:
            m, t, ma, tl, r = self.next_batch()
            m_mask = np.ones_like(m[..., 0])
            # not ideal, in theory could also hit on 0 mels but we aren't using this for now
            # should find contiguous chunk starting from the end
            m_mask[np.sum(m, axis=-1) == 0] = 0.

            t_mask = np.zeros_like(t[..., 0])
            ma_mask = np.zeros_like(ma[..., 0])
            # was [:tli], making mask of all 1s...
            for mbi, tli in enumerate(tl):
                t_mask[:tli, mbi] = 1.
                ma_mask[:tli, mbi] = 1.
            return m, m_mask, t, t_mask, ma, ma_mask, r

    def cache_calculate_mean_and_std_normalization(self, n_estimate=1000):
        normpath = self._fpathmaker("norm-mean-std")
        if not os.path.exists(normpath):
            logger.info("Calculating normalization per-dim mean and std")
            for i in range(n_estimate):
                if (i % 10) == 0:
                    logger.info("Normalization batch {} of {}".format(i, n_estimate))
                m, m_mask, t, t_mask, ma, ma_mask, r = self.next_masked_batch()
                m = m[m_mask > 0]
                m = m.reshape(-1, m.shape[-1])
                if i == 0:
                    normalization_mean = np.mean(m, axis=0)
                    normalization_std = np.std(m, axis=0)
                    normalization_count = len(m)
                else:
                    nmean, nstd, ncount = _incremental_mean_and_var(
                        m, normalization_mean, normalization_std,
                        normalization_count)

                    normalization_mean = nmean
                    normalization_std = nstd
                    normalization_count = ncount
            d = {}
            d["mean"] = normalization_mean
            d["std"] = normalization_std
            d["count"] = normalization_count
            np.savez(normpath, **d)
        norms = np.load(normpath)
        mean = norms["mean"]
        std = norms["std"]
        norms.close()
        return mean, std

    def calculate_log_mel_features(self, sample_rate, waveform, window_size, window_step, lower_edge_hertz, upper_edge_hertz, n_mel_filters):
        res = np.abs(stft(waveform, windowsize=window_size, step=window_step, real=False, compute_onesided=True))
        mels = linear_to_mel_weight_matrix(
            res.shape[1],
            sample_rate,
            lower_edge_hertz=lower_edge_hertz,
            upper_edge_hertz=min(float(sample_rate) // 2, upper_edge_hertz),
            n_filts=n_mel_filters, dtype=np.float64)
        mel_res = np.dot(res, mels)
        log_mel_res = np.log1p(mel_res)
        return log_mel_res

    def _fpathmaker(self, fname):
        melpart = "-logmel-wsz{}-wst{}-leh{}-ueh{}-nmel{}.npz".format(self.window_size, self.window_step, int(self.lower_edge_hertz), int(self.upper_edge_hertz), self.n_mel_filters)
        if self.txtfile_list is not None:
            txtpart = "-txt-clean{}".format(str("".join(self.clean_names)))
            npzpath = self.cache + os.sep + fname + txtpart + melpart
        else:
            npzpath = self.cache + os.sep + fname + melpart
        return npzpath

    def cache_read_wav_and_txt_features(self, wavpath, txtpath, force_refresh=False):
        wavfeats, npzfile, npzpath = self.cache_read_wav_features(wavpath, return_npz=True, force_refresh=force_refresh)
        txtfeats, txtmask = self.cache_read_txt_features(txtpath, npzfile=npzfile, npzpath=npzpath, force_refresh=force_refresh)
        npzfile.close()
        return wavfeats, txtfeats, txtmask

    def cache_read_wav_features(self, wavpath, return_npz=False, force_refresh=False):
        fname = ".".join(wavpath.split(os.sep)[-1].split(".")[:-1])
        npzpath = self._fpathmaker(fname)
        if force_refresh or not os.path.exists(npzpath):
            sr, d = wavfile.read(wavpath)
            d = d.astype("float64")
            d = d / float(self.wav_scale)
            log_mels = self.calculate_log_mel_features(sr, d, self.window_size, self.window_step,
                                                       self.lower_edge_hertz, self.upper_edge_hertz, self.n_mel_filters)
            np.savez(npzpath, wavpath=wavpath, sample_rate=sr, log_mels=log_mels)
        npzfile = np.load(npzpath)
        log_mels = npzfile["log_mels"]
        if self.return_normalized is True:
            log_mels = (log_mels - self._mean) / self._std
        if return_npz:
            return log_mels, npzfile, npzpath
        else:
            return log_mels

    def cache_read_txt_features(self, txtpath, npzfile=None, npzpath=None, force_refresh=False):
        if npzfile is None or "word_list" not in npzfile:
            if not txtpath.endswith(".json"):
                raise ValueError("Expected .json file, path given was {}".format(txtpath))
            with open(txtpath, "rb") as f:
                tj = json.load(f)
            # loaded json, now we need info
            char_txt = tj["transcript"]
            char_txt = char_txt.replace(u"\u2018", "'").replace(u"\u2019", "'")
            char_txt = char_txt.replace("-", " ")
            char_txt = char_txt.encode("ascii", "replace")
            try:
                clean_char_txt = cleaners.english_cleaners(char_txt)
            except Exception as e:
                print("unicode devil in cache read txt features")
                print(e)
                from IPython import embed; embed(); raise ValueError()
            clean_char_txt_split = clean_char_txt.split(" ")

            # need to get all the words and their paired phones, but also re-inject punctuations not found after cleaning... oy
            # triplets of transcript word, aligned word, and tuple of phones
            amalgam = []
            int_clean_char_chunks = []
            int_clean_phone_chunks = []
            # offset to handle edge case with "uh/ah" recognition
            offset = 0
            for i in range(len(tj["words"])):
                if i + offset >= len(clean_char_txt_split):
                    # edge case for 'uh' at the end of sentence
                    break
                this_word = tj["words"][i]
                this_base = this_word["word"]
                if this_word["case"] == "not-found-in-transcript":
                    # we skip this...
                    offset -= 1
                    continue

                if "alignedWord" in this_word:
                    this_align = this_word["alignedWord"]
                elif this_word["case"] == "not-found-in-transcript":
                    this_align = this_base
                elif this_word["case"] == "not-found-in-audio":
                    # if its not in the audio skip it
                    continue
                else:
                    print("new case in cache read txt features")
                    from IPython import embed; embed(); raise ValueError()

                try:
                    this_join_chars = str(clean_char_txt_split[i + offset])
                except:
                    print("another except in cache read txt features")
                    from IPython import embed; embed(); raise ValueError()

                int_clean_char_chunks.append(text_to_sequence(this_join_chars, [self.clean_names[0]])[:-1])

                if "phones" in this_word:
                    this_phones = this_word["phones"]
                    hack_phones = [tp.split("_")[0] for tp in [_["phone"] for _ in this_phones]]
                    # add leading @
                    this_join_phones = "@" + "@".join(hack_phones)
                    specials = "!?.,;:"
                    if this_join_chars[-1] in specials:
                        this_join_phones += this_join_chars[-1]
                    int_clean_phone_chunks.append(text_to_sequence(this_join_phones, [self.clean_names[1]])[:-2])
                else:
                    this_join_phones = [None]
                    this_phones = [None]
                    int_clean_phone_chunks.append([None])
                amalgam.append((this_base, this_align, this_join_chars, this_join_phones, this_phones))

                # check inversion is OK
                #print(sequence_to_text(int_clean_char_chunks[i], [self.clean_names[0]]))
                #print(sequence_to_text(int_clean_phone_chunks[i], [self.clean_names[1]]))

            #aa = [sequence_to_text(int_clean_char_chunks[i], [self.clean_names[0]]) for i in range(len(int_clean_char_chunks))]
            #cc = [sequence_to_text(int_clean_phone_chunks[i], [self.clean_names[1]]) for i in range(len(int_clean_phone_chunks))]
            #bb = [a[2] for a in amalgam]
            #dd = [a[3] for a in amalgam]
            # check inversion is OK
            #assert(aa == bb)
            #assert(cc == dd)
            word_list_invert = [sequence_to_text(int_clean_char_chunks[i], [self.clean_names[0]]) for i in range(len(int_clean_char_chunks))]
            phone_list_invert = [sequence_to_text(int_clean_phone_chunks[i], [self.clean_names[1]]) for i in range(len(int_clean_phone_chunks))]
            word_list = [a[2] for a in amalgam]
            phone_list = [a[3] for a in amalgam]

            # TODO: put em all in the npz, then figure out how / what to do on load...
            if force_refresh or (npzfile is not None and "word_list" not in npzfile):
                d = {k: v for k, v in npzfile.items()}
                npzfile.close()
                d["transcript"] = char_txt
                d["clean_transcript"] = clean_char_txt
                d["word_list"] = word_list
                d["word_list_invert"] = word_list_invert
                d["phone_list"] = phone_list
                d["phone_list_invert"] = phone_list_invert
                d["int_phone_chunks"] = int_clean_phone_chunks
                d["int_char_chunks"] = int_clean_char_chunks
                d["cleaners"] = "+".join(self.clean_names)
                np.savez(npzpath, **d)
        npzfile = np.load(npzpath)
        int_char_chunks = [list(c) for c in npzfile["int_char_chunks"]]
        int_phone_chunks = [list(p) for p in npzfile["int_phone_chunks"]]
        if len(int_char_chunks) != len(int_phone_chunks):
            # will need to handle edge case of no valid phones here...
            print("handle the char / phone different length edge case here cache read txt features")
            from IPython import embed; embed(); raise ValueError()
        else:
            if self.symbol_processing == "both":
                spc = text_to_sequence(" ", [self.clean_names[0]])[0]
                spc2 = text_to_sequence(" ", [self.clean_names[1]])[0]
                first_char = int_char_chunks[0]
                first_phones = int_phone_chunks[0]
                for ii in range(len(int_char_chunks) - 1):
                    first_char += [spc]
                    first_char += int_char_chunks[ii + 1]
                for ii in range(len(int_phone_chunks) - 1):
                    first_phones += [spc2]
                    first_phones += int_phone_chunks[ii + 1]
                return first_char, first_phones
            #w = [sequence_to_text(int_char_chunks[i], [self.clean_names[0]]) for i in range(len(int_char_chunks))]
            #p = [sequence_to_text(int_phone_chunks[i], [self.clean_names[1]]) for i in range(len(int_phone_chunks))]
            # 50/50 split right now for blended, allow this balance to be set manually?
            char_phone_mask = [0] * len(int_char_chunks) + [1] * len(int_phone_chunks)
            self.random_state.shuffle(char_phone_mask)
            char_phone_mask = char_phone_mask[:len(int_char_chunks)]
            # setting char_phone_mask to 0 will use chars, 1 will use phones
            # these if statements override the default for blended... (above)
            if self.symbol_processing == "blended_pref":
                char_phone_mask = [0 if len(int_phone_chunks[i]) == 0 else 1 for i in range(len(int_char_chunks))]
            elif self.symbol_processing == "phones_only":
                # set the mask to use only phones
                # all files should have phones because of earlier preproc...
                char_phone_mask = [1 for i in range(len(char_phone_mask))]
            elif self.symbol_processing == "chars_only":
                # only use chars
                char_phone_mask = [0 for i in range(len(char_phone_mask))]

            # if the phones entry is None, the word was OOV or not recognized
            char_phone_int_seq = [int_char_chunks[i] if (len(int_phone_chunks[i]) == 0 or char_phone_mask[i] == 0) else int_phone_chunks[i] for i in range(len(int_char_chunks))]
            # check the inverse is ok
            #char_phone_txt = [sequence_to_text(char_phone_int_seq[i], [self.clean_names[char_phone_mask[i]]]) for i in range(len(char_phone_int_seq))]
            # combine into 1 sequence
            cphi = char_phone_int_seq[0]
            cpm = [char_phone_mask[0]] * len(char_phone_int_seq[0])
            if self.symbol_processing != "phones_only":
                spc = text_to_sequence(" ", [self.clean_names[0]])[0]
            else:
                spc = text_to_sequence(" ", [self.clean_names[1]])[0]
            for i in range(len(char_phone_int_seq[1:])):
                # add space
                cphi += [spc]
                # always treat space as char unless in phones only mode
                if self.symbol_processing != "phones_only":
                    cpm += [0]
                else:
                    cpm += [1]
                cphi += char_phone_int_seq[i + 1]
                cpm += [char_phone_mask[i + 1]] * len(char_phone_int_seq[i + 1])
            # check inverse
            #cpt = "".join([sequence_to_text([cphi[i]], [self.clean_names[cpm[i]]]) for i in range(len(cphi))])
            return cphi, cpm

    def transform_txt(self, char_seq, auto_pronounce=True, phone_seq=None, force_char_spc=True):
        """
        chars format example: "i am learning english."
        phone_seq format example: "@ay @ae@m @l@er@n@ih@ng @ih@ng@g@l@ih@sh"

        phone_seq formatting can be gotten from text, using the pronounce_chars function with 'from tfbldr.datasets.text import pronounce_chars'
            Uses cmudict to do pronunciation
        """
        if phone_seq is None and auto_pronounce is False and self.symbol_processing != "chars_only":
            raise ValueError("phone_seq argument must be provided for iterator with self.symbol_processing != 'chars_only', currently '{}'".format(self.symbol_processing))
        clean_char_seq = cleaners.english_cleaners(char_seq)
        char_seq_chunk = clean_char_seq.split(" ")
        dirty_seq_chunk = char_seq.split(" ")

        if auto_pronounce is True:
            if phone_seq is not None:
                raise ValueError("auto_pronounce set to True, but phone_seq was provided! Pass phone_seq=None for auto_pronounce=True")
            # take out specials then put them back...
            specials = "!?.,;:"
            puncts = "!?."
            tsc = []
            for n, csc in enumerate(char_seq_chunk):
                broke = False
                for s in specials:
                    if s in csc:
                        new = csc.replace(s, "")
                        tsc.append(new)
                        broke = True
                        break
                if not broke:
                    tsc.append(csc)

            if self.symbol_processing == "blended_pref":
                chunky_phone_seq_chunk = [pronounce_chars(w, raw_line=dirty_seq_chunk[ii], cmu_only=True) for ii, w in enumerate(tsc)]
                phone_seq_chunk = [cpsc[0] if cpsc != None else None for cpsc in chunky_phone_seq_chunk]
            else:
                phone_seq_chunk = [pronounce_chars(w) for w in tsc]
            for n, psc in enumerate(phone_seq_chunk):
                for s in specials:
                    if char_seq_chunk[n][-1] == s and phone_seq_chunk[n] != None:
                        phone_seq_chunk[n] += char_seq_chunk[n][-1]
                        #if char_seq_chunk[n][-1] in puncts and n != (len(phone_seq_chunk) - 1):
                        #    # add eos
                        #    char_seq_chunk[n] += "~"
                        #    phone_seq_chunk[n] += "~"
                        break
        else:
            raise ValueError("Non auto_pronounce setting not yet configured")

        if len(char_seq_chunk) != len(phone_seq_chunk):
            raise ValueError("Char and phone chunking resulted in different lengths {} and {}!\n{}\n{}".format(len(char_seq_chunk), len(phone_seq_chunk), char_seq_chunk, phone_seq_chunk))

        if self.symbol_processing != "phones_only":
            spc = text_to_sequence(" ", [self.clean_names[0]])[0]
        else:
            spc = text_to_sequence(" ", [self.clean_names[1]])[0]

        int_char_chunks = []
        int_phone_chunks = []
        for n in range(len(char_seq_chunk)):
            int_char_chunks.append(text_to_sequence(char_seq_chunk[n], [self.clean_names[0]])[:-1])
            if phone_seq_chunk[n] == None:
                int_phone_chunks.append([])
            else:
                int_phone_chunks.append(text_to_sequence(phone_seq_chunk[n], [self.clean_names[1]])[:-2])

        # check inverses
        # w = [sequence_to_text(int_char_chunks[i], [self.clean_names[0]]) for i in range(len(int_char_chunks))]
        # p = [sequence_to_text(int_phone_chunks[i], [self.clean_names[1]]) for i in range(len(int_phone_chunks))]

        # TODO: Unify the two functions?
        char_phone_mask = [0] * len(int_char_chunks) + [1] * len(int_phone_chunks)
        self.random_state.shuffle(char_phone_mask)
        char_phone_mask = char_phone_mask[:len(int_char_chunks)]
        # setting char_phone_mask to 0 will use chars, 1 will use phones
        # these if statements override the default for blended... (above)
        if self.symbol_processing == "blended_pref":
            char_phone_mask = [0 if len(int_phone_chunks[i]) == 0 else 1 for i in range(len(int_char_chunks))]
        elif self.symbol_processing == "phones_only":
            # set the mask to use only phones
            # all files should have phones because of earlier preproc...
            char_phone_mask = [1 for i in range(len(char_phone_mask))]
        elif self.symbol_processing == "chars_only":
            # only use chars
            char_phone_mask = [0 for i in range(len(char_phone_mask))]

        # if the phones entry is None, the word was OOV or not recognized
        char_phone_int_seq = [int_char_chunks[i] if (len(int_phone_chunks[i]) == 0 or char_phone_mask[i] == 0) else int_phone_chunks[i] for i in range(len(int_char_chunks))]
        # check the inverse is ok
        # char_phone_txt = [sequence_to_text(char_phone_int_seq[i], [self.clean_names[char_phone_mask[i]]]) for i in range(len(char_phone_int_seq))]
        # combine into 1 sequence
        cphi = char_phone_int_seq[0]
        cpm = [char_phone_mask[0]] * len(char_phone_int_seq[0])
        if force_char_spc or self.symbol_processing != "phones_only":
            spc = text_to_sequence(" ", [self.clean_names[0]])[0]
        else:
            spc = text_to_sequence(" ", [self.clean_names[1]])[0]
        for i in range(len(char_phone_int_seq[1:])):
            # add space
            cphi += [spc]
            # always treat space as char unless in phones only mode
            if force_char_spc or self.symbol_processing != "phones_only":
                cpm += [0]
            else:
                cpm += [1]
            cphi += char_phone_int_seq[i + 1]
            cpm += [char_phone_mask[i + 1]] * len(char_phone_int_seq[i + 1])
        # trailing space
        #cphi = cphi + [spc]
        # trailing eos
        cphi = cphi + [1]
        # add trailing symbol
        if self.symbol_processing != "phones_only":
            cpm += [0]
        else:
            cpm += [1]
        # check inverse
        #cpt = "".join([sequence_to_text([cphi[i]], [self.clean_names[cpm[i]]]) for i in range(len(cphi))])
        #if None in phone_seq_chunk:
            #print("NUN")
            #print(cpt)
            #from IPython import embed; embed(); raise ValueError()
        return cphi, cpm

    def inverse_transform_txt(self, int_seq, mask):
        """
        mask set to zero will use chars, mask set to 1 will use phones

        should invert the transform_txt function
        """
        cphi = int_seq
        cpm = mask
        cpt = "".join([sequence_to_text([cphi[i]], [self.clean_names[cpm[i]]]) for i in range(len(cphi))])
        return cpt
        # setting char_phone_mask to 0 will use chars, 1 will use phones



if __name__ == "__main__":
    # example usage
    """
    # convert all characters to indices
    data_in_int = [[symbol_to_index[s] for s in s_i] for s_i in data_in]
    data_out_int = [[symbol_to_index[s] for s in s_i] for s_i in data_out]

    train_itr = masked_list_multisequence_iterator([data_in_int[:3000], data_out_int[:3000]], minibatch_size, random_state=random_state, pad_with=0)
    valid_itr = masked_list_multisequence_iterator([data_in_int[-100:], data_out_int[-100:]], minibatch_size, random_state=random_state, pad_with=0)
    """
