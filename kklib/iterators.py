import numpy as np

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

if __name__ == "__main__":
    # example usage
    """
    # convert all characters to indices
    data_in_int = [[symbol_to_index[s] for s in s_i] for s_i in data_in]
    data_out_int = [[symbol_to_index[s] for s in s_i] for s_i in data_out]

    train_itr = masked_list_multisequence_iterator([data_in_int[:3000], data_out_int[:3000]], minibatch_size, random_state=random_state, pad_with=0)
    valid_itr = masked_list_multisequence_iterator([data_in_int[-100:], data_out_int[-100:]], minibatch_size, random_state=random_state, pad_with=0)
    """
