from veccity.data.dataset.dataset_subclass.utils import gen_all_slots, W2VData,HuffmanTree
import numpy as np

class TaleData(W2VData):
    def __init__(self, sentences, timestamps, time_slice_len, influence_span_length, indi_context=True):
        """
        @param sentences: sequences of locations.
        @param minutes: UTC minutes corresponding to sentences.
        @param time_slice_len: length of one time slice, in minute.
        """
        temp_sentence = []
        slices, props = [], []
        for sentence, timestamp in zip(sentences, timestamps):
            slice_row, prop_row = [], []
            minute = list(map(lambda x: x / 60, timestamp))
            for token, one_minute in zip(sentence, minute):
                slice, prop = gen_all_slots(one_minute, time_slice_len, influence_span_length)
                temp_sentence += ['{}-{}'.format(token, s) for s in slice]
                slice_row.append(slice)
                prop_row.append(prop)
            slices.append(slice_row)
            props.append(prop_row)

        super().__init__([temp_sentence])

        self.id2index = {id: index for index, id in enumerate(self.word_freq[:, 0])}
        self.word_freq[:, 0] = np.array([self.id2index[x] for x in self.word_freq[:, 0]])
        self.word_freq = self.word_freq.astype(int)
        self.huffman_tree = HuffmanTree(self.word_freq)

        self.sentences = sentences
        self.slices = slices
        self.props = props
        self.indi_context = indi_context

    def get_path_pairs(self, window_size):
        path_pairs = []
        for sentence, slice, prop in zip(self.sentences, self.slices, self.props):
            for i in range(0, len(sentence) - (2 * window_size + 1)):
                temp_targets = ['{}-{}'.format(sentence[i + window_size], s) for s in slice[i + window_size]]
                target_indices = [self.id2index[t] for t in temp_targets]  # (num_overlapping_slices)
                pos_paths = [self.huffman_tree.id2pos[t] for t in target_indices]
                neg_paths = [self.huffman_tree.id2neg[t] for t in target_indices]
                context = sentence[i:i + window_size] + sentence[i + window_size + 1:i + 2 * window_size + 1]
                if self.indi_context:
                    path_pairs += [[[c], pos_paths, neg_paths, prop[i + window_size]] for c in context]
                else:
                    path_pairs.append([context, pos_paths, neg_paths, prop[i + window_size]])
        return path_pairs