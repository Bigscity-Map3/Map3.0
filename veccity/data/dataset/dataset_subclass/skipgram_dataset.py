from .utils import W2VData,gen_neg_sample_table, HuffmanTree
import numpy as np

class SkipGramData(W2VData):
    def __init__(self, sentences, sample=1e-3):
        super().__init__(sentences)
        self.sentences = sentences

        # Initialize negative sampling table.
        self.sample_table = gen_neg_sample_table(self.word_freq, clip_ratio=sample)

    def gen_pos_pairs(self, window_size):
        pos_pairs = []
        for sentence in self.sentences:
            for i in range(0, len(sentence) - (2 * window_size + 1)):
                target = sentence[i + window_size]
                context = sentence[i:i + window_size] + sentence[i + window_size + 1:i + 2 * window_size + 1]
                # pos_pairs += [[target, [c]] for c in context]
                pos_pairs.append([target, context])
        return pos_pairs

    def get_neg_v_sampling(self, batch_size, num_neg):
        neg_v = np.random.choice(self.sample_table, size=(batch_size, num_neg))
        return neg_v


class HSData(W2VData):
    def __init__(self, sentences):
        super().__init__(sentences)
        self.sentences = sentences
        self.huffman_tree = HuffmanTree(self.word_freq)

    def get_path_pairs(self, window_size):
        path_pairs = []
        for sentence in self.sentences:
            for i in range(0, len(sentence) - (2 * window_size + 1)):
                target = sentence[i + window_size]
                pos_path = self.huffman_tree.id2pos[target]
                neg_path = self.huffman_tree.id2neg[target]
                context = sentence[i:i + window_size] + sentence[i + window_size + 1:i + 2 * window_size + 1]
                # path_pairs += [[[c], pos_path, neg_path] for c in context]
                path_pairs.append([context, pos_path, neg_path])
        return path_pairs
