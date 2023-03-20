from collections import Counter
import numpy as np

from libcity.data.dataset.traffic_representation_dataset import TrafficRepresentationDataset


def gen_token_freq(sentences):
    freq = Counter()
    for sentence in sentences:
        freq.update(sentence)
    freq = np.array(sorted(freq.items()))
    return freq


def gen_neg_sample_table(freq_array, sample_table_size=1e8, clip_ratio=1e-3):
    sample_table = []
    pow_freq = freq_array[:, 1] ** 0.75

    words_pow = pow_freq.sum()
    ratio = pow_freq / words_pow
    ratio = np.clip(ratio, a_min=0, a_max=clip_ratio)
    ratio = ratio / ratio.sum()

    count = np.round(ratio * sample_table_size)
    for word_index, c in enumerate(count):
        sample_table += [freq_array[word_index, 0]] * int(c)
    sample_table = np.array(sample_table)
    return sample_table


class HuffmanNode:
    """
    A node in the Huffman tree.
    """
    def __init__(self, id, frequency):
        """
        :param id: index of word (leaf nodes) or inner nodes.
        :param frequency: frequency of word.
        """
        self.id = id
        self.frequency = frequency

        self.left = None
        self.right = None
        self.father = None
        self.huffman_code = []
        self.path = []  # (path from root node to leaf node)

    def __str__(self):
        return 'HuffmanNode#{},freq{}'.format(self.id, self.frequency)


class HuffmanTree:
    """
    Huffman Tree class used for Hierarchical Softmax calculation.
    """
    def __init__(self, freq_array):
        """
        :param freq_array: numpy array containing all words' frequencies, format {id: frequency}.
        """
        self.num_words = freq_array.shape[0]
        self.id2code = {}
        self.id2path = {}
        self.id2pos = {}
        self.id2neg = {}
        self.root = None  # Root node of this tree.
        self.num_inner_nodes = 0  # Records the number of inner nodes of this tree.

        unmerged_node_list = [HuffmanNode(id, frequency) for id, frequency in freq_array]
        self.tree = {node.id: node for node in unmerged_node_list}
        self.id_offset = max(self.tree.keys())  # Records the starting-off ID of this tree.
        # Because the ID of leaf nodes will not be needed during calculation,
        # you can minus this value to all inner nodes' IDs to save some space in output embeddings.

        self._offset = self.id_offset
        self._build_tree(unmerged_node_list)
        self._gen_path()
        self._get_all_pos_neg()

    def _merge_node(self, node1: HuffmanNode, node2: HuffmanNode):
        """
        Merge two nodes into one, adding their frequencies.
        """
        sum_freq = node1.frequency + node2.frequency
        self._offset += 1
        mid_node_id = self._offset
        father_node = HuffmanNode(mid_node_id, sum_freq)
        if node1.frequency >= node2.frequency:
            father_node.left, father_node.right = node1, node2
        else:
            father_node.left, father_node.right = node2, node1
        self.tree[mid_node_id] = father_node
        self.num_inner_nodes += 1
        return father_node

    def _build_tree(self, node_list):
        while len(node_list) > 1:
            i1, i2 = 0, 1
            if node_list[i2].frequency < node_list[i1].frequency:
                i1, i2 = i2, i1
            for i in range(2, len(node_list)):
                if node_list[i].frequency < node_list[i2].frequency:
                    i2 = i
                    if node_list[i2].frequency < node_list[i1].frequency:
                        i1, i2 = i2, i1
            father_node = self._merge_node(node_list[i1], node_list[i2])
            assert not i1 == i2
            if i1 < i2:
                node_list.pop(i2)
                node_list.pop(i1)
            else:
                node_list.pop(i1)
                node_list.pop(i2)
            node_list.insert(0, father_node)
        self.root = node_list[0]

    def _gen_path(self):
        stack = [self.root]
        while len(stack) > 0:
            node = stack.pop()
            while node.left or node.right:
                code = node.huffman_code
                path = node.path
                node.left.huffman_code = code + [1]
                node.right.huffman_code = code + [0]
                node.left.path = path + [node.id]
                node.right.path = path + [node.id]
                stack.append(node.right)
                node = node.left
            id = node.id
            code = node.huffman_code
            path = node.path
            self.tree[id].huffman_code, self.tree[id].path = code, path
            self.id2code[id], self.id2path[id] = code, path

    def _get_all_pos_neg(self):
        for id in self.id2code.keys():
            pos_id = []
            neg_id = []
            for i, code in enumerate(self.tree[id].huffman_code):
                if code == 1:
                    pos_id.append(self.tree[id].path[i] - self.id_offset)  # This will make the generated inner node IDs starting from 1.
                else:
                    neg_id.append(self.tree[id].path[i] - self.id_offset)
            self.id2pos[id] = pos_id
            self.id2neg[id] = neg_id


class W2VData:
    def __init__(self, sentences, indi_context):
        self.indi_context = indi_context
        self.word_freq = gen_token_freq(sentences)  # (num_vocab, 2)


class NSData(W2VData, TrafficRepresentationDataset):
    """
    Data supporter for negative sampling.
    """
    def __init__(self, sentences, indi_context, sample=1e-3):
        super().__init__(sentences, indi_context)
        self.sentences = sentences

        # Initialize negative sampling table.
        self.sample_table = gen_neg_sample_table(self.word_freq, clip_ratio=sample)

    def gen_pos_pairs(self, window_size):
        pos_pairs = []
        for sentence in self.sentences:
            for i in range(0, len(sentence) - (2 * window_size + 1) + 1):
                target = sentence[i+window_size]
                context = sentence[i:i+window_size] + sentence[i+window_size+1:i+2*window_size+1]
                if self.indi_context:
                    pos_pairs += [[target, [c]] for c in context]
                else:
                    pos_pairs.append([target, context])
        return pos_pairs

    def get_neg_v_sampling(self, batch_size, num_neg):
        neg_v = np.random.choice(self.sample_table, size=(batch_size, num_neg))
        return neg_v

    def get_data_feature(self):
        """
        返回一个 dict，包含数据集的相关特征

        Returns:
            dict: 包含数据集的相关特征的字典
        """

        return {

        }


class HSData(W2VData, TrafficRepresentationDataset):
    """
    Data supporter for Hierarchical Softmax.
    """
    def __init__(self, sentences, indi_context):
        super().__init__(sentences, indi_context)
        self.sentences = sentences
        self.huffman_tree = HuffmanTree(self.word_freq)

    def get_path_pairs(self, window_size):
        path_pairs = []
        for sentence in self.sentences:
            for i in range(0, len(sentence) - (2 * window_size + 1) + 1):
                target = sentence[i+window_size]
                pos_path = self.huffman_tree.id2pos[target]
                neg_path = self.huffman_tree.id2neg[target]
                context = sentence[i:i+window_size] + sentence[i+window_size+1:i+2*window_size+1]
                if self.indi_context:
                    path_pairs += [[[c], pos_path, neg_path] for c in context]
                else:
                    path_pairs.append([context, pos_path, neg_path])
        return path_pairs

    def get_data_feature(self):
        """
        返回一个 dict，包含数据集的相关特征

        Returns:
            dict: 包含数据集的相关特征的字典
        """

        return {

        }