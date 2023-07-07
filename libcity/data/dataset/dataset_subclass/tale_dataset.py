import numpy as np
import os
import pandas as pd
from libcity.data.dataset.poi_representation_dataset import PoiRepresentationDataset


class TaleDataset(PoiRepresentationDataset):
    def __init__(self, config):
        self.config = config
        self.dataset = self.config.get('dataset', '')
        self.data_path = './raw_data/' + self.dataset + '/'
        self.geo_file = self.config.get('geo_file', self.dataset)
        self.rel_file = self.config.get('rel_file', self.dataset)
        self.dyna_file = self.config.get('dyna_file', self.dataset)
        if not os.path.exists('./libcity/cache/GEOTEASER_{}'.format(self.dataset)):
            os.mkdir('./libcity/cache/GEOTEASER_{}'.format(self.dataset))
        super().__init__(config)


        self.time_slice_len = self.config.get('tale_slice', 60)
        self.influence_span_length = self.config.get('tale_span', 0)
        self.indi_context = self.config.get('tale_indi_context', True)
        self.timestamps = self.embed_train_timestamp

        temp_sentence = []
        slices, props = [], []
        for sentence, timestamp in zip(self.sentences, self.timestamps):
            slice_row, prop_row = [], []
            minute = list(map(lambda x: x / 60, timestamp))
            for token, one_minute in zip(sentence, minute):
                slice, prop = self.gen_all_slots(one_minute, self.time_slice_len, self.influence_span_length)
                temp_sentence += ['{}-{}'.format(token, s) for s in slice]
                slice_row.append(slice)
                prop_row.append(prop)
            slices.append(slice_row)
            props.append(prop_row)

        self.word_freq = self.get_token_freq([temp_sentence])

        self.id2index = {id: index for index, id in enumerate(self.word_freq[:, 0])}
        self.word_freq[:, 0] = np.array([self.id2index[x] for x in self.word_freq[:, 0]])
        self.word_freq = self.word_freq.astype(int)
        self.huffman_tree = HuffmanTree(self.word_freq)

        self.slices = slices
        self.props = props


    def gen_all_slots(self, minute, time_slice_length, influence_span_length):
            """
            @param minute: UTC timestamp in minute.
            @param time_slice_length: length of one slot in seconds.
            @param influence_span_length: length of influence span in seconds.
            """

            def _cal_slice(x):
                return int((x % (24 * 60)) / time_slice_length)

            # max_num_slots = math.ceil(time_slice_length / influence_span_length) + 1
            if influence_span_length == 0:
                slices, props = [_cal_slice(minute)], [1.0]

            else:
                minute_floors = list({minute - influence_span_length / 2, minute + influence_span_length / 2} |
                                     set(range((int((
                                                                minute - influence_span_length / 2) / time_slice_length) + 1) * time_slice_length,
                                               int(minute + influence_span_length / 2), time_slice_length)))
                minute_floors.sort()

                slices = [_cal_slice(time_minute) for time_minute in minute_floors[:-1]]
                props = [(minute_floors[index + 1] - minute_floors[index]) / influence_span_length
                         for index in range(len(minute_floors) - 1)]

            return slices, props

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


    def get_data_feature(self):
        self.num_temp_vocab= len(self.id2index)
        self.poi_num = self.traj_poi['poi_id'].value_counts().count()
        self.path_pairs = self.get_path_pairs(self.w2v_window_size)
        return {"num_loc": self.poi_num,
                "path_pairs": self.path_pairs,
                "num_temp_vocab":self.num_temp_vocab}


class HuffmanNode:
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
        # Because the ID of leaf nodes will not be needed during calculation, you can minus this value to all inner nodes' IDs to save some space in output embeddings.

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
