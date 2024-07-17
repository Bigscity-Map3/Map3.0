from collections import Counter
import math
import heapq
from itertools import zip_longest
import numpy as np
import copy
from math import radians, cos, sin, asin, sqrt

def get_relativeTime(arrival_times): 
    first_time_list = [arrival_times[0] for _ in range(len(arrival_times))]
    return list(map(delta_minutes, arrival_times, first_time_list))

def gen_all_slots(minute, time_slice_length, influence_span_length):
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

        # mask_length = max_num_slots - len(slices)
        # slices += [slices[-1]] * mask_length
        # props += [0.0] * mask_length

    return slices, props


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


class W2VData:
    def __init__(self, sentences):
        self.word_freq = gen_token_freq(sentences)  # (num_vocab, 2)


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

    def __lt__(self, other):
        return self.frequency < other.frequency

    def __eq__(self, other):
        return self.frequency == other.frequency


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

    def _build_tree(self, node_heap):
        heapq.heapify(node_heap)
        while len(node_heap) > 1:
            n1 = heapq.heappop(node_heap)
            n2 = heapq.heappop(node_heap)
            father_node = self._merge_node(n1, n2)
            heapq.heappush(node_heap, father_node)
        self.root = heapq.heappop(node_heap)
        # while len(node_list) > 1:
        #     i1, i2 = 0, 1
        #     if node_list[i2].frequency < node_list[i1].frequency:
        #         i1, i2 = i2, i1
        #     for i in range(2, len(node_list)):
        #         if node_list[i].frequency < node_list[i2].frequency:
        #             i2 = i
        #             if node_list[i2].frequency < node_list[i1].frequency:
        #                 i1, i2 = i2, i1
        #     father_node = self._merge_node(node_list[i1], node_list[i2])
        #     assert not i1 == i2
        #     if i1 < i2:
        #         node_list.pop(i2)
        #         node_list.pop(i1)
        #     else:
        #         node_list.pop(i1)
        #         node_list.pop(i2)
        #     node_list.insert(0, father_node)
        # self.root = node_list[0]

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
                    pos_id.append(self.tree[id].path[i] - self.id_offset)
                    # This will make the generated inner node IDs starting from 1.
                else:
                    neg_id.append(self.tree[id].path[i] - self.id_offset)
            self.id2pos[id] = pos_id
            self.id2neg[id] = neg_id

# CACSR utils
def construct_spatial_matrix_accordingDistance(distance_theta, venue_cnt, venue_lng, venue_lat, gaussian_beta=None):
    SS_distance = np.zeros((venue_cnt, venue_cnt))  
    SS_gaussian_distance = np.zeros((venue_cnt, venue_cnt))  
    SS_proximity = np.zeros((venue_cnt, venue_cnt))  
    for i in range(venue_cnt):
        for j in range(venue_cnt):
            distance_score = distance(venue_lng[i], venue_lat[i], venue_lng[j], venue_lat[j])
            SS_distance[i, j] = distance_score  
            if gaussian_beta is not None:
                distance_gaussian_score = np.exp(-gaussian_beta * distance_score) 
                SS_gaussian_distance[i, j] = distance_gaussian_score  
            if SS_distance[i, j] < distance_theta:  
                SS_proximity[i, j] = 1
        if i % 500 == 0:
            print("constructing spatial matrix: ", i, "/", venue_cnt)
    return SS_distance, SS_proximity, SS_gaussian_distance


def get_delta(arrival_times):
    copy_times = copy.deepcopy(arrival_times)
    copy_times.insert(0, copy_times[0]) 
    copy_times.pop(-1)
    return list(map(delta_minutes, arrival_times, copy_times))


def split_sampleSeq2sessions(sampleSeq_delta_times, min_session_mins):

    sessions = []  
    split_index = []  #
    sessions_lengths = []

    for i in range(1, len(sampleSeq_delta_times)):
        if sampleSeq_delta_times[i] >= min_session_mins:
            split_index.append(i)
    # print('split_index:', split_index)
    if len(split_index) == 0:  
        sessions.append(sampleSeq_delta_times)
        sessions_lengths.append(len(sampleSeq_delta_times))
        # print('sessions:', sessions)
        # print('sessions_lengths:', sessions_lengths)
        return sessions, split_index, sessions_lengths
    else:
        start_index = 0
        for i in range(0, len(split_index)):
            split = split_index[i]
            if split-start_index > 1:  
                sampleSeq_delta_times[start_index] = 0  
                sessions.append(sampleSeq_delta_times[start_index:split])
                sessions_lengths.append(len(sampleSeq_delta_times[start_index:split]))
            start_index = split
        if len(sampleSeq_delta_times[split_index[-1]:]) > 1: 
            sampleSeq_delta_times[split_index[-1]] = 0
            sessions.append(sampleSeq_delta_times[split_index[-1]:])
            sessions_lengths.append(len(sampleSeq_delta_times[split_index[-1]:]))
        # print('sessions:', sessions)
        # print('sessions_lengths:', sessions_lengths)
        return sessions, split_index, sessions_lengths  

def splitSeq_basedonSessions(seq, split_index):

    sessions = []
    if len(split_index) == 0:
        sessions.append(seq)
    else:
        start_index = 0
        for i in range(0, len(split_index)):
            split = split_index[i]
            if split-start_index > 1:
                sessions.append(seq[start_index:split])
            start_index = split
        if len(seq[split_index[-1]:]) > 1: 
            sessions.append(seq[split_index[-1]:])
    return sessions

def delta_minutes(ori, cp):
    delta = (ori.timestamp() - cp.timestamp())/60
    if delta < 0:
        delta = 1
    return delta

def tid_list_48(tm):
    if tm.weekday() in [0, 1, 2, 3, 4]:
        tid = int(tm.hour)
    else:
        tid = int(tm.hour) + 24
    return tid

def distance(lon1, lat1, lon2, lat2):  
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
   
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    if lon1 == 0 or lat1 ==0 or lon2==0 or lat2==0:
        return 0
    # haversine
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  
    return c * r  
