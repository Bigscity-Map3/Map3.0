import numpy as np
import os
import pandas as pd
from libcity.data.dataset.traffic_representation_dataset import TrafficRepresentationDataset


class Rec:
    """
    Rectangle class for calculating the size of overlapping areas.
    """
    def __init__(self, left, right, top, bottom):
        self.__dict__.update(locals())

    def overlap(self, other):
        dx = min(self.top, other.top) - max(self.bottom, other.bottom)
        dy = min(self.right, other.right) - max(self.left, other.left)
        assert dx >= 0 and dy >= 0
        return dx * dy


class AreaNode:
    """
    One Area Node represent one rectangle area in POI2Vec,
    and also serve as root node of the corresponding Huffman Tree.
    """
    total_count = 0
    leaf_id2node = {}

    def __init__(self, left, right, top, bottom, level, theta):
        """
        Initializer of tree node.
        left, right, top, down are used to define the area this node represents.
        """
        # Left and right sub-nodes
        self.ln = None
        self.rn = None
        AreaNode.total_count += 1
        self.id = AreaNode.total_count

        self.__dict__.update(locals())

    def build(self):
        """
        Build this node's sub-nodes.
        """
        if self.level % 2 == 0:
            # If level is even, split area horizontally.
            if (self.right - (self.right + self.left) / 2) > 2 * self.theta:
                self.ln = AreaNode(self.left, (self.left + self.right) / 2, self.top, self.bottom,
                                   level=self.level + 1, theta=self.theta)
                self.rn = AreaNode((self.left + self.right) / 2, self.right, self.top, self.bottom,
                                   level=self.level + 1, theta=self.theta)
                self.ln.build()
                self.rn.build()
            else:
                AreaNode.leaf_id2node[self.id] = self
        else:
            # If level is odd, split area vertically.
            if (self.top - (self.bottom + self.top) / 2) > 2 * self.theta:
                self.ln = AreaNode(self.left, self.right, self.top, (self.bottom + self.top) / 2,
                                   level=self.level + 1, theta=self.theta)
                self.rn = AreaNode(self.left, self.right, (self.bottom + self.top) / 2, self.bottom,
                                   level=self.level + 1, theta=self.theta)
                self.ln.build()
                self.rn.build()
            else:
                AreaNode.leaf_id2node[self.id] = self

    def find_route(self, x, y):
        """
        Find route for given coordinate.
        :return: route and left-right choices. 0 for left, 1 for right.
        """
        if self.ln is None:
            route = [self.id]
            code = []
            return route, code

        if self.level % 2 == 0:
            # Left sub-node's right boundary can view as split boundary of this node.
            if self.ln.right < x:
                route, code = self.rn.find_route(x, y)
                code.append(1)
            else:
                route, code = self.ln.find_route(x, y)
                code.append(0)
        else:
            # Left sub-node's bottom boundary can view as split boundary of this node.
            if self.ln.bottom < y:
                route, code = self.ln.find_route(x, y)
                code.append(0)
            else:
                route, code = self.rn.find_route(x, y)
                code.append(1)

        route.append(self.id)
        return route, code

    def __repr__(self):
        return f'AreaNode [<-{self.left}, ^{self.top}, v{self.bottom}, ->{self.right}], #{self.id}, LV{self.level}'


class GeoTeaserDataset(TrafficRepresentationDataset):
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
        self.data_preprocess()

    def data_preprocess(self):
        assert self.representation_object == "poi"

        traj_num = self.dyna_file_raw['total_traj_id'].max()+1
        poi_list = []
        for i in range(traj_num):
            poi_list.append(self.road2poi[self.dyna_file_raw[i]['geo_id']])
        self.df_dyna_file_raw = pd.DataFrame(self.dyna_file_raw)
        self.df_dyna_file_raw = self.dyna_file[['time', 'entity_id','traj_id','geo_id']]

        self.df_dyna_file_raw['geo_id'] = pd.Series(poi_list)
        print(self.df_dyna_file_raw.head())





    def gen_path_pairs(self, window_size):
        path_pairs = []
        for sentence in self.sentences:
            for i in range(0, len(sentence) - (2 * window_size + 1) + 1):
                target = sentence[i+window_size]
                area_pos = self.id2area_pos[target]
                area_neg = self.id2area_neg[target]
                huffman_pos = [(np.array(self.leaf2huffman_tree[area_leaf].id2pos[target]) + self.leaf2offset[area_leaf]).tolist() + area_pos[i]
                               for i, area_leaf in enumerate(self.id2area_leaf[target])]
                huffman_neg = [(np.array(self.leaf2huffman_tree[area_leaf].id2neg[target]) + self.leaf2offset[area_leaf]).tolist() + area_neg[i]
                               for i, area_leaf in enumerate(self.id2area_leaf[target])]
                context = sentence[i:i+window_size] + sentence[i+window_size+1:i+2*window_size+1]
                prob = self.id2prob[target]
                if self.indi_context:
                    path_pairs += [[[c], huffman_pos, huffman_neg, prob] for c in context]
                else:
                    path_pairs.append([context, huffman_pos, huffman_neg, prob])
        return path_pairs

    def get_data_feature(self):
        """
        返回一个 dict，包含数据集的相关特征

        Returns:
            dict: 包含数据集的相关特征的字典
        """

        return {

        }
