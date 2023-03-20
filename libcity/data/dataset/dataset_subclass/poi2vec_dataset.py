import numpy as np

from libcity.data.dataset.dataset_subclass.word2vec_dataset import W2VData, HuffmanTree
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


class P2VData(W2VData, TrafficRepresentationDataset):
    """
    Data supporter for POI2Vec.
    """
    def __init__(self, sentences, coor_df, theta, indi_context):
        """
        :param coor_df: coordinate dataframe, with poi_index as its index, and two columns 'lat' and 'lng'.
        """
        super().__init__(sentences, indi_context)
        self.sentences = sentences

        self.tree = AreaNode(left=coor_df['lng'].min(), right=coor_df['lng'].max(),
                             top=coor_df['lat'].max(), bottom=coor_df['lat'].min(),
                             level=0, theta=theta)
        self.tree.build()

        # Mapping poi_id to positive and negative node indices.
        # Every poi_id corresponds to four corners, so will be mapped to four lists of pos/neg,
        # and four probability values.
        self.id2area_pos = {}
        self.id2area_neg = {}
        self.id2prob = {}
        # Mapping poi_id to all area leaf nodes it locates in.
        self.id2area_leaf = {}

        # Record the set of POIs corresponds to leaf nodes in the AreaTree.
        # Format: leaf_node_id -> [POIs that locates in this area]
        leaf2poi = {}
        self.leaf2huffman_tree = {}  # Records the Huffman tree corresponding to this leaf.
        self.leaf2offset = {}  # Records the node ID offset corresponding to each leaf node.

        for poi_id, (lat, lng) in coor_df.iterrows():
            # The rectangle class represents the influence area of location.
            poi_rec = Rec(lng - 0.5 * theta, lng + 0.5 * theta,
                          lat + 0.5 * theta, lat - 0.5 * theta)
            # All four corners of the fore-defined rectangle.
            poi_corners = [(lng - 0.5 * theta, lat - 0.5 * theta), (lng - 0.5 * theta, lat + 0.5 * theta),
                           (lng + 0.5 * theta, lat - 0.5 * theta), (lng + 0.5 * theta, lat + 0.5 * theta)]

            # Positive and negative node indices of this POI.
            area_pos = [[], [], [], []]
            area_neg = [[], [], [], []]
            # Probability distribution and area node IDs of this POI.
            prob, area_leaf = [], []

            for i, corner in enumerate(poi_corners):
                # Check where the four corners of the influence area locates at in the AreaTree.
                # Get the route and huffman code of four corners of a location in the AreaTree.
                route_row, code_row = self.tree.find_route(*corner)
                for route, code in zip(route_row[1:], code_row):
                    if code == 0:
                        area_pos[i].append(route)
                    else:
                        area_neg[i].append(route)

                leaf_node_id = route_row[0]
                leaf_node = AreaNode.leaf_id2node[leaf_node_id]
                area_leaf.append(leaf_node_id)

                leaf_rec = Rec(leaf_node.left, leaf_node.right, leaf_node.top, leaf_node.bottom)
                prob.append(leaf_rec.overlap(poi_rec))
                leaf2poi[leaf_node_id] = leaf2poi.get(leaf_node_id, set()) | {poi_id}
            prob = np.divide(prob, sum(prob)).tolist()

            self.id2area_pos[poi_id] = area_pos
            self.id2area_neg[poi_id] = area_neg
            self.id2prob[poi_id] = prob
            self.id2area_leaf[poi_id] = area_leaf
        print('Number of area leaf nodes:', len(leaf2poi))

        _total_offset = AreaNode.total_count
        for leaf_id, poi_set in leaf2poi.items():
            select_poi_freq = self.word_freq[np.isin(self.word_freq[:, 0].astype(int),
                                                     np.array(list(poi_set)).astype(int))]
            if select_poi_freq.shape[0] > 0:
                # Create a huffman tree given the POIs that locate in this Area.
                sub_huffman_tree = HuffmanTree(select_poi_freq)
                self.leaf2huffman_tree[leaf_id] = sub_huffman_tree
                self.leaf2offset[leaf_id] = _total_offset
                _total_offset += sub_huffman_tree.num_inner_nodes
        self.total_offset = _total_offset + 1

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

