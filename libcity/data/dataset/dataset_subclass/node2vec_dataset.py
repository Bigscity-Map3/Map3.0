import os

import numpy as np

from libcity.data.dataset.traffic_representation_dataset import TrafficRepresentationDataset


class Node2VecDataset(TrafficRepresentationDataset):
    def __init__(self,config):
        self.config = config
        self.dataset = self.config.get('dataset', '')
        self.data_path = './raw_data/' + self.dataset + '/'
        self.geo_file = self.config.get('geo_file', self.dataset)
        self.rel_file = self.config.get('rel_file', self.dataset)
        self.dyna_file = self.config.get('dyna_file',self.dataset)
        assert os.path.exists(self.data_path + self.geo_file + '.geo')
        assert os.path.exists(self.data_path + self.rel_file + '.rel')
        assert os.path.exists(self.data_path + self.dyna_file + '.dyna')
        super().__init__(config)
        self.construct_graph()
        self.construct_od_matrix()

    def get_data(self):
        return None,None,None


    def construct_graph(self):
        """
        先采用region_od矩阵当作邻接矩阵
        :return:adj_mx
        """
        assert self.representation_object == "region"
        self.adj_mx = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        for traj in self.traj_road:
            origin_region = list(self.road2region[self.road2region['origin_id'] == traj[0]]['destination_id'])[0]
            destination_region  = list(self.road2region[self.road2region['origin_id'] == traj[-1]]['destination_id'])[0]
            self.adj_mx[origin_region][destination_region] += 1
        return self.adj_mx


    def construct_od_matrix(self):
        assert self.representation_object == "region"
        self.od_label = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        for traj in self.traj_road:
            origin_region = list(self.road2region[self.road2region['origin_id'] == traj[0]]['destination_id'])[0]
            destination_region  = list(self.road2region[self.road2region['origin_id'] == traj[-1]]['destination_id'])[0]
            self.od_label[origin_region][destination_region] += 1
        return self.od_label
    def get_data_feature(self):
        """
        返回一个 dict，包含数据集的相关特征

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        return {"adj_mx": self.adj_mx, "num_nodes": self.num_nodes,
                "geo_to_ind": self.geo_to_ind, "ind_to_geo": self.ind_to_geo,
                "label":{"od_matrix_predict":self.od_label,"function_cluster":np.array(self.function)}}