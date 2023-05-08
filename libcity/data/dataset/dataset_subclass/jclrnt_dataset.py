import os

import numpy as np
import pandas as pd
import torch

from libcity.data.dataset.traffic_representation_dataset import TrafficRepresentationDataset


class JCLRNTDataset(TrafficRepresentationDataset):
    def __init__(self,config):
        self.config = config
        self.dataset = self.config.get('dataset', '')
        self.data_path = './raw_data/' + self.dataset + '/'
        self.label_data_path = './raw_data/' + self.dataset + '/label_data/'
        self.geo_file = self.config.get('geo_file', self.dataset)
        self.rel_file = self.config.get('rel_file', self.dataset)
        self.dyna_file = self.config.get('dyna_file',self.dataset)
        assert os.path.exists(self.data_path + self.geo_file + '.geo')
        assert os.path.exists(self.data_path + self.rel_file + '.rel')
        assert os.path.exists(self.data_path + self.dyna_file + '.dyna')
        super().__init__(config)
        self._load_rel()
        self.prepare_traj_data()
        self.construct_graph()
        self.construct_od_matrix()
        self.read_processed_data()

    def get_data(self):
        return None,None,None


    def construct_graph(self):
        """
        先采用road_od矩阵当作邻接矩阵
        :return:adj_mx
        """
        assert self.representation_object == "road"
        self.edge_index = []
        self.adj_mx = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        for traj in self.traj_road:
            origin_region = list(self.road2region[self.road2region['origin_id'] == traj[0]]['destination_id'])[0]
            destination_region  = list(self.road2region[self.road2region['origin_id'] == traj[-1]]['destination_id'])[0]
            self.adj_mx[origin_region][destination_region] += 1
            self.edge_index.append((origin_region, destination_region))
        self.edge_index = np.array(self.edge_index,dtype=np.int).transpose()
        self.edge_index = torch.Tensor(self.edge_index).int().cuda()
        return self.adj_mx


    def construct_od_matrix(self):
        assert self.representation_object == "road"
        self.od_label = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        for traj in self.traj_road:
            origin_region = list(self.road2region[self.road2region['origin_id'] == traj[0]]['destination_id'])[0]
            destination_region  = list(self.road2region[self.road2region['origin_id'] == traj[-1]]['destination_id'])[0]
            self.od_label[origin_region][destination_region] += 1
        return self.od_label

    def read_processed_data(self):
        assert self.representation_object == "road"
        self.label = {"speed_inference":{},"time_estimation":{}}
        self.length_label  = pd.read_csv(self.label_data_path+"length.csv")

        self.speed_label = pd.read_csv(self.label_data_path+"speed.csv")
        self.speed_label.sort_values(by="index" , inplace=True, ascending=True)

        min_len, max_len = 1, 100
        self.time_label = pd.read_csv(self.label_data_path+"time.csv")

        self.time_label['path'] = self.time_label['trajs'].map(eval)


        self.time_label['path_len'] = self.time_label['path'].map(len)
        self.time_label = self.time_label.loc[(self.time_label['path_len'] > min_len) & (self.time_label['path_len'] < max_len)]

    def prepare_traj_data(self):
        num_samples = self.config.get("num_samples",10000)
        dynafile = pd.read_csv(self.data_path + self.dyna_file + '.dyna',nrows= num_samples)
        traj_num = dynafile['total_traj_id'].max() + 1
        for i in range(traj_num):
            road_list = list(dynafile[dynafile['total_traj_id'] == i]['geo_id'])
            self.traj_road.append(road_list)
        self._logger.info("Loaded file " + self.dyna_file + '.dyna')

    def get_data_feature(self):
        """
        返回一个 dict，包含数据集的相关特征

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        return {"adj_mx": self.adj_mx, "num_nodes": self.num_nodes,
                "traj_road": self.traj_road, "edge_index": self.edge_index,
                "geo_to_ind": self.geo_to_ind, "ind_to_geo": self.ind_to_geo,
                "label":{"od_matrix_predict":self.od_label,"function_cluster":np.array(self.function),
                         'speed_inference':{'speed': self.speed_label},
                         'time_estimation':{'time': self.time_label,'padding_id':self.num_nodes}}}