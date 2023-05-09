import os

import numpy as np

from libcity.data.dataset.traffic_representation_dataset import TrafficRepresentationDataset


class Node2VecDataset(TrafficRepresentationDataset):
    def __init__(self,config):
        self.config = config
        super().__init__(config)
        self.dataset = self.config.get('dataset', '')
        self.data_path = './raw_data/' + self.dataset + '/'
        self.geo_file = self.config.get('geo_file', self.dataset)
        self.rel_file = self.config.get('rel_file', self.dataset)
        self.dyna_file = self.config.get('dyna_file',self.dataset)
        assert os.path.exists(self.data_path + self.geo_file + '.geo')
        assert os.path.exists(self.data_path + self.rel_file + '.rel')
        assert os.path.exists(self.data_path + self.dyna_file + '.dyna')
        if not os.path.exists('./libcity/cache/Node2Vec_{}'.format(self.dataset)):
            os.mkdir('./libcity/cache/Node2Vec_{}'.format(self.dataset))
        self.od_label_path = './libcity/cache/Node2Vec_{}/od_label_{}.npy'.format(self.dataset, self.remove_node_type)
        self.construct_od_matrix()
        self.construct_graph()

    def get_data(self):
        return None,None,None


    def construct_graph(self):
        """
        先采用region_od矩阵当作邻接矩阵
        :return:adj_mx
        """
        self.adj_mx = self.od_label


    def construct_od_matrix(self):
        if os.path.exists(self.od_label_path):
            self.od_label = np.load(self.od_label_path)
            self._logger.info("finish construct od graph")
            return
        assert self.representation_object == "region"
        self.od_label = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        for traj in self.traj_road:
            origin_region_geo_id = list(self.road2region[self.road2region['origin_id'] == traj[0]]['destination_id'])[0]
            destination_region_geo_id = list(self.road2region[self.road2region['origin_id'] == traj[-1]]['destination_id'])[0]
            if origin_region_geo_id in self.geo_to_ind and destination_region_geo_id in self.geo_to_ind:
                origin_region = self.geo_to_ind[origin_region_geo_id]
                destination_region = self.geo_to_ind[destination_region_geo_id]
                self.od_label[origin_region][destination_region] += 1
        np.save(self.od_label_path,self.od_label)
        self._logger.info("finish construct od graph")
        return self.od_label
    def get_data_feature(self):
        """
        返回一个 dict，包含数据集的相关特征

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        return {"adj_mx": self.adj_mx, "num_nodes": self.num_nodes,
                "geo_to_ind": self.geo_to_ind, "ind_to_geo": self.ind_to_geo,
                "label":{"od_matrix_predict":self.od_label,"function_cluster":self.function}}