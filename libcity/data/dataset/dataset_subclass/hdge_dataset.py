import os

import numpy
import numpy as np
from libcity.data.dataset import TrafficRepresentationDataset


class HDGEDataset(TrafficRepresentationDataset):
 def __init__(self,config):
     self.config = config
     self.dataset = self.config.get('dataset', '')
     self.data_path = './raw_data/' + self.dataset + '/'
     self.geo_file = self.config.get('geo_file', self.dataset)
     self.rel_file = self.config.get('rel_file', self.dataset)
     self.dyna_file = self.config.get('dyna_file', self.dataset)
     self.time_slice = self.config.get('time_slice',8)
     self.C = self.config.get('C',1)
     assert (24%self.time_slice == 0)
     self.flow_graph_path = './libcity/cache/HDGE_{}/{}_slice_flow_graph.npy'.format(self.dataset,self.time_slice)
     self.spatial_graph_path = './libcity/cache/HDGE_{}/C={}_spatial_graph.npy'.format(self.dataset,self.C)
     assert os.path.exists(self.data_path + self.geo_file + '.geo')
     assert os.path.exists(self.data_path + self.rel_file + '.rel')
     assert os.path.exists(self.data_path + self.dyna_file + '.dyna')
     super().__init__(config)
     self.construct_flow_graph()
     self.construct_spatial_graph()
     self.construct_od_matrix()

 def get_data(self):
    return None, None, None

 def construct_flow_graph(self):
    """
    这里需要对时间分片，一共是time_slice * node_num个节点
    :return:
    """
    if os.path.exists(self.flow_graph_path):
        self.flow_graph = np.load(self.flow_graph_path)
        self._logger.info("finish consturcting flow graph")
        return
    time_each_slice = 24/self.time_slice
    flow_graph = np.zeros([self.time_slice,self.num_nodes,self.num_nodes])
    for traj,time_list in zip(self.traj_road,self.traj_time):
        origin_region = list(self.road2region[self.road2region['origin_id'] == traj[0]]['destination_id'])[0]
        destination_region = list(self.road2region[self.road2region['origin_id'] == traj[-1]]['destination_id'])[0]
        origin_hour = time_list[6].hour
        flow_graph[origin_hour/time_each_slice][origin_region][destination_region] += 1
    self.flow_graph = flow_graph
    np.save(self.flow_graph_path,self.flow_graph)
    self._logger.info("finish consturcting flow graph")

 def construct_spatial_graph(self):
    """

    :return:
    """
    if os.path.exists(self.spatial_graph_path):
        self.spatial_graph = np.load(self.spatial_graph)
        self._logger.info("finish constructing spatial graph")
        return
    self.centroid = self.region_geometry.centroid
    self.spatial_graph = np.zeros([self.num_nodes,self.num_nodes])
    for i in range(self.num_nodes):
        for j in range(self.num_nodes):
            distance = self.centroid[i].distance(self.centroid[j])
            self.spatial_graph[i][j] = numpy.exp(-self.C * distance)
    np.save(self.spatial_graph_path,self.spatial_graph)
    self._logger.info("finish consturcting spatial graph")

 def construct_od_matrix(self):
     assert self.representation_object == "region"
     self.od_label = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
     for traj in self.traj_road:
         origin_region = list(self.road2region[self.road2region['origin_id'] == traj[0]]['destination_id'])[0]
         destination_region = list(self.road2region[self.road2region['origin_id'] == traj[-1]]['destination_id'])[0]
         self.od_label[origin_region][destination_region] += 1
     return self.od_label

 def get_data_feature(self):
     """
     返回一个 dict，包含数据集的相关特征

     Returns:
         dict: 包含数据集的相关特征的字典
     """
     return {"flow_graph":self.flow_graph,"spatial_graph":self.spatial_graph,
             "num_nodes": self.num_nodes,"geo_to_ind": self.geo_to_ind, "ind_to_geo": self.ind_to_geo,
                "label":{"od_matrix_predict":self.od_label,"function_cluster":np.array(self.function)}}



