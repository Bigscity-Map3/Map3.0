from datetime import datetime
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
     super().__init__(config)
     assert (24%self.time_slice == 0)
     if not os.path.exists('./libcity/cache/HDGE_{}'.format(self.dataset)):
         os.mkdir('./libcity/cache/HDGE_{}'.format(self.dataset))
     self.od_label_path = './libcity/cache/HDGE_{}/od_label_{}.npy'.format(self.dataset, self.remove_node_type)
     self.flow_graph_path = './libcity/cache/HDGE_{}/{}_slice_flow_graph.npy'.format(self.dataset,self.time_slice)
     self.spatial_graph_path = './libcity/cache/HDGE_{}/C={}_spatial_graph.npy'.format(self.dataset,self.C)
     self.combine_graph_path = './libcity/cache/HDGE_{}/C={} and {}_slice.npy'.format(self.dataset ,self.C ,self.time_slice)
     assert os.path.exists(self.data_path + self.geo_file + '.geo')
     assert os.path.exists(self.data_path + self.rel_file + '.rel')
     assert os.path.exists(self.data_path + self.dyna_file + '.dyna')
     self.construct_flow_graph()
     self.construct_spatial_graph()
     self.construct_od_matrix()
     self.combine_graph = self.combine_matrix(self.flow_graph,self.spatial_graph)

 def get_data(self):
    return None, None, None

 def construct_flow_graph(self):
    """
    这里需要对时间分片，一共是time_slice * node_num个节点
    :return:
    """
    if os.path.exists(self.flow_graph_path):
        self.flow_graph = np.load(self.flow_graph_path)
        self._logger.info("finish constructing flow graph")
        return
    time_each_slice = 24//self.time_slice
    flow_graph = np.zeros([self.time_slice,self.num_nodes,self.num_nodes])
    for traj,time_list in zip(self.traj_road,self.traj_time):
        origin_region_geo_id = int(list(self.road2region[self.road2region['origin_id'] == traj[0]]['destination_id'])[0])
        destination_region_geo_id = int(
            list(self.road2region[self.road2region['origin_id'] == traj[-1]]['destination_id'])[0])
        if origin_region_geo_id in self.geo_to_ind and destination_region_geo_id in self.geo_to_ind:
            origin_region = self.geo_to_ind[origin_region_geo_id]
            destination_region = self.geo_to_ind[destination_region_geo_id]
            origin_hour = datetime.strptime(time_list[0], '%Y-%m-%d %H:%M:%S').hour
            flow_graph[origin_hour//time_each_slice][origin_region][destination_region] += 1
    self.flow_graph = flow_graph
    np.save(self.flow_graph_path,self.flow_graph)
    self._logger.info("finish consturcting flow graph")

 def construct_spatial_graph(self):
    """

    :return:
    """
    if os.path.exists(self.spatial_graph_path):
        self.spatial_graph = np.load(self.spatial_graph_path)
        self._logger.info("finish constructing spatial graph")
        return
    self.centroid = self.region_geometry.centroid
    self.spatial_graph = np.zeros([self.num_nodes,self.num_nodes])
    for i in range(self.num_nodes):
        for j in range(i,self.num_nodes):
            distance = self.centroid[self.ind_to_geo[i]].distance(self.centroid[self.ind_to_geo[j]])
            self.spatial_graph[i][j] = numpy.exp(-self.C * distance)
            self.spatial_graph[j][i] = numpy.exp(-self.C * distance)
    np.save(self.spatial_graph_path,self.spatial_graph)
    self._logger.info("finish consturcting spatial graph")

 def construct_od_matrix(self):
     if os.path.exists(self.od_label_path):
         self.od_label = np.load(self.od_label_path)
         self._logger.info("finish construct od graph")
         return
     assert self.representation_object == "region"
     self.od_label = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
     for traj in self.traj_road:
         origin_region_geo_id = list(self.road2region[self.road2region['origin_id'] == traj[0]]['destination_id'])[0]
         destination_region_geo_id = \
         list(self.road2region[self.road2region['origin_id'] == traj[-1]]['destination_id'])[0]
         if origin_region_geo_id in self.geo_to_ind and destination_region_geo_id in self.geo_to_ind:
             origin_region = self.geo_to_ind[origin_region_geo_id]
             destination_region = self.geo_to_ind[destination_region_geo_id]
             self.od_label[origin_region][destination_region] += 1
     np.save(self.od_label_path, self.od_label)
     self._logger.info("finish construct od graph")
     return self.od_label

 def combine_matrix(self, flow_mx, spatial_mx):
     """
     将两个矩阵归一化，合成一个图
     :param flow_mx: 流图的结构[time_slice,num_node,num_node]
     :param spatial_mx: [num_node,num_node]
     :return:combined_mx:[time_slice,num_node,num_node]
     """
     if os.path.exists(self.combine_graph_path):
         combine_graph = np.load(self.combine_graph_path)
         self._logger.info("finish combine graph")
         return combine_graph

     spatial_mx_scaler = np.zeros(shape=(self.num_nodes, self.num_nodes), dtype=np.float32)
     combined_mx = np.zeros(shape=(self.time_slice, self.num_nodes, self.num_nodes), dtype=np.float32)
     for i in range(self.num_nodes):
         unnormalized_probs = [spatial_mx[i][j] for j in range(self.num_nodes)]
         norm_cost = 2 * sum(unnormalized_probs)
         spatial_mx_scaler[i] = spatial_mx[i] / norm_cost

     for t in range(self.time_slice):
         for i in range(self.num_nodes):
             unnormalized_probs_probs = [flow_mx[t][i][j] for j in range(self.num_nodes)]
             norm_cost = 2 * sum(unnormalized_probs_probs)
             if norm_cost == 0:
                 combined_mx[t][i] = spatial_mx_scaler[i] * 2
             else:
                 combined_mx[t][i] = flow_mx[t][i] / norm_cost + spatial_mx_scaler[i]
     np.save(self.combine_graph_path,combined_mx)
     self._logger.info("finish combine graph")
     return combined_mx

 def get_data_feature(self):
     """
     返回一个 dict，包含数据集的相关特征

     Returns:
         dict: 包含数据集的相关特征的字典
     """
     return {"combine_graph":self.combine_graph,"flow_graph":self.flow_graph,"spatial_graph":self.spatial_graph,"time_slice":self.time_slice,
             "C":self.C,"num_nodes": self.num_nodes,"geo_to_ind": self.geo_to_ind, "ind_to_geo": self.ind_to_geo,
                "label":{"od_matrix_predict":self.od_label,"function_cluster":self.function}}



