from datetime import datetime
import os
import numpy
import numpy as np
from libcity.data.dataset import TrafficRepresentationDataset
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from tqdm import  *

class MGFNDataset(TrafficRepresentationDataset):
    def __init__(self,config):
        self.config = config
        self.dataset = self.config.get('dataset', '')
        self.data_path = './raw_data/' + self.dataset + '/'
        self.geo_file = self.config.get('geo_file', self.dataset)
        self.rel_file = self.config.get('rel_file', self.dataset)
        self.dyna_file = self.config.get('dyna_file', self.dataset)
        self.time_slice = self.config.get('time_slice',24)
        self.n_cluster = self.config.get('n_cluster',7)
        assert (24 % self.time_slice == 0)
        if not os.path.exists('./libcity/cache/MGFN_{}'.format(self.dataset)):
            os.mkdir('./libcity/cache/MGFN_{}'.format(self.dataset))
        self.multi_graph = None
        self.mob_patterns_path = './libcity/cache/MGFN_{}/{}_slice_{}_clusters_mob_patterns.npy'.format(self.dataset,self.time_slice,self.n_cluster)
        assert os.path.exists(self.data_path + self.geo_file + '.geo')
        assert os.path.exists(self.data_path + self.rel_file + '.rel')
        assert os.path.exists(self.data_path + self.dyna_file + '.dyna')
        super().__init__(config)
        self.construct_od_matrix()
        if os.path.exists(self.mob_patterns_path):
            self.mob_patterns = np.load(self.mob_patterns_path)
            self._logger.info("finish get Mobility Pattern")
        else:
            self.multi_graph = self.construct_multi_graph()
            self.mob_patterns, self.cluster_label = self.getPatternWithMGD(self.multi_graph)

    def get_data(self):
        return None, None, None

    def construct_od_matrix(self):
        assert self.representation_object == "region"
        self.od_label = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        for traj in self.traj_road:
            origin_region = list(self.road2region[self.road2region['origin_id'] == traj[0]]['destination_id'])[0]
            destination_region = list(self.road2region[self.road2region['origin_id'] == traj[-1]]['destination_id'])[0]
            self.od_label[origin_region][destination_region] += 1
        return self.od_label

    def construct_multi_graph(self):
        time_each_slice = 24 // self.time_slice
        self.multi_graph = np.zeros([self.time_slice, self.num_nodes, self.num_nodes])
        for traj, time_list in zip(self.traj_road, self.traj_time):
            origin_region = int(list(self.road2region[self.road2region['origin_id'] == traj[0]]['destination_id'])[0])
            destination_region = int(
                list(self.road2region[self.road2region['origin_id'] == traj[-1]]['destination_id'])[0])
            origin_hour = datetime.strptime(time_list[0], '%Y-%m-%d %H:%M:%S').hour
            self.multi_graph[origin_hour // time_each_slice][origin_region][destination_region] += 1
        return self.multi_graph

    def propertyFunc_var(self,adj_matrix):
        return adj_matrix.var()

    def propertyFunc_mean(self,adj_matrix):
        return adj_matrix.mean()

    def propertyFunc_std(self,adj_matrix):
        return adj_matrix.std()

    def propertyFunc_UnidirectionalIndex(self,adj_matrix):
        unidirectionalIndex = 0
        for i in range(len(adj_matrix)):
            for j in range(len(adj_matrix[0])):
                unidirectionalIndex = unidirectionalIndex + \
                                      abs(adj_matrix[i][j] - adj_matrix[j][i])
        return unidirectionalIndex

    def getPropertyArrayWithPropertyFunc(self,data_input, property_func):
        result = []
        for i in range(len(data_input)):
            result.append(property_func(data_input[i]))
        # -- standardlize
        return np.array(result)

    def getDistanceMatrixWithPropertyArray(self,data_x, property_array, isSigmoid=False):
        sampleNum = data_x.shape[0]
        disMatrix = np.zeros([sampleNum, sampleNum])
        for i in range(0, sampleNum):
            for j in range(0, sampleNum):
                if isSigmoid:
                    hour_i = i % 24
                    hour_j = j % 24
                    hour_dis = abs(hour_i - hour_j)
                    if hour_dis == 23:
                        hour_dis = 1
                    c = self.sigmoid(hour_dis / 24)
                else:
                    c = 1
                disMatrix[i][j] = c * abs(property_array[i] - property_array[j])
        disMatrix = (disMatrix - disMatrix.min()) / (disMatrix.max() - disMatrix.min())
        return disMatrix

    def getDistanceMatrixWithPropertyFunc(self,data_x, property_func, isSigmoid=False):
        property_array = self.getPropertyArrayWithPropertyFunc(data_x, property_func)
        disMatrix = self.getDistanceMatrixWithPropertyArray(data_x, property_array, isSigmoid=isSigmoid)
        return disMatrix

    def get_SSEncode2D(self,one_data, mean_data):
        result = []
        for i in range(len(one_data)):
            for j in range(len(one_data[0])):
                if one_data[i][j] > mean_data[i][j]:
                    result.append(1)
                else:
                    result.append(0)
        return np.array(result)

    def getDistanceMatrixWith_SSIndex(self,input_data, isSigmoid=True):
        sampleNum = len(input_data)
        input_data_mean = input_data.mean(axis=0)
        property_array = []
        for i in range(len(input_data)):
            property_array.append(self.get_SSEncode2D(input_data[i], input_data_mean))
        property_array = np.array(property_array)
        disMatrix = np.zeros([sampleNum, sampleNum])
        for i in range(0, sampleNum):
            for j in range(0, sampleNum):
                if isSigmoid:
                    hour_i = i % 24
                    hour_j = j % 24
                    sub_hour = abs(hour_i - hour_j)
                    if sub_hour == 23:
                        sub_hour = 1
                    c = self.sigmoid(sub_hour / 24)
                else:
                    c = 1
                sub_encode = abs(property_array[i] - property_array[j])
                disMatrix[i][j] = c * sub_encode.sum()
        disMatrix = (disMatrix - disMatrix.min()) / (disMatrix.max() - disMatrix.min())
        label_pred = self.getClusterLabelWithDisMatrix(disMatrix, display_dis_matrix=False)
        return disMatrix

    def sigmoid(self,z):
        return 1 / (1 + np.exp(-z))

    def Mobility_Graph_Distance(self,m_graphs):
        """
        :param m_graphs: (N, M, M).  N graphs, each graph has M nodes
        :return: (N, N). Distance matrix between every two graphs
        """
        # Mean
        isSigmoid = True
        mean_dis_matrix = self.getDistanceMatrixWithPropertyFunc(
            m_graphs, self.propertyFunc_mean, isSigmoid=isSigmoid)
        # Uniflow
        unidirIndex_dis_matrix = self.getDistanceMatrixWithPropertyFunc(
            m_graphs, self.propertyFunc_UnidirectionalIndex, isSigmoid=isSigmoid
        )
        # Var
        var_dis_matrix = self.getDistanceMatrixWithPropertyFunc(
            m_graphs, self.propertyFunc_var, isSigmoid=isSigmoid
        )
        # SS distance
        ss_dis_matrix = self.getDistanceMatrixWith_SSIndex(m_graphs, isSigmoid=isSigmoid)
        c_mean_dis = 1
        c_unidirIndex_dis = 1
        c_std_dis = 1
        c_ss_dis = 1
        disMatrix = (c_mean_dis * mean_dis_matrix) \
                    + (c_unidirIndex_dis * unidirIndex_dis_matrix) \
                    + (c_std_dis * var_dis_matrix) \
                    + (c_ss_dis * ss_dis_matrix)
        return disMatrix

    def getClusterLabelWithDisMatrix(self,dis_matrix, display_dis_matrix=False):
        n_clusters = self.n_cluster
        # # linkage: single, average, complete
        linkage = "complete"
        # ---
        # t1 = time.time()
        if display_dis_matrix:
            sns.heatmap(dis_matrix)
            plt.show()
        # ---
        estimator = AgglomerativeClustering(
            n_clusters=n_clusters, linkage=linkage, affinity="precomputed", )
        estimator.fit(dis_matrix)
        label_pred = estimator.labels_
        # print("The time consuming of clustering (known disMatrix)：", time.time() - t1)
        return label_pred

    def getPatternWithMGD(self,m_graphs):
        """
        :param m_graphs: (N, M, M).  N graphs, each graph has M nodes
        :return mob_patterns:
        :return cluster_label:
        """
        n_clusters = self.n_cluster
        linkage = "complete"
        disMatrix = self.Mobility_Graph_Distance(m_graphs)
        # -- Agglomerative Cluster
        estimator = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage, affinity="precomputed", )
        estimator.fit(disMatrix)
        label_pred = estimator.labels_
        cluster_label = label_pred
        # -- Generate Mobility Pattern
        patterns = []
        pbar = tqdm(range(n_clusters))
        pbar.set_description('get Pattern')
        for i in pbar:
            this_cluster_index_s = np.argwhere(label_pred == i).flatten()
            this_cluster_graph_s = m_graphs[this_cluster_index_s]
            patterns.append(this_cluster_graph_s.sum(axis=0))
        mob_patterns = np.array(patterns)
        np.save(self.mob_patterns_path,mob_patterns)
        self._logger.info("finish get Mobility Pattern")
        return mob_patterns, cluster_label

    def get_data_feature(self):
        """
        返回一个 dict，包含数据集的相关特征

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        return {"n_cluster":self.n_cluster,"od_matrix":self.od_label,"mob_patterns":self.mob_patterns,"num_nodes": self.num_nodes,"geo_to_ind": self.geo_to_ind, "ind_to_geo": self.ind_to_geo,
            "label":{"od_matrix_predict":self.od_label,"function_cluster":np.array(self.function)}}

