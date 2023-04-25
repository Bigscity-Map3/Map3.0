import logging
from datetime import datetime
import os

import numpy
import numpy as np
from libcity.data.dataset import TrafficRepresentationDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
class MVUREDataset(TrafficRepresentationDataset):

    def __init__(self, config):
        self.config = config
        self.dataset = self.config.get('dataset', '')
        self.data_path = './raw_data/' + self.dataset + '/'
        self.geo_file = self.config.get('geo_file', self.dataset)
        self.rel_file = self.config.get('rel_file', self.dataset)
        self.dyna_file = self.config.get('dyna_file', self.dataset)
        if not os.path.exists('./libcity/cache/MVURE_{}'.format(self.dataset)):
            os.mkdir('./libcity/cache/MVURE_{}'.format(self.dataset))
        self.remove_node_type = self.config.get('remove_node_type', 'od')
        self.in_flow_adj_path = './libcity/cache/MVURE_{}/in_flow_adj_{}.npy'.format(self.dataset,self.remove_node_type)
        self.out_flow_adj_path = './libcity/cache/MVURE_{}/out_flow_adj_{}.npy'.format(self.dataset,self.remove_node_type)
        self.poi_simi_path = './libcity/cache/MVURE_{}/poi_simi_{}.npy'.format(self.dataset,self.remove_node_type)
        self.od_label_path = './libcity/cache/MVURE_{}/od_label_{}.npy'.format(self.dataset,self.remove_node_type)
        assert os.path.exists(self.data_path + self.geo_file + '.geo')
        assert os.path.exists(self.data_path + self.rel_file + '.rel')
        assert os.path.exists(self.data_path + self.dyna_file + '.dyna')
        super().__init__(config)
        self.construct_od_matrix()
        self.construct_flow_adj()
        self.construct_poi_simi()
        self.data_preprocess()
    def get_data(self):
        return None,None,None

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




    def get_cos_similarity(self,v1,v2):
        if np.linalg.norm(v1) == 0:
            return 0
        if np.linalg.norm(v2) == 0:
            return 0
        num = float(np.dot(v1, v2))
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)
        return num / denom

    def construct_flow_adj(self):
        if os.path.exists(self.in_flow_adj_path) and os.path.exists(self.out_flow_adj_path):
            self.inflow_adj = np.load(self.in_flow_adj_path)
            self.outflow_adj = np.load(self.out_flow_adj_path)
            self._logger.info("finish construct flow graph")
            return
        inflow_adj = numpy.zeros([self.num_regions,self.num_regions])
        outflow_adj = numpy.zeros([self.num_regions,self.num_regions])
        for i in range(self.num_regions):
            for j in range(self.num_regions):
                p_i_in = numpy.zeros([self.num_regions])
                p_i_out = numpy.zeros([self.num_regions])
                p_j_in = numpy.zeros([self.num_regions])
                p_j_out = numpy.zeros(self.num_regions)
                sum_i_in = 0
                sum_i_out = 0
                sum_j_in = 0
                sum_j_out = 0
                for k in range(self.num_regions):
                    p_i_in[k] = self.od_label[k][i]
                    p_i_out[k] = self.od_label[i][k]
                    sum_i_in = sum_i_in + p_i_in[k]
                    sum_i_out = sum_i_out + p_i_out[k]
                    p_j_in[k] = self.od_label[k][j]
                    p_j_out[k] = self.od_label[j][k]
                    sum_j_in = sum_j_in + p_j_in[k]
                    sum_j_out = sum_j_out + p_j_out[k]
                if sum_i_in > 0:
                    p_i_in = p_i_in / sum_i_in
                if sum_j_in > 0:
                    p_j_in = p_j_in / sum_j_in
                if sum_i_out > 0:
                    p_i_out = p_i_out / sum_i_out
                if sum_j_out > 0:
                    p_j_out = p_j_out / sum_j_out
                inflow_adj[i][j] = self.get_cos_similarity(p_i_in,p_j_in)
                outflow_adj[i][j] = self.get_cos_similarity(p_i_out,p_j_out)
        self.inflow_adj = inflow_adj
        self.outflow_adj = outflow_adj
        np.save(self.in_flow_adj_path,self.inflow_adj)
        np.save(self.out_flow_adj_path,self.outflow_adj)
        self._logger.info("finish construct flow graph")

    def construct_poi_simi(self):
        if os.path.exists(self.poi_simi_path):
            self.poi_simi = np.load(self.poi_simi_path)
            self._logger.info("finish construct poi_simi")
            return
        self.poi_simi = np.zeros([self.num_regions,self.num_regions])
        contents = []
        for i in range(self.num_regions):
            content = ''
            poi_ids = list(self.region2poi[self.region2poi["origin_id"]==self.ind_to_geo[i]]["destination_id"])
            for poi_id in poi_ids:
                content=content+ (self.geofile.loc[poi_id,"function"])+' '
            contents.append(content)
        vectorizer = TfidfVectorizer()
        tf_idf_transformer = TfidfTransformer()
        X = vectorizer.fit_transform(contents)
        tf_idf = tf_idf_transformer.fit_transform(X)
        poi_attr = tf_idf.toarray()
        for i in range(self.num_regions):
            for j in range(self.num_regions):
                self.poi_simi[i][j] = self.get_cos_similarity(poi_attr[i],poi_attr[j])
        np.save(self.poi_simi_path,self.poi_simi)
        self._logger.info("finish construct poi_simi")

    def data_preprocess(self):
        self.mob_adj = np.zeros([self.num_nodes,self.num_nodes])
        self.mob_adj = np.copy(self.od_label)
        n, _ = self.mob_adj.shape
        self.mob_adj = self.mob_adj/np.mean(self.mob_adj,axis=(0,1))
        #TODO:k可能需要修改
        k = self.num_nodes//5
        self.inflow_adj_sp = np.copy(self.inflow_adj)
        for i in range(n):
            self.inflow_adj_sp[np.argsort( self.inflow_adj_sp[:, i])[:-k], i] = 0
            self.inflow_adj_sp[i, np.argsort(self.inflow_adj_sp[i, :])[:-k]] = 0
        self.outflow_adj_sp = np.copy(self.outflow_adj)
        for i in range(n):
            self.outflow_adj_sp[np.argsort(self.outflow_adj_sp[:, i])[:-k], i] = 0
            self.outflow_adj_sp[i, np.argsort(self.outflow_adj_sp[i, :])[:-k]] = 0

        k= self.num_nodes//5
        self.poi_adj_sp = np.copy(self.poi_simi)
        for i in range(n):
            self.poi_adj_sp[np.argsort(self.poi_adj_sp[:, i])[:-k], i] = 0
            self.poi_adj_sp[i, np.argsort(self.poi_adj_sp[i, :])[:-k]] = 0
        self.feature = np.random.uniform(-1, 1, size=(self.num_nodes, 250))
        self.feature = self.feature[np.newaxis]


    def get_data_feature(self):
        """
        返回一个 dict，包含数据集的相关特征

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        return {"mob_adj":self.mob_adj,"s_adj_sp":self.inflow_adj_sp,"t_adj_sp":self.outflow_adj_sp,
                "poi_adj":self.poi_simi,"poi_adj_sp":self.poi_adj_sp,"feature":self.feature
            ,"num_nodes": self.num_nodes,"geo_to_ind": self.geo_to_ind, "ind_to_geo": self.ind_to_geo,
            "label":{"od_matrix_predict":self.od_label,"function_cluster":self.function}}


