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
        self.construct_poi_simi()
        self.data_preprocess()

    def get_data(self):
        return None,None,None


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
            if content == '':
                print("i")
            contents.append(content)
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(contents)
        poi_attr = np.array(X.todense())
        for i in range(self.num_regions):
            for j in range(self.num_regions):
                poi_attr_i = poi_attr[i]
                poi_attr_j = poi_attr[j]
                self.poi_simi[i][j] = self.get_cos_similarity(poi_attr_i,poi_attr_j)
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
            "label":{"function_cluster":self.function}}


