import os
from logging import getLogger
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy
import numpy as np
from tqdm import tqdm

from libcity.data.dataset.abstract_dataset import AbstractDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from libcity.data.preprocess import preprocess_all, cache_dir

class MVUREDataset(AbstractDataset):

    def __init__(self, config):
        self.config = config
        preprocess_all(config)
        self._logger = getLogger()
        self.num_nodes = 0
        self.dataset = self.config.get('dataset', '')
        self.data_path = './raw_data/' + self.dataset + '/'
        if not os.path.exists('./libcity/cache/MVURE_{}'.format(self.dataset)):
            os.mkdir('./libcity/cache/MVURE_{}'.format(self.dataset))
        self.in_flow_adj_path = './libcity/cache/MVURE_{}/in_flow_adj.npy'.format(self.dataset)
        self.out_flow_adj_path = './libcity/cache/MVURE_{}/out_flow_adj.npy'.format(self.dataset)
        self.poi_simi_path = './libcity/cache/MVURE_{}/poi_simi.npy'.format(self.dataset)
        self.od_label_path = os.path.join(cache_dir, self.dataset, 'traj_region_train_od.npy')
        self.mob_adj = np.load(self.od_label_path)
        self.num_regions = self.mob_adj.shape[0]
        self.num_nodes = self.num_regions
        self.construct_flow_adj()
        self.construct_poi_simi()
        self.data_preprocess()

    def get_data(self):
        return None,None,None

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
        self.od_label = self.mob_adj
        # inflow_adj = numpy.zeros([self.num_regions,self.num_regions])
        # outflow_adj = numpy.zeros([self.num_regions,self.num_regions])
        # for i in tqdm(range(self.num_regions)):
        #
        #     for j in range(self.num_regions):
        #         p_i_in = self.od_label[:,i]
        #         p_i_out = self.od_label[i,:]
        #         p_j_in = self.od_label[:,j]
        #         p_j_out = self.od_label[j,:]
        #         if self.od_label[:i].sum() > 0:
        #             p_i_in = self.od_label[:, i] / self.od_label[:i].sum()
        #         if self.od_label[i,:].sum() > 0:
        #             p_i_out = self.od_label[i,:]/self.od_label[i,:].sum()
        #         if self.od_label[:,j].sum() > 0:
        #             p_j_in = self.od_label[:,j]/self.od_label[:,j].sum()
        #         if self.od_label[j,:].sum() > 0:
        #             p_j_out = self.od_label[j,:]/self.od_label[j,:].sum()
        #         inflow_adj[i][j] = self.get_cos_similarity(p_i_in, p_j_in)
        #         outflow_adj[i][j] = self.get_cos_similarity(p_i_out, p_j_out)
        # self.inflow_adj = inflow_adj
        # self.outflow_adj = outflow_adj
        self.od_label = self.od_label + np.eye(self.num_nodes)
        row_norms = np.linalg.norm(self.od_label,axis=1,keepdims=True)
        in_flow_vector = self.od_label/row_norms
        self._logger.info("calculate in flow cosine similarity")
        self.inflow_adj = cosine_similarity(in_flow_vector)
        self._logger.info("shape = " + str(self.inflow_adj.shape))
        od_T = self.od_label.transpose()
        row_norms = np.linalg.norm(od_T,axis=1,keepdims=True)
        out_flow_vector = od_T/row_norms
        self._logger.info("calculate out flow cosine similarity")
        self.outflow_adj = cosine_similarity(out_flow_vector)
        self._logger.info("shape = "+str(self.outflow_adj.shape))
        np.save(self.in_flow_adj_path, self.inflow_adj)
        np.save(self.out_flow_adj_path, self.outflow_adj)
        self._logger.info("finish construct flow graph")

    def construct_poi_simi(self):
        if os.path.exists(self.poi_simi_path):
            self.poi_simi = np.load(self.poi_simi_path)
            self._logger.info("finish construct poi_simi")
            return
        self.poi_simi = np.zeros([self.num_regions,self.num_regions])
        poi_simi_file = pd.read_csv(self.data_path+"regionmap_"+self.dataset+"/"+"regionmap_"+self.dataset+".sem")
        for i in range(len(poi_simi_file)):
            origin_id = poi_simi_file.loc[i, "origin_id"]
            destination_id = poi_simi_file.loc[i, "destination_id"]
            self.poi_simi[origin_id][destination_id] = poi_simi_file.loc[i,"semantic_weight"]
        np.save(self.poi_simi_path,self.poi_simi)
        self._logger.info("finish construct poi_simi")

    def data_preprocess(self):
        n, _ = self.mob_adj.shape
        self.mob_adj = self.mob_adj/np.mean(self.mob_adj,axis=(0,1))
        self.feature = np.random.uniform(-1, 1, size=(self.num_nodes, 250))
        self.feature = self.feature[np.newaxis]


    def get_data_feature(self):
        """
        返回一个 dict，包含数据集的相关特征

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        return {"mob_adj":self.mob_adj,"s_adj_sp":self.inflow_adj,"t_adj_sp":self.outflow_adj,
                "poi_adj":self.poi_simi,"feature":self.feature,"num_nodes": self.num_nodes}


