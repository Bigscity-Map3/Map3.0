import numpy as np
from logging import getLogger
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score
import json

from libcity.evaluator.downstream_models.abstract_model import AbstractModel


class KmeansModel(AbstractModel):

    def __init__(self, config):
        self._logger = getLogger()
        self.n_clusters = config.get('n_clusters', 2)
        self.random_state = config.get('random_state',3)
        self.exp_id = config.get('exp_id', None)
        self.result_path = './libcity/cache/{}/evaluate_cache/kmeans_category_{}_{}.json'.\
            format(self.exp_id,self.n_clusters,self.random_state)

    def run(self,node_emb,label):
        self.n_clusters = np.unique(label).shape[0]
        kmeans = KMeans(n_clusters=self.n_clusters,random_state=self.random_state)
        self._logger.info("K-Means Cluster:n_clusters={},random_state={}".format(self.n_clusters,self.random_state))
        predict = kmeans.fit_predict(node_emb)
        np.save(self.result_path,predict)
        nmi = normalized_mutual_info_score(label, predict)
        ars = adjusted_rand_score(label, predict)
        result={'nmi':nmi,'ars':ars}
        self._logger.info("finish Kmeans cluster,result is {nmi="+str(nmi)+",ars="+str(ars)+"}")
        return result
    
    def clear(self):
        pass
    
    def save_result(self, result_token,save_path, filename=None):
        pass