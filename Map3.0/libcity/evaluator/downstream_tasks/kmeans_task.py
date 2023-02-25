from libcity.evaluator.downstream_tasks.abstract_task import AbstractTask
from logging import getLogger
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score
import json

class KmeansTask(AbstractTask):

    def __init__(self, config):
        self._logger = getLogger()
        self.n_clusters = config.get('n_clusters', 2)
        self.random_state = config.get('random_state',3)
        self.result_path = './libcity/cache/{}/evaluate_cache/kmeans_category_{}_{}_{}.json'.\
            format(self.exp_id,self.n_clusters,self.random_state)

    def run(self,node_emb,label):

        kmeans = KMeans(self.n_clusters,self.random_state)
        predict = kmeans.fit_predict(node_emb)
        self.save_result(predict,self.result_path)
        nmi = normalized_mutual_info_score(label, predict)
        ars = adjusted_rand_score(label, predict)
        result={'nmi':nmi,'ars':ars}    
        return result
    
    def clear(self):
        pass
    
    def save_result(self, result_token,save_path, filename=None):
        json.dump(result_token, open(save_path, 'w'))
        self._logger.info('Kmeans category is saved at {}'.format(save_path))