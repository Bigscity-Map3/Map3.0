from logging import getLogger

import numpy as np

from libcity.evaluator.abstract_evaluator import AbstractEvaluator
from libcity.evaluator.downstream_models.gbrt_model import GBRT_Predictor
#GMEL下游训练GBRT模型用于od预测
class GmelEvaluator(AbstractEvaluator):
    def __init__(self, config, data_feature):
        self._logger = getLogger()
        self.config=config
        self.all_result=[]
        self.model = config.get('model', '')
        self.dataset = config.get('dataset', '')
        self.exp_id = config.get('exp_id', None)
        self.output_dim = config.get('output_dim', 96)
        self.data_feature = data_feature
        self.data_path = './raw_data/' + self.dataset + '/'
        self.data_feature = data_feature
        self.org_embedding_cache_file = './libcity/cache/{}/evaluate_cache/org_embedding_{}_{}_{}.npy'. \
            format(self.exp_id, self.model, self.dataset, self.output_dim)
        self.dst_embedding_cache_file = './libcity/cache/{}/evaluate_cache/dst_embedding_{}_{}_{}.npy'. \
            format(self.exp_id, self.model, self.dataset, self.output_dim)
        self.result_path = './libcity/cache/{}/evaluate_cache/result_{}_{}_{}.json'\
            .format(self.exp_id, self.model, self.dataset, self.output_dim)
    def collect(self, batch):
        pass

    def evaluate(self):
        src_emb = np.load(self.org_embedding_cache_file)
        dst_emb = np.load(self.dst_embedding_cache_file)
        gbrt = GBRT_Predictor(self.config)
        gbrt.run(src_emb,dst_emb,self.data_feature["distm"],self.data_feature["train"],self.data_feature["valid"],self.data_feature["test"])
        return
