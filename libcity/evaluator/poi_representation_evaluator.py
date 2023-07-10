import math
import json
import numpy as np
from logging import getLogger
import importlib

from tqdm import tqdm
from sklearn.utils import shuffle
from libcity.evaluator.abstract_evaluator import AbstractEvaluator



class PoiRepresentationEvaluator(AbstractEvaluator):

    def __init__(self, config, data_feature):
        self._logger = getLogger()
        self.config = config
        self.evaluate_tasks = self.config.get('evaluate_tasks', ["loc_classification"])
        self.evaluate_model = self.config.get('evaluate_model', ["LocClassificationModel"])
        self.all_result = []
        self.model = config.get('model', '')
        self.dataset = config.get('dataset', '')
        self.exp_id = config.get('exp_id', None)
        self.output_dim = config.get('output_dim', 128)
        self.data_path = './raw_data/' + self.dataset + '/'
        self.data_feature = data_feature
        self.embedding_path = './libcity/cache/{}/evaluate_cache/embedding_{}_{}_{}.npy' \
            .format(self.exp_id, self.model, self.dataset, self.output_dim)
        self.result_path = './libcity/cache/{}/evaluate_cache/result_{}_{}_{}.json' \
            .format(self.exp_id, self.model, self.dataset, self.output_dim)

    def get_downstream_model(self, model):
        try:
            return getattr(importlib.import_module('libcity.evaluator.downstream_models'), model)(self.config)
        except AttributeError:
            raise AttributeError('evaluate model is not found')

    def collect(self, batch):
        pass

    def evaluate(self):
        poi_emb = np.load(self.embedding_path)  # (N, F)
        for task, model in zip(self.evaluate_tasks, self.evaluate_model):
            downstream_model = self.get_downstream_model(model)
            label = self.data_feature["label"][task]
            x = poi_emb
            if task == "loc_classification":
                result = downstream_model.run(x, label)
                result = {task: result}
            self.all_result.append(result)

        self._logger.info('result {}'.format(self.all_result))
        self.save_result(self.result_path)




    def save_result(self, save_path,filename=None):
        json.dump(self.all_result, open(self.result_path, 'w'))
        self._logger.info('result save in {}'.format(self.result_path))

    def clear(self):
        pass
