import math
import json
import numpy as np
import pandas as pd
from logging import getLogger
from sklearn.cluster import KMeans
import importlib
from libcity.evaluator.abstract_evaluator import AbstractEvaluator


class RoadRepresentationEvaluator(AbstractEvaluator):

    def __init__(self, config,data_feature):
        self._logger = getLogger()
        self.config=config
        self.data_feature=data_feature

        self.evaluate_tasks=self.config.get('evaluate_tasks',[])

        self.model = config.get('model', '')
        self.dataset = config.get('dataset', '')
        self.exp_id = config.get('exp_id', None)
        self.data_path = './raw_data/' + self.dataset + '/'
        self.geo_file = config.get('geo_file', self.dataset)
        self.output_dim = config.get('output_dim', 32)
        self.embedding_path = './libcity/cache/{}/evaluate_cache/embedding_{}_{}_{}.npy'\
            .format(self.exp_id, self.model, self.dataset, self.output_dim)



    def get_downstream_model(self,task,represetations):
        """
        according the config['evaluator'] to create the evaluator

        Args:
            config(ConfigParser): config

        Returns:
            AbstractEvaluator: the loaded evaluator
        """
        try:
            return getattr(importlib.import_module('libcity.evaluator.downstream_tasks'),
                        task)(self.config,self.data_feature,represetations)
        except AttributeError:
            raise AttributeError('evaluate task is not found')


    def collect(self, batch):
        pass

    def evaluate(self,representations,dataloader):
        pass

    def save_result(self, save_path, filename=None):
        pass

    def clear(self):
        pass
