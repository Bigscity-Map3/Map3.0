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
        self.all_result=[]

        self.model = config.get('model', '')
        self.dataset = config.get('dataset', '')
        self.exp_id = config.get('exp_id', None)
        self.result_path = './libcity/cache/{}/evaluate_cache/result_{}_{}.json'\
            .format(self.exp_id, self.model, self.dataset)



    def get_downstream_model(self,task):
        """
        according the config['evaluator'] to create the evaluator

        Args:
            config(ConfigParser): config

        Returns:
            AbstractEvaluator: the loaded evaluator
        """
        try:
            return getattr(importlib.import_module('libcity.evaluator.downstream_tasks'),
                        task)(self.config,self.data_feature)
        except AttributeError:
            raise AttributeError('evaluate task is not found')


    def collect(self, batch):
        pass

    def evaluate(self,representations,dataloader):
        for task in self.evaluate_tasks:
            model = self.get_downstream_model(task)
            result = model.run(representations,dataloader)
            result={task:result}
            self.all_result.update(result)
        
        self._logger.info('result {}'.format(self.all_result))
        self.save_result()

    def save_result(self, filename=None):
        json.dump(self.all_result,open(self.result_path,'w'))
        self._logger.info('result save in {}'.format(self.result_path))

    def clear(self):
        pass
