import importlib
import math
import json
import numpy as np
import pandas as pd
from logging import getLogger
from libcity.evaluator.abstract_evaluator import AbstractEvaluator
from libcity.evaluator.downstream_models.loc_pred_model import *


class POIRepresentationEvaluator(AbstractEvaluator):

    def __init__(self, config, data_feature):
        self._logger = getLogger()
        self.config = config
        self.data_feature = data_feature
        self.model_name = self.config.get('downstream_model', '')
        self.device = self.config.get('device')

    def collect(self, batch):
        pass

    def evaluate_loc_pre(self):
        pre_model_seq2seq = self.config.get('pre_model_seq2seq', True)
        st_aux_embed_size = self.config.get('st_aux_embed_size', 16)
        st_num_slots = self.config.get('st_num_slots', 10)
        embed_layer = self.data_feature.get('embed_layer')
        num_loc = self.data_feature.get('num_loc')
        embed_size = self.config.get('embed_size', 128)
        hidden_size = embed_size * 4
        pre_len = self.config.get('pre_len', 3)
        task_epoch = self.config.get('task_epoch', 5)
        train_set = self.data_feature.get('train_set')
        test_set = self.data_feature.get('test_set')
        downstream_batch_size = self.data_feature.get('downstream_batch_size', 32)
        if self.model_name == 'erpp':
            pre_model = ErppLocPredictor(embed_layer, input_size=embed_size, lstm_hidden_size=hidden_size,
                                         fc_hidden_size=hidden_size, output_size=num_loc, num_layers=2,
                                         seq2seq=pre_model_seq2seq)
        elif self.model_name == 'stlstm':
            pre_model = StlstmLocPredictor(embed_layer, num_slots=st_num_slots, aux_embed_size=st_aux_embed_size,
                                           time_thres=10800, dist_thres=0.1,
                                           input_size=embed_size, lstm_hidden_size=hidden_size,
                                           fc_hidden_size=hidden_size, output_size=num_loc, num_layers=2,
                                           seq2seq=pre_model_seq2seq)
        elif self.model_name == 'strnn':
            st_time_window = self.config.get('st_time_window', 7200)
            st_dist_window = self.config.get('st_dist_window', 0.1)
            st_inter_size = self.config.get('st_inter_size', 4)
            pre_model = StrnnLocPredictor(embed_layer, num_slots=st_num_slots,
                                          time_window=st_time_window, dist_window=st_dist_window,
                                          input_size=embed_size, hidden_size=hidden_size,
                                          inter_size=st_inter_size, output_size=num_loc)
        elif self.model_name == 'rnn':
            pre_model = RnnLocPredictor(embed_layer, input_size=embed_size, rnn_hidden_size=hidden_size,
                                        fc_hidden_size=hidden_size,
                                        output_size=num_loc, num_layers=1, seq2seq=pre_model_seq2seq)
        else:
            pre_model = Seq2SeqLocPredictor(embed_layer, input_size=embed_size, hidden_size=hidden_size,
                                            output_size=num_loc, num_layers=2)
        loc_prediction(train_set, test_set, num_loc, pre_model, pre_len=pre_len,
                       num_epoch=task_epoch, batch_size=downstream_batch_size, device=self.device)

    def evaluate(self):
        self._logger.info('Start evaluating ...')
        task_name = self.config.get('downstream_task', 'loc_pre')
        self._logger.info('Downstream Model: {}'.format(self.model_name))
        self._logger.info('Downstream Task: {}'.format(task_name))
        if task_name == 'loc_pre':
            self.evaluate_loc_pre()

    def save_result(self, save_path, filename=None):
        pass

    def clear(self):
        pass
