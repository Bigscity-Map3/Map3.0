import os
import math
from itertools import zip_longest
import pandas as pd
import numpy as np
from logging import getLogger
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn import init
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from libcity.evaluator.downstream_models.abstract_model import AbstractModel


def weight_init(m):
    """
    Usage:
        model = Model()
        model.apply(weight_init)
    """
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.Embedding):
        embed_size = m.weight.size(-1)
        if embed_size > 0:
            init_range = 0.5/m.weight.size(-1)
            init.uniform_(m.weight.data, -init_range, init_range)


class LstmLocPredictor(nn.Module):
    def __init__(self, loc_embed_layer, loc_embed_size, num_loc, hidden_size):
        super().__init__()
        self.num_loc = num_loc

        self.loc_embed_layer = loc_embed_layer
        self.add_module('loc_embed_layer', self.loc_embed_layer)

        # self.rnn = nn.LSTM(loc_embed_size, hidden_size, num_layers=1, batch_first=True)
        self.rnn = nn.GRU(loc_embed_size, hidden_size, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(0.1)
        self.out_linear = nn.Sequential(nn.Tanh(), nn.Linear(hidden_size, num_loc))

        self.apply(weight_init)

    def forward(self, current_poi_seq):
        """
        :param current_poi_seq: current poi sequence, shape (batch_size, max_current_seq_len)
        :return: prediction of the next visited location, shape (batch_size, num_loc)
        """
        valid_len = (current_poi_seq != self.num_loc).long().sum(-1)  # (batch_size)

        poi_embed = self.dropout(self.loc_embed_layer(current_poi_seq))  # (batch_size, max_current_seq_len, loc_embed_size)
        packed_poi_embed = pack_padded_sequence(poi_embed, valid_len, batch_first=True, enforce_sorted=False)
        # _, (hidden, _) = self.rnn(packed_poi_embed)
        _, hidden = self.rnn(packed_poi_embed)
        hidden = hidden.squeeze(0)  # (batch_size, hidden_size)

        out = self.out_linear(hidden)
        return out


def top_n_accuracy(truths, preds, n):
    best_n = np.argsort(preds, axis=1)[:, -n:]
    successes = 0
    for i, truth in enumerate(truths):
        if truth in best_n[i, :]:
            successes += 1
    return float(successes) / truths.shape[0]


def cal_classify_metric(pre_dists, pres, labels, top_n_list):
    precision, recall, f1 = precision_score(labels, pres, average='macro'), \
                            recall_score(labels, pres, average='macro'), \
                            f1_score(labels, pres, average='macro')
    if pre_dists is not None:
        top_n_acc = [top_n_accuracy(labels, pre_dists, n) for n in top_n_list]
    else:
        top_n_acc = [accuracy_score(labels, pres)] + [-1.0 for _ in range(len(top_n_list)-1)]
    score_series = pd.Series([precision, recall, f1] + top_n_acc,
                             index=['macro-pre', 'macro-rec', 'macro-f1'] + ['acc@{}'.format(n) for n in top_n_list])
    return score_series


class NextLocPreModel(AbstractModel):
    def __init__(self, config):
        super().__init__()
        self._logger = getLogger()
        self.loc_embed_size = config.get('loc_embed_size', 128)
        self.num_loc = config.get('num_loc', 2)
        self.hidden_size = config.get('hidden_size', 128)
        self.exp_id = config.get('exp_id', None)

        self.result_path = './libcity/cache/{}/evaluate_cache/nextloc_pre_{}_{}_{}.json'. \
            format(self.exp_id, self.loc_embed_size, self.num_loc, self.hidden_size)

    def run(self, current_poi_seq, labels):

        lstm_loc_predictor = LstmLocPredictor(self.loc_embed_size, self.num_loc, self.hidden_size)
        predicts = lstm_loc_predictor(current_poi_seq)
        pres = predicts.argmax(-1)
        top_n_list = list(range(1, 11)) + [15, 20]
        score_series = cal_classify_metric(predicts, pres, labels, top_n_list)
        result = {'macro-f1': score_series['macro-f1'], 'acc@1': score_series['acc@1']}
        self._logger.info("finish Next Location Prediction {macro-f1=" + str(score_series['macro-f1'])
                          + ",acc@1=" + str(score_series['acc@1']) + "}")
        return result

    def clear(self):
        pass

    def save_result(self):
        pass