import os
import math
import pandas as pd
import numpy as np
from logging import getLogger
import torch
from torch import nn
from torch.nn import init
from sklearn.utils import shuffle
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


class FCClassifier(nn.Module):
    def __init__(self, embed_layer, input_size, output_size, hidden_size):
        super().__init__()

        self.embed_layer = embed_layer
        self.add_module('embed_layer', self.embed_layer)

        self.input_linear = nn.Linear(input_size, hidden_size)
        self.hidden_linear = nn.Linear(hidden_size, hidden_size)
        self.output_linear = nn.Linear(hidden_size, output_size)
        self.act = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.1)

        self.apply(weight_init)

    def forward(self, x):
        """
        :param x: input batch of location tokens, shape (batch_size)
        :return: prediction of the corresponding location categories, shape (batch_size, output_size)
        """
        h = self.dropout(self.embed_layer(x))  # (batch_size, input_size)
        h = self.dropout(self.act(self.input_linear(h)))
        h = self.dropout(self.act(self.hidden_linear(h)))
        out = self.output_linear(h)
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


class LocClassificationModel(AbstractModel):
    def __init__(self, config):
        super().__init__()
        self._logger = getLogger()
        self.input_size = config.get('input_size', 128)
        self.output_size = config.get('output_size', 1)
        self.hidden_size = config.get('hidden_size', 128)
        self.exp_id = config.get('exp_id', None)

        self.result_path = './libcity/cache/{}/evaluate_cache/loc_classification_{}_{}_{}.json'. \
            format(self.exp_id, self.input_size, self.output_size, self.hidden_size)

    def run(self, x, labels):
        fc_classifier = FCClassifier(self.input_size, self.output_size, self.hidden_size)
        predicts = fc_classifier(x)
        pres = predicts.argmax(-1)
        top_n_list = list(range(1, 11)) + [15, 20]
        score_series = cal_classify_metric(predicts, pres, labels, top_n_list)
        result = {'macro-f1': score_series['macro-f1'], 'acc@1': score_series['acc@1']}
        self._logger.info("finish Location Classification {macro-f1=" + str(score_series['macro-f1'])
                          + ",acc@1=" + str(score_series['acc@1']) + "}")
        return result

    def clear(self):
        pass

    def save_result(self):
        pass