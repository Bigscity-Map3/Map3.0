import random
import json
import numpy as np

import torch
from torch import optim
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn

from logging import getLogger
from libcity.model.abstract_traffic_tradition_model import AbstractTraditionModel


class ZEMob(AbstractTraditionModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self.ppmi_matrix = data_feature.get('ppmi_matrix')
        self.G_matrix = data_feature.get('G_matrix')
        self.region_num = data_feature.get('region_num')
        self.mobility_event_num = data_feature.get('mobility_event_num')
        self._logger = getLogger()

        self.output_dim = config.get('output_dim', 64)
        self.iter = config.get('max_epoch', 1000)
        self.batch_size = config.get('batch_size', 64)
        self.lr = config.get('learning_rate', 0.001)

        self.model = config.get('model', '')
        self.dataset = config.get('dataset', '')
        self.exp_id = config.get('exp_id', None)
        self.txt_cache_file = './libcity/cache/{}/evaluate_cache/embedding_{}_{}_{}.txt'. \
            format(self.exp_id, self.model, self.dataset, self.output_dim)
        self.model_cache_file = './libcity/cache/{}/model_cache/embedding_{}_{}_{}.m'. \
            format(self.exp_id, self.model, self.dataset, self.output_dim)
        self.npy_cache_file = './libcity/cache/{}/evaluate_cache/embedding_{}_{}_{}.npy'. \
            format(self.exp_id, self.model, self.dataset, self.output_dim)

        self.region_list = list(range(self.region_num))
        self.mobility_event_list = list(range(self.mobility_event_num))
        # self.train_data = ZEMobDataSet(self.region_list)
        # self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=0)
        self.model = ZEMobModel(self.region_num, self.mobility_event_num, self.output_dim, self.ppmi_matrix)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)

    def run(self, data=None):
        for epoch in range(1, self.iter + 1):
            self.train(epoch)

        with open(self.txt_cache_file, 'w', encoding='UTF-8') as f:
            f.write('{} {}\n'.format(self.region_num, self.output_dim))
            embeddings = self.model.zone_embedding.weight.data.numpy()
            for i in range(self.region_num):
                embedding = embeddings[i]
                embedding = ' '.join(map((lambda x: str(x)), embedding))
                f.write('{}\n'.format(embedding))

        self._logger.info('词向量和模型保存完成')
        self._logger.info('词向量维度：(' + str(self.region_num) + ',' + str(self.output_dim) + ')')

    def train(self, epoch):
        # train_loss = 0
        # for data in self.train_loader:
        #     self.optimizer.zero_grad()
        #     loss = self.model.forward(data)
        #     loss.backward()
        #     self.optimizer.step()
        #     train_loss += loss.item() * data.size(0)
        # train_loss = train_loss / len(self.train_loader.dataset)
        # self._logger.info('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))

        train_loss = 0
        self.optimizer.zero_grad()
        loss = self.model.forward()
        loss.backward()
        self.optimizer.step()
        train_loss = (train_loss + loss.item()) / self.region_num
        self._logger.info('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))

class ZEMobModel(nn.Module):
    def __init__(self, zone_num, mobility_event_num, embedding_dim, ppmi_matrix):
        super(ZEMobModel, self).__init__()
        self.zone_num = zone_num
        self.mobility_event_num = mobility_event_num
        self.embedding_dim = embedding_dim
        self.ppmi_matrix = torch.tensor(ppmi_matrix)

        self.zone_embedding = nn.Embedding(self.zone_num, self.embedding_dim)
        self.event_embedding = nn.Embedding(self.mobility_event_num, self.embedding_dim)

        initrange = 0.5 / self.embedding_dim
        self.zone_embedding.weight.data.uniform_(-initrange, initrange)
        self.event_embedding.weight.data.uniform_(-initrange, initrange)

        self.all_events = torch.arange(mobility_event_num)
        self.all_zones = torch.arange(zone_num)

    def forward(self):
        # batch_zone = self.zone_embedding(batch_zones)
        # batch_event = self.event_embedding(self.all_events)
        # return torch.sum(torch.pow(torch.sub(self.ppmi_matrix[batch_zones], torch.mm(batch_zone, batch_event.t())), 2))
        return torch.sum(torch.pow(torch.sub(
            self.ppmi_matrix, torch.mm(self.zone_embedding(self.all_zones), self.event_embedding(self.all_events).t())
        ), 2)) / 2


# class ZEMobDataSet(Dataset):
#     def __init__(self, zones):
#         self.zones = zones
#
#     def __len__(self):
#         return len(self.zones)
#
#     def __getitem__(self, idx):
#         zone = self.zones[idx]
#         return zone
