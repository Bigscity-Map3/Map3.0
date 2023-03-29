import random
import json
import networkx as nx
import numpy as np
from deap import base, creator, tools
from logging import getLogger
from libcity.model.abstract_traffic_tradition_model import AbstractTraditionModel
import torch
import torch.nn as nn
import torch.optim as optim


class ZEMob(AbstractTraditionModel):

    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        # libcity/data/dataset/dataset_subclass/zemob_dataset.py
        self.M = torch.from_numpy(data_feature.get('M'))
        self.z_num = data_feature.get('z_num', None)
        self.e_num = data_feature.get('e_num', None)
        self.G_star = torch.from_numpy(data_feature.get('G_star', None))
        self._logger = getLogger()
        # libcity/config/model/region_representation/ZEMob.json
        self.output_dim = config.get('output_dim', 64)
        self.iter = config.get('max_epoch', 10)
        # Terminal configs
        self.model = config.get('model', '')
        self.dataset = config.get('dataset', '')
        self.exp_id = config.get('exp_id', None)
        self.txt_cache_file = './libcity/cache/{}/evaluate_cache/embedding_{}_{}_{}.txt'. \
            format(self.exp_id, self.model, self.dataset, self.output_dim)
        self.model_cache_file = './libcity/cache/{}/model_cache/embedding_{}_{}_{}.m'. \
            format(self.exp_id, self.model, self.dataset, self.output_dim)
        self.npy_cache_file = './libcity/cache/{}/evaluate_cache/embedding_{}_{}_{}.npy'. \
            format(self.exp_id, self.model, self.dataset, self.output_dim)
        # initialize embeddings
        self.zones_embedding = nn.Embedding(self.z_num, self.output_dim)
        self.events_embedding = nn.Embedding(self.e_num, self.output_dim)

    def loss_function(self, zones_embeddings, events_embeddings):
        sim_matrix = torch.matmul(zones_embeddings, events_embeddings.t())
        mse_matrix = ((self.M - sim_matrix) ** 2) * self.G_star
        mse = torch.sum(mse_matrix) / 2
        return mse

    def run(self, data=None):
        num_epochs = self.iter
        optimizer = optim.SGD(list(self.zones_embedding.parameters()) + list(self.events_embedding.parameters()),
                              lr=0.001)
        batch_size = 128
        zones_data = torch.arange(0, self.z_num)
        events_data = torch.arange(0, self.e_num)
        # zones_batches = torch.split(zones_data, batch_size)
        # events_batches = torch.split(events_data, batch_size)
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            zones_embeddings = self.zones_embedding(zones_data)
            events_embeddings = self.events_embedding(events_data)
            loss = self.loss_function(zones_embeddings, events_embeddings)
            loss.backward()
            optimizer.step()
            # for batch_idx, batch_data in enumerate(zip(zones_batches, events_batches)):
            #     optimizer.zero_grad()
            #     zones_embeddings = self.zones_embedding(batch_data[0])
            #     events_embeddings = self.events_embedding(batch_data[1])
            #     loss = self.loss_function(zones_embeddings, events_embeddings)
            #     loss.backward()
            #     optimizer.step()
            #     total_loss += loss.item()
            self._logger.info('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))
        np.save(self.npy_cache_file, self.zones_embedding.weight.data.numpy())
        self._logger.info('词向量和模型保存完成')
        self._logger.info('词向量维度：{}'.format(self.zones_embedding.embedding_dim) )
