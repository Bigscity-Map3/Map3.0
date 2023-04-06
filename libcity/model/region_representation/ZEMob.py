import random
import json
import networkx as nx
import numpy as np
from deap import base, creator, tools
from logging import getLogger

from torch.utils.data import Dataset

from libcity.model.abstract_traffic_tradition_model import AbstractTraditionModel
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau


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
        self.output_dim = config.get('output_dim', 128)
        self.iter = config.get('max_epoch', 1000)
        # Terminal configs
        self.model = config.get('model', '')
        self.dataset = config.get('dataset', '')
        self.exp_id = config.get('exp_id', None)
        self.lr =self.config.get('learning_rate', 0.06)
        self.txt_cache_file = './libcity/cache/{}/evaluate_cache/embedding_{}_{}_{}.txt'. \
            format(self.exp_id, self.model, self.dataset, self.output_dim)
        self.model_cache_file = './libcity/cache/{}/model_cache/embedding_{}_{}_{}.m'. \
            format(self.exp_id, self.model, self.dataset, self.output_dim)
        self.npy_cache_file = './libcity/cache/{}/evaluate_cache/embedding_{}_{}_{}.npy'. \
            format(self.exp_id, self.model, self.dataset, self.output_dim)
        # initialize embeddings
        self.zones_embedding = nn.Embedding(self.z_num, self.output_dim)
        self.events_embedding = nn.Embedding(self.e_num, self.output_dim)
        initrange = 0.5 / self.output_dim
        self.zones_embedding.weight.data.uniform_(-initrange, initrange)
        self.events_embedding.weight.data.uniform_(-initrange, initrange)

    def loss_function(self, zones_embeddings, events_embeddings):
        # self._logger.info("caculating loss of zones {} and events {}".format(zones_embeddings.shape,events_embeddings.shape))
        sim_matrix = torch.matmul(zones_embeddings, events_embeddings.t())
        mse_matrix = torch.pow( torch.sub(self.M , sim_matrix), 2) * self.G_star
        # mse_matrix = ((self.M - sim_matrix) ** 2)
        mse = torch.sum(mse_matrix) / 2
        return mse

    def run(self, data=None):
        num_epochs = self.iter
        optimizer = optim.SGD(list(self.zones_embedding.parameters()) + list(self.events_embedding.parameters()),
                              lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.1, verbose=True)
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
            scheduler.step(loss)
            # for batch_idx, batch_data in enumerate(zip(zones_batches, events_batches)):
            #     optimizer.zero_grad()
            #     zones_embeddings = self.zones_embedding(batch_data[0])
            #     events_embeddings = self.events_embedding(batch_data[1])
            #     loss = self.loss_function(zones_embeddings, events_embeddings)
            #     loss.backward()
            #     optimizer.step()
            #     total_loss += loss.item()
            if epoch % 50 == 0 or epoch<10:
                self._logger.info('Epoch [{}/{}], Loss: {:.4f}'.format(epoch + 1, num_epochs, loss.item()))
        np.save(self.npy_cache_file, self.zones_embedding.weight.data.numpy())
        self._logger.info('词向量和模型保存完成')
        self._logger.info('词向量维度：{}'.format(self.zones_embedding.embedding_dim))


# class ZEMob_Model(torch.nn):
#     def __init__(self,zone_num,event_num,M,G_star,output_dim):
#         self.output_dim = output_dim
#         self.M = M
#         self.G_star = G_star
#         self.event_num = event_num
#         self.zone_num = zone_num
#         self.zone_embedding=nn.Embedding(zone_num,output_dim)
#         self.event_embedding=nn.Embedding(zone_num,output_dim)
#         nn.init.normal_(self.zone_embedding.weight)
#         nn.init.normal_(self.event_embedding.weight)
#     def forward(self, batch):
#         batch_gpu = batch.to(self.device)
#         batch_zone = self.zone_embedding(batch_gpu)
#         batch_event = self.event_embedding(self.all_events)
#
#         batch_ppmi = self.ppmi_matrix[batch].to(self.device)
#         batch_G = self.G_matrix[batch].to(self.device)
#
#         return torch.sum(torch.pow(torch.sub(batch_ppmi, torch.mm(batch_zone, batch_event.t())), 2) * batch_G) / 2
#
#
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
