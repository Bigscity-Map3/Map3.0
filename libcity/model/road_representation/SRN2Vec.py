from logging import getLogger

from tqdm import tqdm

from libcity.model.abstract_replearning_model import AbstractReprLearningModel
import numpy as np
import torch
import torch.nn as nn
import time


class SRN2Vec(AbstractReprLearningModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self.device = config.get('device')
        self.dataloader = data_feature.get('dataloader')
        self.num_nodes = data_feature.get("num_nodes")
        self._logger = getLogger()
        self.output_dim = config.get('output_dim', 128)
        self.iter = config.get('max_epoch', 10)
        self.model = config.get('model', '')
        self.exp_id = config.get('exp_id', None)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.dataset = config.get('dataset', '')
        self.txt_cache_file = './libcity/cache/{}/evaluate_cache/road_embedding_{}_{}_{}.txt'. \
            format(self.exp_id, self.model, self.dataset, self.output_dim)
        self.model_cache_file = './libcity/cache/{}/model_cache/embedding_{}_{}_{}.m'. \
            format(self.exp_id, self.model, self.dataset, self.output_dim)
        self.npy_cache_file = './libcity/cache/{}/evaluate_cache/road_embedding_{}_{}_{}.npy'. \
            format(self.exp_id, self.model, self.dataset, self.output_dim)
        self.road_embedding_path = './libcity/cache/{}/evaluate_cache/road_embedding_{}_{}_{}.npy'. \
            format(self.exp_id, self.model, self.dataset, self.output_dim)
        self.model = SRN2VecModule(node_num=self.num_nodes, device=self.device, emb_dim=self.output_dim, out_dim=2)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def run(self, data=None):
        self.model.to(self.device)
        start_time = time.time()
        for epoch in tqdm(range(self.iter)):
            self.model.train()
            total_loss = 0
            for data, labels in self.dataloader:
                X = data.to(self.device)
                y = labels.float().to(self.device)
                self.optim.zero_grad()
                yh = self.model(X)
                loss = self.model.loss_func(yh.squeeze(), y.squeeze())
                loss.backward()
                self.optim.step()
                total_loss += loss.item()
            if epoch % 100 == 0:
                self._logger.info("Epoch {}, Loss {}".format(epoch, total_loss))
        t1 = time.time() - start_time
        self._logger.info("cost time is " + str(t1 / self.iter))
        node_embedding = self.model.embedding.weight.data.cpu().detach().numpy()
        np.save(self.npy_cache_file, node_embedding)
        np.save(self.road_embedding_path, node_embedding)
        self._logger.info('词向量和模型保存完成')
        self._logger.info('保存至 ' + self.npy_cache_file)
        self._logger.info('词向量维度：(' + str(len(node_embedding)) + ',' + str(len(node_embedding[0])) + ')')


class SRN2VecModule(nn.Module):
    def __init__(self, node_num, device, emb_dim: int = 128, out_dim: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(node_num, emb_dim)
        self.lin_vx = nn.Linear(emb_dim, emb_dim)
        self.lin_vy = nn.Linear(emb_dim, emb_dim)
        self.lin_out = nn.Linear(emb_dim, out_dim)
        self.act_out = nn.Sigmoid()
        self.loss_func = nn.BCELoss()

    def forward(self, x):
        emb = self.embedding(x)
        # y_emb = self.embedding(vy)

        # x = self.lin_vx(emb[:, 0])
        # y = self.lin_vy(emb[:, 1])
        x = emb[:, 0, :] * emb[:, 1, :]  # aggregate embeddings

        x = self.lin_out(x)

        yh = self.act_out(x)

        return yh
