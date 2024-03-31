import sys
sys.path.append('../../../')
import os
import json
import math
import pandas as pd
import numpy as np
from logging import getLogger
import faiss
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import pdb

from libcity.evaluator.downstream_models.abstract_model import AbstractModel


class TrajEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, embedding, device):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding = embedding
        self.n_layers = n_layers
        self.device = device
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, dropout=0.1 if n_layers > 1 else 0.0, batch_first=True)

    def forward(self, path, valid_len):
        original_shape = path.shape  # [batch_size, traj_len]
        full_embed = [torch.from_numpy(self.embedding[int(i)]).to(torch.float32) for i in path.view(-1)]
        full_embed = torch.stack(full_embed)
        full_embed = full_embed.view(*original_shape, self.input_dim)  # [batch_size, traj_len, embed_size]
        pack_x = pack_padded_sequence(full_embed, lengths=valid_len, batch_first=True, enforce_sorted=False).to(self.device)
        h0 = torch.zeros(self.n_layers, full_embed.size(0), self.hidden_dim).to(self.device)
        c0 = torch.zeros(self.n_layers, full_embed.size(0), self.hidden_dim).to(self.device)
        out, _ = self.lstm(pack_x, (h0, c0))
        out, _ = pad_packed_sequence(out, batch_first=True)
        out = torch.stack([out[i, ind - 1, :] for i, ind in enumerate(valid_len)])  # [batch_size, hidden_dim]
        return out


class CLModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding, device, n_layers=1, tau=0.08):
        super().__init__()
        self.device = device
        self.tau = tau
        self.traj_encoder = TrajEncoder(input_dim, hidden_dim, n_layers, embedding, device)

    def forward(self, path, pos_list, neg_list):
        valid_len = len(path)
        pos_path, pos_valid_len = zip(*pos_list)
        neg_path, neg_valid_len = zip(*neg_list)
        base_embedding = self.traj_encoder(torch.LongTensor(path).unsqueeze(0), torch.LongTensor([valid_len])).squeeze(0)
        pos_embedding = self.traj_encoder(torch.LongTensor(pos_path), torch.LongTensor(pos_valid_len))
        neg_embedding = self.traj_encoder(torch.LongTensor(neg_path), torch.LongTensor(neg_valid_len))
        pos_scores = torch.matmul(pos_embedding, base_embedding)
        pos_label = torch.Tensor([1 for _ in range(pos_scores.size(0))]).type(torch.FloatTensor).to(self.device)
        neg_scores = torch.matmul(neg_embedding, base_embedding)
        neg_label = torch.Tensor([0 for _ in range(neg_scores.size(0))]).type(torch.FloatTensor).to(self.device)
        scores = torch.cat([pos_scores, neg_scores])
        labels = torch.cat([pos_label, neg_label])
        scores /= self.tau
        loss = -(F.log_softmax(scores, dim=0) * labels).sum() / labels.sum()
        return loss

class SimilaritySearchModel(AbstractModel):
    def __init__(self, config):
        self._logger = getLogger()
        self.config = config
        self.device = config.get('device')
        self.dataset = config.get('dataset')
        self.model = config.get('model')
        self.exp_id = config.get('exp_id', 0)
        self.output_dim = config.get('output_dim')
        self.representation_object = config.get('representation_object', 'region')
        geo_df = pd.read_csv(os.path.join('raw_data', self.dataset, self.dataset + '.geo'))
        self.num_nodes = geo_df[geo_df['traffic_type'] == self.representation_object].shape[0]
        self.min_len = config.get('min_len', 20)
        self.max_len = config.get('max_len', 50)
        self.detour_rate = config.get('detour_rate', 0.8)
        self.downstream_epoch = config.get('downstream_epoch', 10)
        self.learning_rate = self.config.get('learning_rate', 0.001)
        self.weight_decay = self.config.get('weight_decay', 1e-5)
        self.hidden_dim = self.config.get('downstream_hidden_dim', 512)
        self.neg_size = self.config.get('neg_size', 10)
        self.train_traj_df = self.filter_traj(os.path.join('libcity/cache/dataset_cache',
            self.dataset, 'traj_' + self.representation_object + '_train.csv'))
        self.test_traj_df = self.filter_traj(os.path.join('libcity/cache/dataset_cache',
            self.dataset, 'traj_' + self.representation_object + '_test.csv'))
        self.embedding_path = 'libcity/cache/{}/evaluate_cache/{}_embedding_{}_{}_{}.npy'.format(
            self.exp_id, self.representation_object, self.model, self.dataset, self.output_dim
        )
        self.embedding = np.load(self.embedding_path)
        new_row = np.zeros((1, self.embedding.shape[1]))
        self.embedding = np.concatenate((self.embedding, new_row), axis=0)  # embedding[padding_id] = 0
        self._logger.info('Load road emb {}, shape = {}'.format(self.embedding_path, self.embedding.shape))

    def filter_traj(self, file_path):
        df = pd.read_csv(file_path)
        df['path'] = df['path'].map(eval)
        df['path_len'] = df['path'].map(len)
        df = df.loc[(df['path_len'] >= self.min_len) & (df['path_len'] <= self.max_len)]
        return df
    
    def detour(self, path, aug_type):
        rate = self.detour_rate
        new_path = []
        for node in path:
            p = np.random.random_sample()
            if aug_type == 'add':
                new_path.append(node)
                if p > rate:
                    new_path.append(np.random.randint(self.num_nodes))
            elif aug_type == 'delete':
                if p <= rate:
                    new_path.append(node)
            elif aug_type == 'replace':
                new_path.append(np.random.randint(self.num_nodes) if p > rate else node)
        if len(new_path) > self.max_len:
            new_path = new_path[:self.max_len]
        valid_length = len(new_path)
        return new_path, valid_length

    def data_loader(self, padding_id, num_queries, detour_rate=0.1):        
        num_samples = len(self.test_traj_df)
        x_arr = np.full([num_samples, self.max_len], padding_id, dtype=np.int32)
        x_len = np.zeros([num_samples], dtype=np.int32)
        for i in range(num_samples):
            row = self.test_traj_df.iloc[i]
            path_arr = np.array(row['path'], dtype=np.int32)
            x_arr[i, :row['path_len']] = path_arr
            x_len[i] = row['path_len']
        
        random_index = np.random.permutation(num_samples)
        q_arr = np.full([num_queries, self.max_len], padding_id, dtype=np.int32)
        q_len = np.zeros([num_queries], dtype=np.int32)
        for i in range(num_queries):
            row = self.test_traj_df.iloc[random_index[i]]
            new_path, valid_length = self.detour(row['path'], np.random.choice(['add', 'delete', 'replace']))
            q_arr[i, :valid_length] = np.array(new_path, dtype=np.int32)
            q_len[i] = valid_length
        
        y = random_index[:num_queries]
        return torch.Tensor(x_arr), torch.LongTensor(x_len), torch.Tensor(q_arr), torch.LongTensor(q_len), y
    
    def evaluation(self):
        def next_batch_index(ds, bs, shuffle=True):
            num_batches = math.ceil(ds / bs)
            index = np.arange(ds)
            if shuffle:
                index = np.random.permutation(index)
            for i in range(num_batches):
                if i == num_batches - 1:
                    batch_index = index[bs * i:]
                else:
                    batch_index = index[bs * i: bs * (i + 1)]
                yield batch_index
        
        self._logger.info("--- Trajectory Similarity Search ---")
        num_nodes = self.num_nodes
        batch_size = 64
        num_queries = 5000
        data, data_len, queries, queries_len, y = self.data_loader(num_nodes, num_queries)
        data_size = data.shape[0]

        self.downstream_model.eval()
        x = []
        for batch_idx in next_batch_index(data_size, batch_size, shuffle=False):
            seq_rep = self.downstream_model.traj_encoder(data[batch_idx], data_len[batch_idx])
            if isinstance(seq_rep, tuple):
                seq_rep = seq_rep[0]
            x.append(seq_rep.detach().cpu())
        x = torch.cat(x, dim=0).numpy()

        q = []
        for batch_idx in next_batch_index(num_queries, batch_size, shuffle=False):
            seq_rep = self.downstream_model.traj_encoder(queries[batch_idx], queries_len[batch_idx])
            if isinstance(seq_rep, tuple):
                seq_rep = seq_rep[0]
            q.append(seq_rep.detach().cpu())
        q = torch.cat(q, dim=0).numpy()

        index = faiss.IndexFlatL2(x.shape[1])
        index.add(x)
        D, I = index.search(q, 10000)
        self.result = {}
        top = [1, 3, 5, 10, 20]
        for k in top:
            hit = 0
            rank_sum = 0
            no_hit = 0
            for i, r in enumerate(I):
                if y[i] in r:
                    rank_sum += np.where(r == y[i])[0][0]
                    if y[i] in r[:k]:
                        hit += 1
                else:
                    no_hit += 1
            self.result['Mean Rank'] = rank_sum / num_queries
            self.result['no_hit'] = no_hit 
            self.result['HR@' + str(k)] =  hit / (num_queries - no_hit)
            self._logger.info(f'HR@{k}: {hit / (num_queries - no_hit)}')
        self._logger.info('Mean Rank: {}, No Hit: {}'.format(self.result['Mean Rank'], self.result['no_hit']))

    def positive_sampling(self, path):
        return [self.detour(path, 'add'), self.detour(path, 'delete'), self.detour(path, 'replace')]

    def negative_sampling(self, id):
        neg_list = []
        for _ in range(self.neg_size):
            neg_id = np.random.randint(len(self.train_traj_df))
            while id == neg_id:
                neg_id = np.random.randint(len(self.train_traj_df))
            neg_list.append(self.detour(self.train_traj_df['path'].iloc[neg_id], np.random.choice(['add', 'delete', 'replace'])))
        return neg_list

    def run(self):
        """
        返回评估结果
        """
        self.downstream_model = CLModel(input_dim=self.output_dim, hidden_dim=self.hidden_dim, embedding=self.embedding, device=self.device)
        self.downstream_model.to(self.device)
        optimizer = Adam(lr=self.learning_rate, params=self.downstream_model.parameters(), weight_decay=self.weight_decay)
        self._logger.info('Start training downstream model...')
        for epoch in range(self.downstream_epoch):
            total_loss = 0.0
            for i, row in self.train_traj_df.iterrows():
                pos_list = self.positive_sampling(row['path'])
                neg_list = self.negative_sampling(i)
                for path in pos_list:
                    path[0].extend([self.num_nodes] * (self.max_len - len(path[0])))
                for path in neg_list:
                    path[0].extend([self.num_nodes] * (self.max_len - len(path[0])))
                loss = self.downstream_model(row['path'], pos_list, neg_list)
                optimizer.zero_grad()
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
            self._logger.info("epoch {} complete! training loss is {:.2f}.".format(epoch, total_loss))
        self.evaluation()
        file_name = '{}_evaluate_similaritysearch_{}_{}_{}.json'.format(self.exp_id, self.model, self.dataset, self.output_dim)
        self.save_result('libcity/cache/{}/evaluate_cache'.format(self.exp_id), file_name)
        return self.result

    def save_result(self, save_path, filename=None):
        """
        将评估结果保存到 save_path 文件夹下的 filename 文件中

        Args:
            save_path: 保存路径
            filename: 保存文件名
        """
        with open(os.path.join(save_path, filename), 'w') as f:
            json.dump(self.result, f)
        self._logger.info('result save in {}'.format(os.path.join(save_path, filename)))

    def clear(self):
        pass


if __name__ == '__main__':
    os.chdir('/home/tangyb/private/tyb/remote/representation')
    config = {
        'dataset': 'new_xa',
        'representation_object': 'region',
        'device': 'cuda:2',
        'model': 'HREP',
        'output_dim': 144,
        'exp_id': 6
    }
    embedding_path = '/home/tangyb/private/tyb/remote/representation/libcity/cache/6/evaluate_cache/region_embedding_HREP_new_xa_144.npy' 
    downstream_model = SimilaritySearchModel(config)
    pdb.set_trace()
    # downstream_model.run()
