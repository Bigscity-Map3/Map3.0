import os
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
import networkx as nx
from itertools import cycle, islice
from tqdm import tqdm
import copy

from veccity.evaluator.downstream_models.abstract_model import AbstractModel
from veccity.data.preprocess import preprocess_detour

def k_shortest_paths_nx(G, source, target, k, weight='weight'):
    return list(islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))

def build_graph(rel_file, geo_file):

    rel = pd.read_csv(rel_file)
    geo = pd.read_csv(geo_file)
    
    edge2len = {}
    geoid2coord = {}
    for i, row in tqdm(geo.iterrows(), total=geo.shape[0]):
        geo_id = row.geo_id
        length = float(row.length)
        edge2len[geo_id] = length
        # geoid2coord[geo_id] = row.coordinates

    graph = nx.DiGraph()

    for i, row in tqdm(rel.iterrows(), total=rel.shape[0]):
        prev_id = row.origin_id
        curr_id = row.destination_id

        # Use length as weight
        # weight = geo.iloc[prev_id].length
        
        # Use avg_speed as weight
        weight = row.avg_time
        if weight == float('inf'):
            # weight = 9999999
            pass
            # print(row)
        graph.add_edge(prev_id, curr_id, weight=weight)

    return graph

class TrajEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, embedding, device):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding = embedding
        self.n_layers = n_layers
        self.device = device
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, dropout=0.1 if n_layers > 1 else 0.0, batch_first=True).to(device)

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
    def __init__(self, input_dim, hidden_dim, embedding, device, n_layers=1, tau=0.08,is_static=True):
        super().__init__()
        self.device = device
        self.tau = tau
        self.hidden_dim = hidden_dim
        self.is_static = is_static
        if is_static:
            self.traj_encoder = TrajEncoder(input_dim, hidden_dim, n_layers, embedding, device)
        elif not is_static:
            self.traj_encoder = embedding.encode_sequence
            self.transfer = nn.Linear(embedding.d_model, hidden_dim).to(device)
        
        self.projection=nn.Sequential(nn.Linear(hidden_dim,hidden_dim),nn.ReLU(),nn.Linear(hidden_dim,hidden_dim)).to(self.device)
        

    def forward(self, path, valid_len, pos_path, pos_valid_len, neg_path, neg_valid_len,**kwargs):
        batch_size = path.shape[0]
        path=path.to(self.device)
        pos_path=pos_path.to(self.device)
        neg_path=neg_path.to(self.device)
        
    
        if self.is_static:
            base_embedding = self.projection(self.traj_encoder(path, valid_len)).unsqueeze(-1)
            pos_embedding = self.projection(self.traj_encoder(pos_path, pos_valid_len)).view(batch_size, -1, self.hidden_dim)
            neg_embedding = self.projection(self.traj_encoder(neg_path, neg_valid_len)).view(batch_size, -1, self.hidden_dim)
        else:
            base_embedding = self.projection(self.transfer(self.traj_encoder(path, valid_len,**kwargs))).unsqueeze(-1)
            pos_embedding = self.projection(self.transfer(self.traj_encoder(pos_path, pos_valid_len,**kwargs))).view(batch_size, -1, self.hidden_dim)
            neg_embedding = self.projection(self.transfer(self.traj_encoder(neg_path, neg_valid_len,**kwargs))).view(batch_size, -1, self.hidden_dim)

        pos_scores = torch.bmm(pos_embedding, base_embedding).view(batch_size, -1)
        pos_label = torch.Tensor(np.full(pos_scores.shape, 1)).type(torch.FloatTensor).to(self.device)
        neg_scores = torch.bmm(neg_embedding, base_embedding).view(batch_size, -1)
        neg_label = torch.Tensor(np.full(neg_scores.shape, 0)).type(torch.FloatTensor).to(self.device)
        scores = torch.cat([pos_scores, neg_scores], axis=1)
        labels = torch.cat([pos_label, neg_label], axis=1)
        scores /= self.tau
        loss = -(F.log_softmax(scores, dim=1) * labels).sum() / labels.sum()
        return loss

class SimilaritySearchModel(AbstractModel):
    def __init__(self, config):
        preprocess_detour(config)
        self._logger = getLogger()
        self._logger.warning('Evaluating Trajectory Similarity Search')
        self.config = config
        self.device = config.get('device')
        self.dataset = config.get('dataset')
        self.model = config.get('model')
        self.exp_id = config.get('exp_id', 0)
        self.output_dim = config.get('output_dim',128)
        self.representation_object = config.get('representation_object', 'region')
        geo_df = pd.read_csv(os.path.join('raw_data', self.dataset, self.dataset + '.geo'))
        self.num_nodes = geo_df[geo_df['traffic_type'] == self.representation_object].shape[0]
        self.min_len = config.get('min_len', 20)
        self.max_len = config.get('max_len', 50)
        self.detour_rate = config.get('detour_rate', 0.8)
        self.downstream_epoch = config.get('downstream_epoch', 20)
        self.batch_size = config.get('downstream_batch_size', 64)
        self.learning_rate = self.config.get('learning_rate', 0.001)
        self.weight_decay = self.config.get('weight_decay', 1e-5)
        self.hidden_dim = self.config.get('downstream_hidden_dim', 512)
        self.pos_size = 3
        self.neg_size = self.config.get('neg_size', 10)
        self.is_static = self.config.get('is_static', True)
        # self.train_traj_df = self.filter_traj(os.path.join('veccity/cache/dataset_cache',
        #     self.dataset, 'traj_' + self.representation_object + '_train.csv'))
        # self.test_traj_df = self.filter_traj(os.path.join('veccity/cache/dataset_cache',
        #     self.dataset, 'traj_' + self.representation_object + '_test.csv'))
        self.embedding_path = 'veccity/cache/{}/evaluate_cache/{}_embedding_{}_{}_{}.npy'.format(
            self.exp_id, self.representation_object, self.model, self.dataset, self.output_dim
        )
        try:
            self.embedding = np.load(self.embedding_path)
            new_row = np.zeros((1, self.embedding.shape[1]))
            self.embedding = np.concatenate((self.embedding, new_row), axis=0)  # embedding[padding_id] = 0
        except:
            self.embedding = None
        self.ori_traj=np.load(os.path.join('veccity/cache/dataset_cache',self.dataset, 'ori_trajs.npz'))
        self.query_traj=np.load(os.path.join('veccity/cache/dataset_cache',self.dataset, 'query_trajs.npz'))# num,path
        
        self.train_index=list(range(5000))
        self.test_index=list(range(5000,10000))


    def data_loader(self, padding_id, num_queries):        
        num_samples = len(self.test_traj_df)
        x_arr = np.full([num_samples, self.max_len], padding_id, dtype=np.int32)
        x_len = np.zeros([num_samples], dtype=np.int32)
        tlist=[]
        for i in range(num_samples):
            row = self.test_traj_df.iloc[i]
            path_arr = np.array(row['path'], dtype=np.int32)
            x_arr[i, :row['path_len']] = path_arr
            x_len[i] = row['path_len']
            tlist.append(path_arr)
        
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
    
    def next_batch_index(self, ds, bs, shuffle=True):
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

    def evaluation(self,**kwargs):
        batch_size = self.batch_size
        num_queries = min(self.config.get('num_queries', 5000), len(self.query_traj['trajs']) - 2000)
        
        random_index = np.random.permutation(num_queries)

        data=torch.from_numpy(self.ori_traj['trajs'][2000:]).to(self.device)
        data_len=torch.from_numpy(self.ori_traj['lengths'][2000:]).to(self.device)
        queries=torch.from_numpy(self.query_traj['trajs'][2000:])[random_index].to(self.device)
        queries_len=torch.from_numpy(self.query_traj['lengths'][2000:])[random_index].to(self.device)

        y=random_index
  
        data_size = data.shape[0]

        self.downstream_model.eval()
        x = []
        for batch_idx in self.next_batch_index(data_size, batch_size, shuffle=False):
            seq_rep = self.downstream_model.traj_encoder(data[batch_idx], data_len[batch_idx],**kwargs)
            if isinstance(seq_rep, tuple):
                seq_rep = seq_rep[0]
            x.append(seq_rep.detach().cpu())
        x = torch.cat(x, dim=0).numpy()

        q = []
        for batch_idx in self.next_batch_index(num_queries, batch_size, shuffle=False):
            seq_rep = self.downstream_model.traj_encoder(queries[batch_idx], queries_len[batch_idx],**kwargs)
            if isinstance(seq_rep, tuple):
                seq_rep = seq_rep[0]
            q.append(seq_rep.detach().cpu())
        q = torch.cat(q, dim=0).numpy()

        index = faiss.IndexFlatL2(x.shape[1])
        index.add(x)
        import pdb
        pdb.set_trace()
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
            self.result['Mean Rank'] = rank_sum / num_queries + 1.0
            self.result['No Hit'] = no_hit 
            self.result['HR@' + str(k)] =  hit / (num_queries - no_hit)
            self._logger.info(f'HR@{k}: {hit / (num_queries - no_hit)}')
        self._logger.info('Mean Rank: {}, No Hit: {}'.format(self.result['Mean Rank'], self.result['No Hit']))


    def run(self,model=None,**kwargs):
        """
        返回评估结果
        """
        ori_paths=torch.from_numpy(self.ori_traj['trajs'][:2000])
        ori_length=torch.from_numpy(self.ori_traj['lengths'][:2000])

        detour_paths=torch.from_numpy(self.query_traj['trajs'][:2000])
        detour_length=torch.from_numpy(self.query_traj['lengths'][:2000])

        all_len=2000

        negtive_index=torch.randperm(all_len)
        negtive_paths=ori_paths[negtive_index]
        negtive_length=ori_length[negtive_index]

                    

        if self.is_static:
            self.downstream_model = CLModel(input_dim=self.output_dim, hidden_dim=self.hidden_dim,
                                        embedding=self.embedding, is_static=self.is_static,device=self.device)
        else:
            self.downstream_model = CLModel(input_dim=self.output_dim, hidden_dim=self.hidden_dim,
                                        embedding=model, is_static=self.is_static,device=self.device)

        self.downstream_model.to(self.device)
        optimizer = Adam(lr=self.learning_rate, params=self.downstream_model.parameters(), weight_decay=self.weight_decay)
        self._logger.info('Start training downstream model...')
        best_loss=-1
        for epoch in range(self.downstream_epoch):
            total_loss = 0.0
            for i in range(0, all_len , self.batch_size):
                l=i
                r=min(i+self.batch_size,all_len)
                loss = self.downstream_model(ori_paths[l:r], ori_length[l:r],
                                            detour_paths[l:r], detour_length[l:r],
                                            negtive_paths[l:r], negtive_length[l:r],**kwargs)
                optimizer.zero_grad()
                total_loss += loss.item()
                loss.backward()
                optimizer.step()

            if best_loss==-1 or total_loss<best_loss:
                self.best_model=copy.deepcopy(self.downstream_model)
            self._logger.info("epoch {} complete! training loss is {:.2f}.".format(epoch, total_loss))
        self.downstream_model=self.best_model
       

        self.evaluation(**kwargs)
        return self.result

    def save_result(self, save_path, filename=None):
        pass

    def clear(self):
        pass