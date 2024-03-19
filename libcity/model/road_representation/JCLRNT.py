from logging import getLogger
from tqdm import tqdm
from libcity.model.abstract_traffic_tradition_model import AbstractTraditionModel
import numpy as np
import torch
import torch.nn as nn
import os
import math
from datetime import datetime
import torch.nn.functional as F
import time
from torch_geometric.nn import GATConv

class JCLRNT(AbstractTraditionModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self.device = config.get('device')
        self.dataloader = data_feature.get('dataloader')
        self.num_nodes = data_feature.get("num_nodes")
        print(self.num_nodes)
        self._logger = getLogger()
        self.output_dim = config.get('output_dim', 128)
        self.iter = config.get('max_epoch', 5)
        self.model = config.get('model', '')
        self.exp_id = config.get('exp_id', None)
        self.dataset = config.get('dataset', '')
        self.edge_index1 = data_feature.get('edge_index')
        self.edge_index2 = data_feature.get('edge_index_aug')
        self.edge_cache_file = './libcity/cache/{}/evaluate_cache/edge_{}_{}_{}'. \
            format(self.exp_id, self.model, self.dataset, self.output_dim)
        self.txt_cache_file = './libcity/cache/{}/evaluate_cache/road_embedding_{}_{}_{}.txt'. \
            format(self.exp_id, self.model, self.dataset, self.output_dim)
        self.model_cache_file = './libcity/cache/{}/model_cache/embedding_{}_{}_{}.m'. \
            format(self.exp_id, self.model, self.dataset, self.output_dim)
        self.npy_cache_file = './libcity/cache/{}/evaluate_cache/road_embedding_{}_{}_{}.npy'. \
            format(self.exp_id, self.model, self.dataset, self.output_dim)
        self.traj_train_embedding_file = './libcity/cache/{}/evaluate_cache/traj_train_embedding_{}_{}_{}.npy'. \
            format(self.exp_id, self.model, self.dataset, self.output_dim)
        self.traj_val_embedding_file = './libcity/cache/{}/evaluate_cache/traj_val_embedding_{}_{}_{}.npy'. \
            format(self.exp_id, self.model, self.dataset, self.output_dim)
        self.traj_test_embedding_file = './libcity/cache/{}/evaluate_cache/traj_test_embedding_{}_{}_{}.npy'. \
            format(self.exp_id, self.model, self.dataset, self.output_dim)
        # self.traj_origin_test_embedding_file = './libcity/cache/{}/evaluate_cache/traj_origin_test_embedding_{}_{}_{}.npy'. \
        #     format(self.exp_id, self.model, self.dataset, self.output_dim)
        # self.traj_detour_test_embedding_file = './libcity/cache/{}/evaluate_cache/traj_detour_test_embedding_{}_{}_{}.npy'. \
        #     format(self.exp_id, self.model, self.dataset, self.output_dim)
        # self.traj_detour_others_embedding_file = './libcity/cache/{}/evaluate_cache/traj_detour_others_embedding_{}_{}_{}.npy'. \
        #     format(self.exp_id, self.model, self.dataset, self.output_dim)
        self.road_embedding_path = './libcity/cache/{}/evaluate_cache/road_embedding_{}_{}_{}.npy'. \
            format(self.exp_id, self.model, self.dataset, self.output_dim)
        self.hidden_size = config.get('hidden_size',128)
        self.drop_rate = config.get('drop_rate', 0.2)
        self.drop_edge_rate = config.get('drop_edge_rate', 0.2)
        self.drop_road_rate = config.get('drop_road_rate', 0.2)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.weight_decay = config.get('weight_decay', 1e-6)
        self.measure = config.get('loss_measure', "jsd")
        self.is_weighted = config.get('weighted_loss', False)
        self.mode = config.get('mode', 's')
        self.l_st = config.get('lambda_st', 0.8)
        self.traj_train = torch.from_numpy(data_feature.get('traj_arr_train'))
        self.traj_eval = torch.from_numpy(data_feature.get('traj_arr_eval'))
        self.traj_test = torch.from_numpy(data_feature.get('traj_arr_test'))
        # self.traj_origin_test = torch.from_numpy(data_feature.get('traj_arr_origin_test'))
        # self.traj_detour_test = torch.from_numpy(data_feature.get('traj_arr_detour_test'))
        # self.traj_detour_others = torch.from_numpy(data_feature.get('traj_arr_detour_others'))
        self.l_ss = self.l_tt = 0.5 * (1 - self.l_st)
        self.activation = {'relu': nn.ReLU(), 'prelu': nn.PReLU()}[config.get("activation", "relu")]
        self.num_epochs = config.get('num_epochs', 5)
        self.graph_encoder1 = GraphEncoder(self.output_dim, self.hidden_size, GATConv, 2, self.activation)
        self.graph_encoder2 = GraphEncoder(self.output_dim, self.hidden_size, GATConv, 2, self.activation)
        self.seq_encoder = TransformerModel(self.hidden_size, 4, self.hidden_size, 2, self.drop_rate)
        self.model = MultiViewModel(self.num_nodes, self.output_dim, self.hidden_size, self.edge_index1, self.edge_index2,
                               self.graph_encoder1, self.graph_encoder2, self.seq_encoder,self.device).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

    def run(self, data=None):
        np.save(self.edge_cache_file, self.edge_index1.cpu())
        start_time = time.time()
        for epoch in tqdm(range(self.iter)):
            total_loss = 0
            self.model.train()
            for n, data_batch in enumerate(self.dataloader):
                data_batch = data_batch.to(self.device)
                w_batch = 0
                self.optimizer.zero_grad()
                node_rep1, node_rep2, seq_rep1, seq_rep2 = self.model(data_batch)
                #loss_ss = node_node_loss(node_rep1, node_rep2, self.measure,self.device)
                loss_ss = 0
                loss_tt = seq_seq_loss(seq_rep1, seq_rep2, self.measure,self.device)
                if self.is_weighted:
                    loss_st1 = weighted_ns_loss(node_rep1, seq_rep2, w_batch, self.measure)
                    loss_st2 = weighted_ns_loss(node_rep2, seq_rep1, w_batch, self.measure)
                else:
                    loss_st1 = node_seq_loss(node_rep1, seq_rep2, data_batch, self.measure,self.device)
                    loss_st2 = node_seq_loss(node_rep2, seq_rep1, data_batch, self.measure,self.device)
                loss_st = (loss_st1 + loss_st2) / 2
                loss = self.l_ss * loss_ss + self.l_tt * loss_tt + self.l_st * loss_st
                total_loss += loss
                loss.backward()
                self.optimizer.step()
            self._logger.info("Epoch {}, Loss {}".format(epoch, total_loss))
        t1 = time.time()-start_time
        self._logger.info('cost time is '+str(t1/self.iter))
        node_embedding = self.model.encode_graph()[0].cpu().detach().numpy()
        np.save(self.road_embedding_path,node_embedding)
        torch.save((self.model.state_dict(), self.optimizer.state_dict()), self.model_cache_file)
        self.save_traj_embedding(self.traj_train,self.traj_train_embedding_file)
        self.save_traj_embedding(self.traj_eval, self.traj_val_embedding_file)
        self.save_traj_embedding(self.traj_test,self.traj_test_embedding_file)
        # self.save_traj_embedding(self.traj_origin_test,self.traj_origin_test_embedding_file)
        # self.save_traj_embedding(self.traj_detour_test,self.traj_detour_test_embedding_file)
        # self.save_traj_embedding(self.traj_detour_others,self.traj_detour_others_embedding_file)
        
    def save_traj_embedding(self,traj_test,traj_embedding_file):
        result_list = []
        traj_num = traj_test.shape[0]
        print('traj_num=' + str(traj_num))
        start_index = 0
        while start_index < traj_num:
            end_index = min(traj_num, start_index + 1280)
            batch_embedding = self.model.encode_sequence(traj_test[start_index:end_index].to(self.device))[
                0].cpu().detach().numpy()
            result_list.append(batch_embedding)
            start_index = end_index
        traj_embedding = np.concatenate(result_list, axis=0)
        self._logger.info('词向量维度：(' + str(len(traj_embedding)) + ',' + str(len(traj_embedding[0])) + ')')
        np.save(traj_embedding_file, traj_embedding)
        self._logger.info('保存至  '+traj_embedding_file)


def jsd(z1, z2, pos_mask):
    neg_mask = 1 - pos_mask

    sim_mat = torch.mm(z1, z2.t())
    E_pos = math.log(2.) - F.softplus(-sim_mat)
    E_neg = F.softplus(-sim_mat) + sim_mat - math.log(2.)
    return (E_neg * neg_mask).sum() / neg_mask.sum() - (E_pos * pos_mask).sum() / pos_mask.sum()


def nce(z1, z2, pos_mask):
    sim_mat = torch.mm(z1, z2.t())
    return nn.BCEWithLogitsLoss(reduction='none')(sim_mat, pos_mask).sum(1).mean()


def ntx(z1, z2, pos_mask, tau=0.5, normalize=False):
    if normalize:
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
    sim_mat = torch.mm(z1, z2.t())
    sim_mat = torch.exp(sim_mat / tau)
    return -torch.log((sim_mat * pos_mask).sum(1) / sim_mat.sum(1) / pos_mask.sum(1)).mean()


def node_node_loss(node_rep1, node_rep2, measure,device):
    num_nodes = node_rep1.shape[0]

    pos_mask = torch.eye(num_nodes).to(device)

    if measure == 'jsd':
        return jsd(node_rep1, node_rep2, pos_mask)
    elif measure == 'nce':
        return nce(node_rep1, node_rep2, pos_mask)
    elif measure == 'ntx':
        return ntx(node_rep1, node_rep2, pos_mask)


def seq_seq_loss(seq_rep1, seq_rep2, measure,device):
    batch_size = seq_rep1.shape[0]

    pos_mask = torch.eye(batch_size).to(device)

    if measure == 'jsd':
        return jsd(seq_rep1, seq_rep2, pos_mask)
    elif measure == 'nce':
        return nce(seq_rep1, seq_rep2, pos_mask)
    elif measure == 'ntx':
        return ntx(seq_rep1, seq_rep2, pos_mask)


def node_seq_loss(node_rep, seq_rep, sequences, measure,device):
    batch_size = seq_rep.shape[0]
    num_nodes = node_rep.shape[0]

    pos_mask = torch.zeros((batch_size, num_nodes + 1)).to(device)
    for row_idx, row in enumerate(sequences):
        row = row.type(torch.long)
        pos_mask[row_idx][row] = 1.
    pos_mask = pos_mask[:, :-1]

    if measure == 'jsd':
        return jsd(seq_rep, node_rep, pos_mask)
    elif measure == 'nce':
        return nce(seq_rep, node_rep, pos_mask)
    elif measure == 'ntx':
        return ntx(seq_rep, node_rep, pos_mask)


def weighted_ns_loss(node_rep, seq_rep, weights, measure):
    if measure == 'jsd':
        return jsd(seq_rep, node_rep, weights)
    elif measure == 'nce':
        return nce(seq_rep, node_rep, weights)
    elif measure == 'ntx':
        return ntx(seq_rep, node_rep, weights)


class PositionalEncoding(nn.Module):
    def __init__(self, dim, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, input_size, num_heads, hidden_size, num_layers, dropout=0.3):
        super(TransformerModel, self).__init__()
        self.pos_encoder = PositionalEncoding(input_size, dropout)
        encoder_layers = nn.TransformerEncoderLayer(input_size, num_heads, hidden_size, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

    def forward(self, src, src_mask, src_key_padding_mask):
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask, src_key_padding_mask)
        return output


class GraphEncoder(nn.Module):
    def __init__(self, input_size, output_size, encoder_layer, num_layers, activation):
        super(GraphEncoder, self).__init__()

        self.num_layers = num_layers
        self.activation = activation

        self.layers = [encoder_layer(input_size, output_size)]
        for _ in range(1, num_layers):
            self.layers.append(encoder_layer(output_size, output_size))
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.activation(self.layers[i](x, edge_index))
        return x


class MultiViewModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, edge_index1, edge_index2,
                 graph_encoder1, graph_encoder2, seq_encoder,device):
        super(MultiViewModel, self).__init__()

        self.vocab_size = vocab_size
        self.node_embedding = nn.Embedding(vocab_size, embed_size)
        self.padding = torch.zeros(1, hidden_size, requires_grad=False).to(device)
        self.edge_index1 = edge_index1
        self.edge_index2 = edge_index2
        self.graph_encoder1 = graph_encoder1
        self.graph_encoder2 = graph_encoder2
        self.seq_encoder = seq_encoder

    def encode_graph(self):
        node_emb = self.node_embedding.weight
        node_enc1 = self.graph_encoder1(node_emb, self.edge_index1)
        node_enc2 = self.graph_encoder2(node_emb, self.edge_index2)
        return node_enc1 + node_enc2, node_enc1, node_enc2

    def encode_sequence(self, sequences):
        _, node_enc1, node_enc2 = self.encode_graph()

        batch_size, max_seq_len = sequences.size()
        src_key_padding_mask = (sequences == self.vocab_size)
        pool_mask = (1 - src_key_padding_mask.int()).transpose(0, 1).unsqueeze(-1)

        lookup_table1 = torch.cat([node_enc1, self.padding], 0)
        seq_emb1 = torch.index_select(
            lookup_table1, 0, sequences.view(-1)).view(batch_size, max_seq_len, -1).transpose(0, 1)
        seq_enc1 = self.seq_encoder(seq_emb1, None, src_key_padding_mask)
        seq_pooled1 = (seq_enc1 * pool_mask).sum(0) / pool_mask.sum(0)

        lookup_table2 = torch.cat([node_enc2, self.padding], 0)
        seq_emb2 = torch.index_select(
            lookup_table2, 0, sequences.view(-1)).view(batch_size, max_seq_len, -1).transpose(0, 1)
        seq_enc2 = self.seq_encoder(seq_emb2, None, src_key_padding_mask)
        seq_pooled2 = (seq_enc2 * pool_mask).sum(0) / pool_mask.sum(0)
        return seq_pooled1 + seq_pooled2, seq_pooled1, seq_pooled2

    def forward(self, sequences):
        _, node_enc1, node_enc2 = self.encode_graph()
        _, seq_pooled1, seq_pooled2 = self.encode_sequence(sequences)
        return node_enc1, node_enc2, seq_pooled1, seq_pooled2
