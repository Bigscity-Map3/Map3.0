import os
import math
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch.utils.data import DataLoader, Dataset

from logging import getLogger
from libcity.model.abstract_replearning_model import AbstractReprLearningModel
from torch_geometric.utils import dropout_edge
from tqdm import tqdm
import numpy as np


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


def node_node_loss(node_rep1, node_rep2, measure):
    num_nodes = node_rep1.shape[0]

    pos_mask = torch.eye(num_nodes).cuda()

    if measure == 'jsd':
        return jsd(node_rep1, node_rep2, pos_mask)
    elif measure == 'nce':
        return nce(node_rep1, node_rep2, pos_mask)
    elif measure == 'ntx':
        return ntx(node_rep1, node_rep2, pos_mask)


def seq_seq_loss(seq_rep1, seq_rep2, measure):
    batch_size = seq_rep1.shape[0]

    pos_mask = torch.eye(batch_size).cuda()

    if measure == 'jsd':
        return jsd(seq_rep1, seq_rep2, pos_mask)
    elif measure == 'nce':
        return nce(seq_rep1, seq_rep2, pos_mask)
    elif measure == 'ntx':
        return ntx(seq_rep1, seq_rep2, pos_mask)


def node_seq_loss(node_rep, seq_rep, sequences, measure):
    batch_size = seq_rep.shape[0]
    num_nodes = node_rep.shape[0]

    pos_mask = torch.zeros((batch_size, num_nodes + 1)).cuda()
    for row_idx, row in enumerate(sequences):
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
                 graph_encoder1, graph_encoder2, seq_encoder):
        super(MultiViewModel, self).__init__()

        self.vocab_size = vocab_size
        self.node_embedding = nn.Embedding(vocab_size, embed_size)
        self.padding = torch.zeros(1, hidden_size, requires_grad=False).cuda()
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

class TrajRoadDataset(Dataset):
    def __init__(self, data, num_nodes):
        self.data = data
        self.num_nodes = num_nodes

    def __getitem__(self, index):

        return self.data[index] - self.num_nodes

    def __len__(self):
        return len(self.data)

class JCLRNT(AbstractReprLearningModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self.adj_mx = data_feature.get('adj_mx')
        self.num_nodes = data_feature.get('num_nodes', 1)
        self.geo_to_ind = data_feature.get('geo_to_ind', None)
        self.ind_to_geo = data_feature.get('ind_to_geo', None)
        self._logger = getLogger()
        self.edge_index1 = data_feature.get('edge_index', None)
        self.edge_index2 = data_feature.get('edge_index_aug', None)
        self.traj_road = data_feature.get('traj_road', None)

        self.output_dim = config.get('output_dim', 64)
        self.is_directed = config.get('is_directed', True)
        self.num_workers = config.get('num_workers', 10)
        self.iter = config.get('max_epoch', 1000)
        self.embed_size = config.get('embed_size',128)
        self.hidden_size = config.get('hidden_size',128)
        self.drop_rate = config.get('drop_rate',0.2)
        self.drop_edge_rate = config.get('drop_edge_rate',0.4)
        self.drop_road_rate = config.get('drop_road_rate',0.2)
        self.learning_rate = config.get('learning_rate',1e-3)
        self.weight_decay = config.get('weight_decay',1e-6)
        self.num_epochs = config.get('num_epochs', 5)
        self.batch_size = config.get('batch_size',64)
        self.measure = config.get('loss_measure',"jsd")
        self.is_weighted = config.get('weighted_loss',False)
        self.mode = config.get('mode','s')
        self.l_st = config.get('lambda_st',0.8)
        self.l_ss = self.l_tt = 0.5 * (1 - self.l_st)
        self.activation = {'relu': nn.ReLU(), 'prelu': nn.PReLU()}[config.get("activation","relu")]
        self.max_len = config.get("max_len", 100)
        self.min_len = config.get("min_len", 10)

        self.model = config.get('model', '')
        self.dataset = config.get('dataset', '')
        self.exp_id = config.get('exp_id', None)

        self.model_cache_file = './libcity/cache/{}/model_cache'.format(self.exp_id)

        self.npy_cache_file = './libcity/cache/{}/evaluate_cache/embedding_{}_{}_{}.npy'. \
            format(self.exp_id, self.model, self.dataset, self.output_dim)




    def run(self, train_dataloader=None,eval_dataloader=None):

        traj_road_temp = [x for x in self.traj_road if len(x) <= self.max_len ]
        traj_road = torch.full([len(traj_road_temp), self.max_len], self.num_nodes)
        traj_road = torch.Tensor(traj_road).cuda()
        train_dataset = TrajRoadDataset(traj_road, self.num_nodes)
        tran_dataloader = DataLoader(train_dataset, batch_size=self.batch_size)


        graph_encoder1 = GraphEncoder(self.embed_size, self.hidden_size, GATConv, 2, self.activation)
        graph_encoder2 = GraphEncoder(self.embed_size, self.hidden_size, GATConv, 2, self.activation)
        seq_encoder = TransformerModel(self.hidden_size, 4, self.hidden_size, 2, self.drop_rate)
        model = MultiViewModel(self.num_nodes, self.embed_size, self.hidden_size, self.edge_index1, self.edge_index2,
                                graph_encoder1, graph_encoder2, seq_encoder).cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)


        current_epoch = 1
        if current_epoch < self.num_epochs:
            print("\n=== Training ===")
            for epoch in range(current_epoch, self.num_epochs + 1):
                for n, data_batch in tqdm(enumerate(tran_dataloader)):
                    w_batch = 0
                    model.train()
                    optimizer.zero_grad()
                    node_rep1, node_rep2, seq_rep1, seq_rep2 = model(data_batch)
                    loss_ss = node_node_loss(node_rep1, node_rep2, self.measure)
                    loss_tt = seq_seq_loss(seq_rep1, seq_rep2, self.measure)
                    if self.is_weighted:
                        loss_st1 = weighted_ns_loss(node_rep1, seq_rep2, w_batch, self.measure)
                        loss_st2 = weighted_ns_loss(node_rep2, seq_rep1, w_batch, self.measure)
                    else:
                        loss_st1 = node_seq_loss(node_rep1, seq_rep2, data_batch, self.measure)
                        loss_st2 = node_seq_loss(node_rep2, seq_rep1, data_batch, self.measure)
                    loss_st = (loss_st1 + loss_st2) / 2
                    loss = self.l_ss * loss_ss + self.l_tt * loss_tt + self.l_st * loss_st
                    loss.backward()
                    optimizer.step()
                    if not (n + 1) % 10:
                        t = datetime.now().strftime('%m-%d %H:%M:%S')
                        print(f'{t} | (Train) | Epoch={epoch}, batch={n + 1} loss={loss.item():.4f}')

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, os.path.join(self.model_cache_file, "_".join([ str(self.dataset), str(self.output_dim), '.pt'])))

            node_embedding = model.node_embedding.weight.detach().cpu().numpy()

            self.npy_cache_file = './libcity/cache/{}/evaluate_cache/embedding_{}_{}_{}.npy'. \
                format(self.exp_id, self.model, self.dataset, self.output_dim)

            np.save(self.npy_cache_file, node_embedding)

        return model