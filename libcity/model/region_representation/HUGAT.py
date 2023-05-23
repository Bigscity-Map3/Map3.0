import datetime

import dgl
import numpy as np

import torch
from dgl.utils import expand_as_pair
from scipy.special import kl_div
from torch import optim
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
from logging import getLogger
from libcity.model.abstract_traffic_tradition_model import AbstractTraditionModel
from dgl.nn.pytorch import GATConv
class HUGAT(AbstractTraditionModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        # 其他参数
        self.model = config.get('model', '')
        self.dataset = config.get('dataset', '')
        self.exp_id = config.get('exp_id', None)
        self.txt_cache_file = './libcity/cache/{}/evaluate_cache/embedding_{}_{}_{}.txt'. \
            format(self.exp_id, self.model, self.dataset, self.output_dim)
        self.model_cache_file = './libcity/cache/{}/model_cache/embedding_{}_{}_{}.pkl'. \
            format(self.exp_id, self.model, self.dataset, self.output_dim)
        self.npy_cache_file = './libcity/cache/{}/evaluate_cache/embedding_{}_{}_{}.npy'. \
            format(self.exp_id, self.model, self.dataset, self.output_dim)
        self._logger = getLogger()
        # model param config
        self.device = config.get('device', torch.device('cpu'))
        self.lr=config.get('lr',0.005 )
        self.num_heads=config.get('num_heads',[8])# 一个列表，里面是每层的注意力头数，一般只有一层，故只有一个整数；list(int)
        self.hidden_units=config.get('hidden_units',8)# 中间过程隐藏状态，最后输出表征维度为：num_heads[-1]*hidden_units
        self.dropout=config.get('dropout',0.6)
        self.weight_decay=config.get('weight_decay',0.001)
        self.num_epochs=config.get('num_epochs',200)
        self.patience=config.get('patience',100)
        # read processed data
        self.num_nodes=data_feature.get('num_nodes')# region的数量，一个整数，int
        self.g=data_feature.get('g')# 异构图，调用dgl的api生成的，DGLGraph
        self.feature=data_feature.get('feature')# 节点初始特征向量，一个n*m的Tensor，n代表region数量，m代表特征向量的维数
        self.meta_path=data_feature.get('meta_path')# 元路径，格式类似于源代码：[["pa", "ap"], ["pf", "fp"]]，
        # 论文中的分布
        self.P_org_dst=data_feature.get('P_org_dst')
        self.P_dst_org=data_feature.get('P_dst_org')
        self.S_chk=data_feature.get('S_chk')
        self.S_land=data_feature.get('S_land')
        self.crime_count_predict=data_feature.get('crime_count_predict')
        # model initialize
        self.han_model=HAN(
            meta_paths=self.meta_path,
            in_size=self.feature.shape[1],
            hidden_size=self.hidden_units,
            num_heads=self.num_heads,
            dropout=self.dropout,
        ).to(self.device)
        self.g=self.g.to(self.device)
        self.loss_fcn = self.loss
        self.optimizer = torch.optim.Adam(
            self.han_model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        # loss的超参数
        self.loss_alpha=0.3
        self.loss_beta=0.6
        self.loss_gama=0.1

    def loss(self,h):
        # h是已经得到的表征(N,feaure_dim)
        h=h.to(self.device)

        # 计算loss_mob
        tmp=torch.exp(torch.dot(h,h.T))
        P_org_dst_hat=tmp/torch.sum(tmp,dim=0)

        P_dst_org_hat=tmp/torch.sum(tmp.T,dim=0)
        loss_mob=F.kl_div(P_org_dst_hat.log(), self.P_org_dst, reduction='sum')+F.kl_div(P_dst_org_hat.log(), self.P_dst_org, reduction='sum')
        # 计算loss_chk
        num_of_node=len(h)
        P_cat_reg_hat=F.softmax(h,dim=1)
        S_chk_hat=torch.zeros(num_of_node,num_of_node)
        for i in range(num_of_node):
            for j in range(i,num_of_node):
                S_chk_hat[i][j]=torch.norm(P_cat_reg_hat[i]-P_cat_reg_hat[j])/torch.sqrt(torch.tensor(2.0))
                S_chk_hat[j][i]=S_chk_hat[i][j]
        loss_chk=torch.sum((S_chk_hat-self.S_chk)**2)
        # 计算loss_land
        loss_land=torch.sum((S_chk_hat-self.S_land)**2)
        return self.loss_alpha*loss_chk+self.loss_beta*loss_land+self.loss_gama*loss_mob



    def run(self, data=None):
        for epoch in range(self.num_epochs):
            self.han_model.train()
            self.region_embedding_matrix = self.han_model(self.g, self.feature)
            loss = self.loss_fcn(self.region_embedding_matrix)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self._logger.info('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, loss))
        # 保存结果
        with open(self.txt_cache_file, 'w', encoding='UTF-8') as f:
            f.write('{} {}\n'.format(self.region_embedding_matrix.shape[0],self.region_embedding_matrix.shape[1]))
            embeddings = self.region_embedding_matrix.numpy()
            for i in range(self.num_nodes):
                embedding = embeddings[i]
                embedding = str(i) + ' ' + (' '.join(map((lambda x: str(x)), embedding)))
                f.write('{}\n'.format(embedding))
        np.save(self.npy_cache_file, embeddings)
        torch.save(self.han_model.state_dict(), self.model_cache_file)

        self._logger.info('词向量和模型保存完成')
        self._logger.info('词向量维度：( {} , {} )'.format(self.region_embedding_matrix.shape[0],self.region_embedding_matrix.shape[1]))





class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )

    def forward(self, z):
        w = self.project(z).mean(0)  # (M, 1)
        beta = torch.softmax(w, dim=0)  # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)

        return (beta * z).sum(1)  # (N, D * K)


class HANLayer(nn.Module):
    """
    HAN layer.

    Arguments
    ---------
    meta_paths : list of metapaths, each as a list of edge types
    in_size : input feature dimension
    out_size : output feature dimension
    layer_num_heads : number of attention heads
    dropout : Dropout probability

    Inputs
    ------
    g : DGLGraph
        The heterogeneous graph
    h : tensor
        Input features

    Outputs
    -------
    tensor
        The output feature
    """

    def __init__(self, meta_paths, in_size, out_size, layer_num_heads, dropout):
        super(HANLayer, self).__init__()

        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        for i in range(len(meta_paths)):
            self.gat_layers.append(
                GATConv(
                    in_size,
                    out_size,
                    layer_num_heads,
                    dropout,
                    dropout,
                    activation=F.elu,
                    allow_zero_in_degree=True,
                )
            )
        self.semantic_attention = SemanticAttention(
            in_size=out_size * layer_num_heads
        )
        self.meta_paths = list(tuple(meta_path) for meta_path in meta_paths)

        self._cached_graph = None
        self._cached_coalesced_graph = {}

    def forward(self, g, h):
        semantic_embeddings = []

        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for meta_path in self.meta_paths:
                self._cached_coalesced_graph[
                    meta_path
                ] = dgl.metapath_reachable_graph(g, meta_path)

        for i, meta_path in enumerate(self.meta_paths):
            new_g = self._cached_coalesced_graph[meta_path]
            semantic_embeddings.append(self.gat_layers[i](new_g, h).flatten(1))
        semantic_embeddings = torch.stack(
            semantic_embeddings, dim=1
        )  # (N, M, D * K)

        return self.semantic_attention(semantic_embeddings)  # (N, D * K)


class HAN(nn.Module):
    def __init__(
        self, meta_paths, in_size, hidden_size, num_heads, dropout
    ):
        super(HAN, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(
            HANLayer(meta_paths, in_size, hidden_size, num_heads[0], dropout)
        )
        for l in range(1, len(num_heads)):
            self.layers.append(
                HANLayer(
                    meta_paths,
                    hidden_size * num_heads[l - 1],
                    hidden_size,
                    num_heads[l],
                    dropout,
                )
            )

    def forward(self, g, h):
        for gnn in self.layers:
            h = gnn(g, h)

        return h

