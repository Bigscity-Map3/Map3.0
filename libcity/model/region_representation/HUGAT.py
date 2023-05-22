import dgl
import numpy as np

import torch
from dgl.utils import expand_as_pair
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
            num_meta_paths=len(self.meta_path),
            in_size=self.feature.shape[1],
            hidden_size=self.hidden_units,
            num_heads=self.num_heads,
            dropout=self.dropout,
        ).to(self.device)
        self.g=self.g.to(self.device)
        # 构建元路径下不同的同构图，等到看到了g的样子再修改
        self.gs=[]
        for mp in self.meta_path:
            homogeneous_graph = dgl.metapath_reachable_graph(self.g, mp)
            homogeneous_graph = dgl.to_homogeneous(homogeneous_graph)
            self.gs.append(homogeneous_graph)


    def run(self, data=None):
        pass

    # 训练
    def train(self, epoch):
        pass






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
    num_meta_paths : number of homogeneous graphs generated from the metapaths.
    in_size : input feature dimension
    out_size : output feature dimension
    layer_num_heads : number of attention heads
    dropout : Dropout probability

    Inputs
    ------
    g : list[DGLGraph]
        List of graphs
    h : tensor
        Input features

    Outputs
    -------
    tensor
        The output feature
    """

    def __init__(
        self, num_meta_paths, in_size, out_size, layer_num_heads, dropout
    ):
        super(HANLayer, self).__init__()

        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        for i in range(num_meta_paths):
            self.gat_layers.append(
                GATConv(
                    in_size,
                    out_size,
                    layer_num_heads,
                    dropout,
                    dropout,
                    activation=F.elu,
                )
            )
        self.semantic_attention = SemanticAttention(
            in_size=out_size * layer_num_heads
        )
        self.num_meta_paths = num_meta_paths

    def forward(self, gs, h):
        semantic_embeddings = []

        for i, g in enumerate(gs):
            semantic_embeddings.append(self.gat_layers[i](g, h).flatten(1))
        semantic_embeddings = torch.stack(
            semantic_embeddings, dim=1
        )  # (N, M, D * K)

        return self.semantic_attention(semantic_embeddings)  # (N, D * K)


class HAN(nn.Module):
    def __init__(
        self, num_meta_paths, in_size, hidden_size,  num_heads, dropout
    ):
        super(HAN, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(
            HANLayer(
                num_meta_paths, in_size, hidden_size, num_heads[0], dropout
            )
        )
        for l in range(1, len(num_heads)):
            self.layers.append(
                HANLayer(
                    num_meta_paths,
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