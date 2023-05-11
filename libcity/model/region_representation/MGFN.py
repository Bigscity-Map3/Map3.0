from logging import getLogger
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from libcity.model.abstract_traffic_tradition_model import AbstractTraditionModel
class MGFN(AbstractTraditionModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self.mob_patterns = data_feature.get("mob_patterns")
        self.mob_adj = data_feature.get("od_matrix")
        self.time_slice = data_feature.get("time_slice")
        self.num_nodes = data_feature.get("num_nodes")
        self.geo_to_ind = data_feature.get('geo_to_ind', None)
        self.ind_to_geo = data_feature.get('ind_to_geo', None)
        self._logger = getLogger()
        self.output_dim = config.get('output_dim', 64)
        self.iter = config.get('max_epoch', 2000)
        self.dataset = config.get('dataset', '')
        self.learning_rate = config.get('learning_rate', 0.0001)
        self.weight_dacay = config.get('weight_dacay', 5e-4)
        self.n_cluster = data_feature.get("n_cluster")
        self.model = config.get('model', '')
        self.exp_id = config.get('exp_id', None)
        self.txt_cache_file = './libcity/cache/{}/evaluate_cache/embedding_{}_{}_{}.txt'. \
            format(self.exp_id, self.model, self.dataset, self.output_dim)
        self.model_cache_file = './libcity/cache/{}/model_cache/embedding_{}_{}_{}.m'. \
            format(self.exp_id, self.model, self.dataset, self.output_dim)
        self.npy_cache_file = './libcity/cache/{}/evaluate_cache/embedding_{}_{}_{}.npy'. \
            format(self.exp_id, self.model, self.dataset, self.output_dim)

    def run(self, data=None):
        input_tensor = torch.Tensor(self.mob_patterns)
        label = torch.Tensor(self.mob_adj)
        criterion = SimLoss()
        model = MGFN_layer(graph_num=self.n_cluster, node_num=self.num_nodes, output_dim=self.output_dim)
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_dacay)
        self._logger.info("start training,lr={},weight_dacay={}".format(self.learning_rate,self.weight_dacay))
        for epoch in range(self.iter):
            model.train()
            s_out, t_out = model(input_tensor)
            loss = criterion(s_out, t_out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch % 50 == 0:
                self._logger.info("Epoch {}, Loss {}".format(epoch, loss.item()))
        node_embedding = model.out_feature()
        node_embedding = node_embedding.detach().numpy()
        np.save(self.npy_cache_file, node_embedding)
        self._logger.info('词向量和模型保存完成')
        self._logger.info('词向量维度：(' + str(len(node_embedding)) + ',' + str(len(node_embedding[0])) + ')')

class DeepFc(nn.Module):
    def __init__(self, input_dim, output_dim):
        # 输入层，隐藏层*2,输出层.隐藏层节点数目为输入层两倍
        super(DeepFc, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.Linear(input_dim * 2, input_dim * 2),
            nn.LeakyReLU(negative_slope=0.3, inplace=True),
            nn.Linear(input_dim * 2, output_dim),
            nn.LeakyReLU(negative_slope=0.3, inplace=True), )

        self.output = None

    def forward(self, x):
        output = self.model(x)
        self.output = output
        return output

    def out_feature(self):
        return self.output


class ConcatLinear(nn.Module):
    """
    input: (*, a) and (*, b)
    output (*, c)
    """
    def __init__(self, in_1, in_2, out, dropout=0.1):
        super(ConcatLinear, self).__init__()
        self.linear1 = nn.Linear(in_1+in_2, out)
        self.act1 = nn.LeakyReLU(negative_slope=0.3, inplace=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(out)

        self.linear2 = nn.Linear(out, out)
        self.act2 = nn.LeakyReLU(negative_slope=0.3, inplace=True)

    def forward(self, x1, x2):
        src = torch.cat([x1, x2], -1)
        out = self.linear1(src)
        out = src + self.dropout1(out)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.act2(self.linear2(out))
        return out


class GraphStructuralEncoder(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,):
        super(GraphStructuralEncoder, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward,)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu

    def forward(self, src):
        src2 = self.self_attn(src, src, src,)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class MobilityPatternJointLearning(nn.Module):
    """
    input: (7, 180, 180)
    output: (180, 144)
    """
    def __init__(self, graph_num, node_num, output_dim):
        super(MobilityPatternJointLearning, self).__init__()
        self.graph_num = graph_num
        self.node_num = node_num
        self.num_multi_pattern_encoder = 3
        self.num_cross_graph_encoder = 1
        self.multi_pattern_blocks = nn.ModuleList(
            [GraphStructuralEncoder(d_model=node_num, nhead=2) for _ in range(self.num_multi_pattern_encoder)])
        self.cross_graph_blocks = nn.ModuleList(
            [GraphStructuralEncoder(d_model=node_num, nhead=2) for _ in range(self.num_cross_graph_encoder)])
        self.fc = DeepFc(self.graph_num*self.node_num, output_dim)
        self.linear_out = nn.Linear(node_num, output_dim)
        self.para1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)#the size is [1]
        self.para1.data.fill_(0.7)
        self.para2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)#the size is [1]
        self.para2.data.fill_(0.3)
        assert node_num % 2 == 0
        self.s_linear = nn.Linear(node_num, int(node_num / 2))
        self.o_linear = nn.Linear(node_num, int(node_num / 2))
        self.concat = ConcatLinear(int(node_num / 2), int(node_num / 2), node_num)

    def forward(self, x):
        out = x
        for multi_pattern in self.multi_pattern_blocks:
            out = multi_pattern(out)
        multi_pattern_emb = out
        out = out.transpose(0, 1)
        for cross_graph in self.cross_graph_blocks:
            out = cross_graph(out)
        out = out.transpose(0, 1)
        out = out*self.para2 + multi_pattern_emb*self.para1
        out = out.contiguous()
        out = out.view(-1, self.node_num*self.graph_num)
        out = self.fc(out)
        return out


class MGFN_layer(nn.Module):
    def __init__(self, graph_num, node_num, output_dim):
        super(MGFN_layer, self).__init__()
        self.encoder = MobilityPatternJointLearning(graph_num=graph_num, node_num=node_num, output_dim=output_dim)
        self.decoder_s = nn.Linear(output_dim, output_dim)
        self.decoder_t = nn.Linear(output_dim, output_dim)
        self.feature = None
        self.name = "MGFN"

    def forward(self, x):
        # x = x.unsqueeze(0)
        self.feature = self.encoder(x)
        out_s = self.decoder_s(self.feature)
        out_t = self.decoder_t(self.feature)
        return out_s, out_t

    def out_feature(self, ):
        return self.feature


def pairwise_inner_product(mat_1, mat_2):
    n, m = mat_1.shape  # (180, 144)
    mat_expand = torch.unsqueeze(mat_2, 0)  # (1, 180, 144),
    mat_expand = mat_expand.expand(n, n, m)  # (180, 180, 144),
    mat_expand = mat_expand.permute(1, 0, 2)  # (180, 180, 144),
    inner_prod = torch.mul(mat_expand, mat_1)  # (180, 180, 144),
    inner_prod = torch.sum(inner_prod, axis=-1)  # (180, 180),
    return inner_prod


def _mob_loss(s_embeddings, t_embeddings, mob):
    inner_prod = pairwise_inner_product(s_embeddings, t_embeddings)
    softmax1 = nn.Softmax(dim=-1)
    phat = softmax1(inner_prod)
    loss = torch.sum(-torch.mul(mob, torch.log(phat+0.0001)))
    inner_prod = pairwise_inner_product(t_embeddings, s_embeddings)
    softmax2 = nn.Softmax(dim=-1)
    phat = softmax2(inner_prod)
    loss += torch.sum(-torch.mul(torch.transpose(mob, 0, 1), torch.log(phat+0.0001)))
    return loss


class SimLoss(nn.Module):
    def __init__(self):
        super(SimLoss, self).__init__()

    def forward(self, out1, out2, label):

        mob_loss = _mob_loss(out1, out2, label)
        loss = mob_loss
        return loss