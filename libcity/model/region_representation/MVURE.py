import math
from logging import getLogger
import numpy as np
import torch
import torch.nn as nn
import dgl
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv
import torch.optim as optim
from libcity.model.abstract_traffic_tradition_model import AbstractTraditionModel
#[2020-IJCAI Multi-View Joint Graph Representation Learning for Urban Region Embedding]
class MVURE(AbstractTraditionModel):
    def __init__(self,config,data_feature):
        super().__init__(config,data_feature)
        self.mob_adj = data_feature.get("mob_adj")
        self.s_adj_sp = data_feature.get("s_adj_sp")
        self.t_adj_sp = data_feature.get("t_adj_sp")
        self.poi_adj = data_feature.get("poi_adj")
        self.poi_adj_sp = data_feature.get("poi_adj_sp")
        self.feature = data_feature.get("feature")
        self.num_nodes = data_feature.get("num_nodes")
        self.geo_to_ind = data_feature.get('geo_to_ind', None)
        self.ind_to_geo = data_feature.get('ind_to_geo', None)
        self._logger = getLogger()
        self.output_dim = config.get('output_dim', 96)
        self.is_directed = config.get('is_directed', True)
        self.dataset = config.get('dataset', '')
        self.iter = config.get('max_epoch', 2000)
        self.model = config.get('model', '')
        self.exp_id = config.get('exp_id', None)
        self.weight_dacay = config.get('weight_dacay', 1e-3)
        self.learning_rate = config.get('learning_rate', 0.005)
        self.early_stopping = config.get('early_stopping',10)
        self.txt_cache_file = './libcity/cache/{}/evaluate_cache/embedding_{}_{}_{}.txt'. \
            format(self.exp_id, self.model, self.dataset, self.output_dim)
        self.model_cache_file = './libcity/cache/{}/model_cache/embedding_{}_{}_{}.m'. \
            format(self.exp_id, self.model, self.dataset, self.output_dim)
        self.npy_cache_file = './libcity/cache/{}/evaluate_cache/embedding_{}_{}_{}.npy'. \
            format(self.exp_id, self.model, self.dataset, self.output_dim)
    def run(self, data=None):
        self.preprocess_features(self.feature)
        model = MVURE_Layer(self.mob_adj, self.s_adj_sp, self.t_adj_sp, self.poi_adj, self.feature,
                                 self.feature.shape[2], self.output_dim)
        optimizer = optim.Adam(model.parameters(),lr=self.learning_rate, weight_decay=self.weight_dacay)
        item_num, _ = self.mob_adj.shape
        self._logger.info("start training,lr={},weight_dacay={}".format(self.learning_rate,self.weight_dacay))
        outs = None
        for epoch in range(self.iter):
            model.train()
            outs = model()
            loss = model.calculate_loss(outs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch%50 == 0:
                self._logger.info("Epoch {}, Loss {}".format(epoch, loss.item()))
        node_embedding = outs[-2]
        node_embedding = node_embedding.detach().numpy()
        np.save(self.npy_cache_file, node_embedding)
        self._logger.info('词向量和模型保存完成')
        self._logger.info('词向量维度：(' + str(len(node_embedding)) + ',' + str(len(node_embedding[0])) + ')')

    def preprocess_features(self,feature):
        """Row-normalize feature matrix and convert to tuple representation"""
        feature = feature[0]
        colvar = np.var(feature, axis=1, keepdims=True)
        colmean = np.mean(feature, axis=1, keepdims=True)
        c_inv = np.power(colvar, -0.5)
        c_inv[np.isinf(c_inv)] = 0.
        feature = np.multiply((feature - colmean), c_inv)
        feature = feature[np.newaxis]
        return feature

    def adj_to_bias(self,adj, sizes, nhood=1):
        adj = adj[np.newaxis]
        nb_graphs = adj.shape[0]  # num_graph个图
        mt = np.empty(adj.shape)  # 输出矩阵的形状和adj相同
        # 图g的转换
        for g in range(nb_graphs):
            mt[g] = np.eye(adj.shape[1])  # 与g形状相同的对角矩阵
            for _ in range(nhood):  # 通过self-loop构建K阶邻接矩阵，即A^(K),这里K=1
                mt[g] = np.matmul(mt[g], (adj[g] + np.eye(adj.shape[1])))
            # 大于0的置1，小于等于0的保持不变
            for i in range(sizes[g]):
                for j in range(sizes[g]):
                    if mt[g][i][j] > 0.0:
                        mt[g][i][j] = 1.0
        # mt中1的位置为0，位置为0的返回很小的负数-1e9
        return -1e9 * (1.0 - mt)

class self_attn(nn.Module):
    def __init__(self,input_dim,hidden_dim):
        #这里的input*dim和hidden_dim是对于每一个节点展平后的维度，因此传入前要乘上num_nodes
        super(self_attn,self).__init__()
        self.hidden_dim = hidden_dim
        # self.Q_linear = nn.Linear(input_dim,hidden_dim)
        # self.K_linear = nn.Linear(input_dim,hidden_dim)

    def forward(self,inputs):
        """
        :param inputs: [num_views,num_nodes,embedding_dim]
        :return:[num_views,num_nodes,embedding_dim],每一个视图做一遍注意力机制
        """
        num_views,num_nodes,embedding_dim = inputs.shape
        inputs_3dim = inputs
        result = torch.zeros([num_views , num_nodes, embedding_dim], dtype=torch.float)
        inputs = inputs.view(num_views, num_nodes * embedding_dim)
        for i in range(num_views):
            Q_linear = nn.Linear(self.input_dim,self.hidden_dim)
            K_linear = nn.Linear(self.input_dim, self.hidden_dim)
            Q = Q_linear(inputs)
            Q = torch.unsqueeze(Q,-1)
            K = K_linear(inputs)
            K = torch.unsqueeze(K,-1)
            d_k = math.sqrt(self.hidden_dim/num_nodes)
            attn = torch.bmm(Q,K.transpose(-2,-1))/d_k
            attn = torch.squeeze(attn)
            for i in range(num_views):
                result[i] += (attn[i]*inputs_3dim[i])
        return result

class mv_attn(nn.Module):
    def __init__(self,input_dim):
        super(mv_attn,self).__init__()
        #输入为单视图表征，输出权值
        self.mlp = nn.Linear(input_dim,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,inputs):
        """
        :param inputs: [num_views,num_nodes,embedding_dim]
        :return:
        """
        num_views,num_nodes,embedding_dim = inputs.shape
        inputs_3dim = inputs
        inputs = inputs.view(num_views, num_nodes * embedding_dim)
        omega = self.mlp(inputs)
        omega = self.sigmoid(omega)
        omega = torch.squeeze(omega)
        result = torch.zeros([num_nodes, embedding_dim], dtype=torch.float)
        for i in range(num_views):
            result += (omega[i]*inputs_3dim[i])
        return result
class MVURE_Layer(nn.Module):
    def __init__(self,mob_adj,s_graph,t_graph,poi_graph,feature,input_dim,output_dim):
        super(MVURE_Layer,self).__init__()
        self.mob_adj = mob_adj
        self.inputs = torch.from_numpy(feature)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.s_gat = GATConv(in_feats=self.inputs.shape[-1],out_feats=self.output_dim,num_heads=12,attn_drop=0.2,activation=F.relu)
        self.t_gat = GATConv(in_feats=self.inputs.shape[-1],out_feats=self.output_dim,num_heads=12,attn_drop=0.2,activation=F.relu)
        self.poi_gat = GATConv(in_feats=self.inputs.shape[-1],out_feats=self.output_dim,num_heads=12,attn_drop=0.2,activation=F.relu)
        self.num_nodes = feature.shape[-2]
        self.fused_layer = self_attn(self.num_nodes*self.output_dim,self.num_nodes*48)
        self.mv_layer = mv_attn(self.num_nodes*self.output_dim)
        self.alpha = 0.8
        self.beta = 0.5
        self.s_graph = s_graph
        self.t_graph = t_graph
        self.poi_graph = poi_graph

    def construct_dgl_graph(self,adj_mx):
        """
        :param adj_mx:邻接矩阵，[num_nodes,num_nodes],np.array
        :return: dgl_graph,将邻接矩阵中大于0的全部算成一条边，
        """
        num_edges = np.count_nonzero(adj_mx)
        src_index = torch.zeros([num_edges])
        dst_index = torch.zeros([num_edges])
        num_nodes = adj_mx.shape[0]
        edge_cnt = 0
        for i in range(num_nodes):
            for j in range(num_nodes):
                if adj_mx[i][j] > 0:
                    src_index[edge_cnt] = i
                    dst_index[edge_cnt] = j
                    edge_cnt += 1
        dgl_graph = dgl.graph((src_index,dst_index),num_nodes=num_nodes)
        return dgl_graph





    def forward(self):
        s_dgl_graph = self.construct_dgl_graph(self.s_graph)
        s_out = self.s_gat(graph = s_dgl_graph,feat = self.inputs)
        t_dgl_graph = self.construct_dgl_graph(self.t_graph)
        t_out = self.t_gat(graph = t_dgl_graph,feat = self.inputs)
        poi_dgl_graph = self.construct_dgl_graph(self.poi_graph)
        poi_out = self.poi_gat(graph=poi_dgl_graph, feat=self.inputs)
        single_view_out = torch.stack([s_out,t_out,poi_out],dim = 0)
        fused_out = self.fused_layer(single_view_out)
        s_out = self.alpha * fused_out[0] + (1 - self.alpha)* s_out
        t_out = self.alpha * fused_out[1] + (1 - self.alpha) * t_out
        poi_out = self.alpha * fused_out[2] + (1 - self.alpha) * poi_out
        fused_out = torch.stack([s_out, t_out, poi_out], dim=0)
        mv_out = self.mv_layer(fused_out)
        s_out = self.beta*s_out + (1-self.beta)*mv_out
        t_out = self.beta * t_out + (1 - self.beta) * mv_out
        poi_out = self.beta * poi_out + (1 - self.beta) * mv_out
        result = torch.stack([s_out, t_out, poi_out], dim=0)
        return result

    def calculate_loss(self,embedding):
        """
        :param embedding:Tensor[num_views,num_nodes,embedding_dim]
        :return:
        """
        s_embeddings = embedding[0]
        t_embeddings = embedding[1]
        poi_embeddings = embedding[2]
        loss1 = self.calculate_mob_loss(s_embeddings,t_embeddings,self.mob_adj)
        loss2 = self.calculate_poi_loss(poi_embeddings,self.poi_graph)
        return loss1 + loss2

    def calculate_mob_loss(self,s_embeddings,t_embeddings,mob_adj):
        """
        :param s_embeddings:tensor[num_nodes,embedding_dim]
        :param t_embeddings: tensor[num_nodes,embedding_dim]
        :param mob_adj: np.array[num_nodes,num_nodes]
        :return:
        """
        inner_prod = self.pairwise_inner_product(s_embeddings,t_embeddings)
        phat = torch.softmax(inner_prod,dim = -1)
        loss = torch.sum(-torch.mm(mob_adj,torch.log(phat)))
        inner_prod = self.pairwise_inner_product(t_embeddings, s_embeddings)
        phat = torch.softmax(inner_prod,dim = -1)
        loss += torch.sum(-torch.mm(mob_adj.T,torch.log(phat)))
        return loss


    def calculate_poi_loss(self,embedding,poi_adj):
        """
        :param embedding: tensor[num_nodes,embedding_dim]
        :param poi_adj: np.array[num_nodes,num_nodes]
        :return:
        """
        inner_prod = self.pairwise_inner_product(embedding, embedding)
        loss_function = nn.MSELoss(reduction="sum")
        loss = loss_function(inner_prod,poi_adj)
        return loss


    def pairwise_inner_product(self,mat_1,mat_2):
        n,_=mat_1.shape
        result = torch.zeros([n,n],dtype=torch.float)
        for i in range(n):
            for j in range(n):
                result[i][j] = torch.dot(mat_1[i],mat_2[j])
        return result



