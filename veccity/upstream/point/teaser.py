import torch
from torch import nn
from torch.nn import functional as F

from veccity.upstream.abstract_model import AbstractModel
from math import sin, cos, radians, acos


class Teaser(AbstractModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        teaser_num_ne = config.get('num_ne', 3)  # (number of unvisited locations)
        teaser_num_nn = config.get('num_nn', 3)  # (number of non-neighbor locations)
        teaser_indi_context = config.get('indi_context', False)
        self.alpha = config.get('alpha', 1.0)
        self.beta = config.get('beta', 0.0)
        week_embed_dimension = config.get('week_embed_size', 2)
        coor_mat = data_feature.get('coor_mat')
        num_vocab = data_feature.get('num_loc')
        num_user = data_feature.get('num_user')
        embed_dimension = config.get('embed_size', 128)
        self.__dict__.update(locals())

        self.u_embeddings = nn.Embedding(num_vocab+1, embed_dimension, sparse=True)
        self.v_embeddings = nn.Embedding(num_vocab, embed_dimension + week_embed_dimension, sparse=True)
        self.user_embeddings = nn.Embedding(num_user, embed_dimension + week_embed_dimension, sparse=True)
        self.week_embeddings = nn.Embedding(2, week_embed_dimension, sparse=True)

        initrange = 0.5 / self.embed_dimension
        self.u_embeddings.weight.data.uniform_(-initrange, initrange)
        self.user_embeddings.weight.data.uniform_(-initrange, initrange)
        self.week_embeddings.weight.data.uniform_(-initrange, initrange)
        self.v_embeddings.weight.data.uniform_(-0, 0)

    def forward(self, pos_u, pos_v, neg_v, user, weekday, neg_ne, neg_nn):
        """
        @param pos_u: positive input tokens, shape (batch_size). target
        @param pos_v: positive output tokens, shape (batch_size, window_size * 2). context
        @param neg_v: negative output tokens, shape (batch_size, num_neg).
        @param user: user indices corresponding to input tokens, shape (batch_size)
        @param weekday: weekday indices corresponding to input tokens, shape (batch_size)
        @param neg_ne: negative unvisited locations, shape (batch_size, num_ne_neg)
        @param neg_nn: negative non-neighborhood locations, shape (batch_size, num_nn_neg)
        """
        embed_u = self.u_embeddings(pos_u)  # (batch_size, embed_size)
        embed_week = self.week_embeddings(weekday)  # (batch_size, week_embed_size)
        embed_cat = torch.cat([embed_u, embed_week], dim=-1)  # (batch_size, embed_size + week_embed_size)

        embed_v = self.v_embeddings(pos_v)  # (batch_size, window_size * 2, embed_size + week_embed_size)
        score = torch.mul(embed_cat.unsqueeze(1), embed_v).squeeze()
        # (batch_size, window_size * 2, embed_size + week_embed_size)
        score = torch.sum(score, dim=-1)  # (batch_size, window_size * 2)
        score = F.logsigmoid(score)

        neg_embed_v = self.v_embeddings(neg_v)  # (batch_size, num_neg, embed_size + week_embed_size)
        neg_score = torch.bmm(neg_embed_v, embed_cat.unsqueeze(2)).squeeze()  # (batch_size, num_neg)
        neg_score = F.logsigmoid(-1 * neg_score)

        embed_user = self.user_embeddings(user)  # (batch_size, embed_size + week_embed_size)
        neg_embed_ne = self.v_embeddings(neg_ne)  # (batch_size, N, embed_size + week_embed_size)
        neg_embed_nn = self.v_embeddings(neg_nn)

        neg_ne_score = torch.bmm(embed_cat.unsqueeze(1) - neg_embed_ne, embed_user.unsqueeze(2)).squeeze()
        # (batch_size, N)
        neg_ne_score = F.logsigmoid(neg_ne_score)
        neg_nn_score = torch.bmm(embed_cat.unsqueeze(1) - neg_embed_nn, embed_user.unsqueeze(2)).squeeze()
        neg_nn_score = F.logsigmoid(neg_nn_score)

        return -1 * (self.alpha * (torch.sum(score) + torch.sum(neg_score)) +
                     self.beta * (torch.sum(neg_ne_score) + torch.sum(neg_nn_score)))

    def static_embed(self):
        return self.u_embeddings.weight[:self.num_vocab].detach().cpu().numpy()

    def calculate_loss(self, batch):
        batch_count, pos_u, pos_v, neg_v, user, week, neg_ne, neg_nn = batch
        return self.forward(pos_u, pos_v, neg_v, user, week, neg_ne, neg_nn)
    
    def encode(self, pos_u, week):
        embed_u = self.u_embeddings(pos_u)
        embed_week = self.week_embeddings(week)
        embed_cat = torch.cat([embed_u, embed_week], dim=-1)
        return embed_cat


def dis(lat1, lng1, lat2, lng2):
    lng1, lat1, lng2, lat2 = map(radians, [lng1, lat1, lng2, lat2])
    dlng = lng2 - lng1

    c = sin(lat1) * sin(lat2) + cos(lat1) * cos(lat2) * cos(dlng)
    r = 6371  # km
    if c > 1:
        c = 1
    return int(r * acos(c))



