from collections import Counter
from torch.nn import init
import math
from itertools import zip_longest

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.utils import shuffle
from libcity.model.abstract_traffic_tradition_model import AbstractTraditionModel
from libcity.model.utils import next_batch


class GeoTeaser(AbstractTraditionModel):
    def __init__(self, config, data_feature):
        self.config = config
        super().__init__(config, data_feature)
        self.num_loc = self.data_feature.get("num_loc")
        self.user_num = self.data_feature.get("user_num")
        self.pos_pairs = self.data_feature.get('pos_pairs')
        self.sample_table = self.data_feature.get('sample_table')
        self.embed_size= self.config.get("embed_size", 128)
        self.teaser_week_embed_size = self.config.get("teaser_week_embed_size", 0)
        self.teaser_beta = self.config.get('teaser_beta', 0.0)
        self.num_neg = self.config.get("num_neg", 5)


        self.init_lr = 1e-3
        self.num_epochs = self.config.get('num_epoch', 5)
        self.batch_size = self.config.get('batch_size', 16)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

        self.model = config.get('model', '')
        self.dataset = config.get('dataset', '')
        self.exp_id = config.get('exp_id', None)
        self.output_dim = config.get('output_dim', 128)
        self.embedding_path = './libcity/cache/{}/evaluate_cache/embedding_{}_{}_{}.npy' \
            .format(self.exp_id, self.model, self.dataset, self.output_dim)

    def get_neg_v_sampling(self, batch_size, num_neg):
        neg_v = np.random.choice(self.sample_table, size=(batch_size, num_neg))
        return neg_v

    def run(self):
        model = GeoTeaserModel(num_vocab=self.num_loc, num_user=self.user_num,
                              embed_dimension=self.embed_size,
                              week_embed_dimension=self.teaser_week_embed_size,
                              beta=self.teaser_beta)
        model = model.to(self.device)
        init_lr = 1e-3
        optimizer = torch.optim.SGD(model.parameters(), lr=init_lr)

        pos_pairs = self.pos_pairs
        trained_batches = 0
        batch_count = math.ceil(self.num_epochs * len(pos_pairs) / self.batch_size)

        avg_loss = 0.
        for epoch in range(self.num_epochs):
            for pair_batch in next_batch(shuffle(pos_pairs), self.batch_size):
                neg_v = self.get_neg_v_sampling(len(pair_batch), self.num_neg)
                neg_v = torch.tensor(neg_v).long().to(self.device)

                user, pos_u, week, pos_v, neg_ne, neg_nn = zip(*pair_batch)
                user, pos_u, week, pos_v, neg_ne, neg_nn = (torch.tensor(item).long().to(self.device)
                                                            for item in (user, pos_u, week, pos_v, neg_ne, neg_nn))

                optimizer.zero_grad()
                loss = model(pos_u, pos_v, neg_v, user, week, neg_ne, neg_nn)
                loss.backward()
                optimizer.step()
                trained_batches += 1
                loss_val = loss.detach().cpu().numpy().tolist()
                avg_loss += loss_val

                if trained_batches % 10000 == 0:
                    lr = init_lr * (1.0 - trained_batches / batch_count)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                    print('Avg loss: %.5f' % (avg_loss / 100), flush=True)
                    avg_loss = 0.
        np.save(self.embedding_path, model.static_embed())
        return model.static_embed()


class GeoTeaserModel(nn.Module):
    def __init__(self, num_vocab, num_user, embed_dimension, week_embed_dimension, beta=2.0):
        super().__init__()
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
        @param pos_u: positive input tokens, shape (batch_size).
        @param pos_v: positive output tokens, shape (batch_size, window_size*2).
        @param neg_v: negative output tokens, shape (batch_size, num_neg).
        @param user: user indices corresponding to input tokens, shape (batch_size)
        @param weekday: weekday indices corresponding to input tokens, shape (batch_size)
        @param neg_ne: negative unvisited locations, shape (batch_size, num_ne_neg)
        @param neg_nn: negative non-neighborhood locations, shape (batch_size, num_nn_neg)
        """
        embed_u = self.u_embeddings(pos_u)  # (batch_size, embed_size)
        embed_week = self.week_embeddings(weekday)  # (batch_size, embed_size)
        embed_cat = torch.cat([embed_u, embed_week], dim=-1)

        embed_v = self.v_embeddings(pos_v)  # (batch_size, N, 2*embed_size)
        score = torch.mul(embed_cat.unsqueeze(1), embed_v).squeeze()  # (batch_size, N, embed_size)
        score = torch.sum(score, dim=-1)  # (batch_size, N)
        score = F.logsigmoid(score)

        neg_embed_v = self.v_embeddings(neg_v)  # (batch_size, num_neg, embed_size)
        neg_score = torch.bmm(neg_embed_v, embed_cat.unsqueeze(2)).squeeze()
        neg_score = F.logsigmoid(-1 * neg_score)

        embed_user = self.user_embeddings(user)  # (batch_size, embed_size + week_embed_size)
        neg_embed_ne = self.v_embeddings(neg_ne)  # (batch_size, N, embed_size + week_embed_size)
        neg_embed_nn = self.v_embeddings(neg_nn)

        neg_ne_score = torch.bmm(embed_cat.unsqueeze(1) - neg_embed_ne, embed_user.unsqueeze(2)).squeeze()  # (batch_size, N)
        neg_ne_score = F.logsigmoid(neg_ne_score)
        neg_nn_score = torch.bmm(embed_cat.unsqueeze(1) - neg_embed_nn, embed_user.unsqueeze(2)).squeeze()
        neg_nn_score = F.logsigmoid(neg_nn_score)

        return -1 * (torch.sum(score) + torch.sum(neg_score) + self.beta * (torch.sum(neg_ne_score) + torch.sum(neg_nn_score)))

    def static_embed(self):
        return self.u_embeddings.weight[:self.num_vocab].detach().cpu().numpy()

