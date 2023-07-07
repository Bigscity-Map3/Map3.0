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


def next_batch(data, batch_size):
    data_length = len(data)
    num_batches = math.ceil(data_length / batch_size)
    for batch_index in range(num_batches):
        start_index = batch_index * batch_size
        end_index = min((batch_index + 1) * batch_size, data_length)
        yield data[start_index:end_index]


class HS(nn.Module):
    def __init__(self, num_vocab, embed_dimension):
        super().__init__()
        self.num_vocab = num_vocab
        self.embed_dimension = embed_dimension

        # Input embedding.
        self.u_embeddings = nn.Embedding(num_vocab, embed_dimension, sparse=True)
        # Output embedding. Here is actually the embedding of inner nodes.
        self.w_embeddings = nn.Embedding(num_vocab, embed_dimension, padding_idx=0, sparse=True)

        initrange = 0.5 / self.embed_dimension
        self.u_embeddings.weight.data.uniform_(-initrange, initrange)
        self.w_embeddings.weight.data.uniform_(-0, 0)

    def forward(self, pos_u, pos_w, neg_w, **kwargs):
        """
        @param pos_u: positive input tokens, shape (batch_size, window_size * 2)
        @param pos_w: positive output tokens, shape (batch_size, num_pos)
        @param neg_w: negative output tokens, shape (batch_size, num_neg)
        @param sum: whether to sum up all scores.
        """
        pos_u_embed = self.u_embeddings(pos_u)  # (batch_size, window_size * 2, embed_size)
        pos_u_embed = pos_u_embed.sum(1, keepdim=True)  # (batch_size, 1, embed_size)

        pos_w_mask = torch.where(pos_w == 0, torch.ones_like(pos_w), torch.zeros_like(pos_w)).bool()  # (batch_size, num_pos)
        pos_w_embed = self.w_embeddings(pos_w)  # (batch_size, num_pos, embed_size)
        score = torch.mul(pos_u_embed, pos_w_embed).sum(dim=-1)  # (batch_size, num_pos)
        score = F.logsigmoid(-1 * score)  # (batch_size, num_pos)
        score = score.masked_fill(pos_w_mask, torch.tensor(0.0).to(pos_u.device))

        neg_w_mask = torch.where(neg_w == 0, torch.ones_like(neg_w), torch.zeros_like(neg_w)).bool()
        neg_w_embed = self.w_embeddings(neg_w)
        neg_score = torch.mul(pos_u_embed, neg_w_embed).sum(dim=-1)  # (batch_size, num_neg)
        neg_score = F.logsigmoid(neg_score)
        neg_score = neg_score.masked_fill(neg_w_mask, torch.tensor(0.0).to(pos_u.device))
        if kwargs.get('sum', True):
            return -1 * (torch.sum(score) + torch.sum(neg_score))
        else:
            return score, neg_score

class Tale(AbstractTraditionModel):
    def __init__(self, config, data_feature):
        self.config = config
        super().__init__(config, data_feature)
        self.num_loc = self.data_feature.get("num_loc")
        self.path_pairs = self.data_feature.get('path_pairs')
        self.num_temp_vocab = self.data_feature.get('num_temp_vocab')
        self.embed_size = self.config.get("embed_size", 128)


        self.num_epochs = self.config.get('num_epoch', 5)
        self.batch_size = self.config.get('batch_size', 16)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')

        self.init_lr = 1e-3

    def run(self):

        model = TaleModel(num_vocab=self.num_loc, num_temp_vocab=self.num_temp_vocab, embed_dimension=self.embed_size)
        model = model.to(self.device)
        init_lr = 1e-3
        optimizer = torch.optim.SGD(model.parameters(), lr=init_lr)

        train_set = self.path_pairs
        trained_batches = 0
        batch_count = math.ceil(self.num_epochs * len(train_set) / self.batch_size)

        avg_loss = 0.
        for epoch in range(self.num_epochs):
            for pair_batch in next_batch(shuffle(train_set), self.batch_size):
                flatten_batch = []
                for row in pair_batch:
                    flatten_batch += [[row[0], p, n, pr] for p, n, pr in zip(*row[1:])]

                context, pos_pairs, neg_pairs, prop = zip(*flatten_batch)
                context = torch.tensor(context).long().to(self.device)
                pos_pairs, neg_pairs = (
                    torch.tensor(list(zip_longest(*item, fillvalue=0))).long().to(self.device).transpose(0, 1)
                    for item in (pos_pairs, neg_pairs))  # (batch_size, longest)
                prop = torch.tensor(prop).float().to(self.device)

                optimizer.zero_grad()
                loss = model(context, pos_pairs, neg_pairs, prop=prop)
                loss.backward()
                optimizer.step()
                trained_batches += 1
                loss_val = loss.detach().cpu().numpy().tolist()
                avg_loss += loss_val

                if trained_batches % 1000 == 0:
                    lr = init_lr * (1.0 - trained_batches / batch_count)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr
                    print('Avg loss: %.5f' % (avg_loss / 1000))
                    avg_loss = 0.
        print(model.u_embeddings.weight.detach().cpu().numpy())
        return model.u_embeddings.weight.detach().cpu().numpy()

class TaleModel(HS):
    def __init__(self, num_vocab, num_temp_vocab, embed_dimension):
        super().__init__(num_vocab, embed_dimension)
        self.w_embeddings = nn.Embedding(num_temp_vocab, embed_dimension, padding_idx=0, sparse=True)



    def forward(self, pos_u, pos_w, neg_w, **kwargs):
        """
        @param pos_u: positive input tokens, shape (batch_size, window_size * 2)
        @param pos_w: positive output tokens, shape (batch_size, pos_path_len)
        @param neg_w: negative output tokens, shape (batch_size, neg_path_len)
        """
        pos_score, neg_score = super().forward(pos_u, pos_w, neg_w, sum=False)  # (batch_size, pos_path_len)
        prop = kwargs['prop']
        pos_score, neg_score = (-1 * (item.sum(axis=1) * prop).sum() for item in (pos_score, neg_score))
        return pos_score + neg_score





