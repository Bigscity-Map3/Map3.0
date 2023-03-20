import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.utils import shuffle

from libcity.model.abstract_model import AbstractModel


def next_batch(data, batch_size):
    data_length = len(data)
    num_batches = math.ceil(data_length / batch_size)
    for batch_index in range(num_batches):
        start_index = batch_index * batch_size
        end_index = min((batch_index + 1) * batch_size, data_length)
        yield data[start_index:end_index]


class HS(nn.Module, AbstractModel):
    """
    A Hierarchical Softmax variation of Word2Vec. Can only operate in CBOW mode.
    """
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

    def predict(self, batch):

        return

    def calculate_loss(self, batch):


        return


class NS(nn.Module, AbstractModel):
    """
    A negative sampling variation of word2vec. Can operate in both Skip-Gram and CBOW mode.
    """
    def __init__(self, num_vocab, embed_dimension, cbow=False):
        super().__init__()
        self.num_vocab = num_vocab
        self.embed_dimension = embed_dimension
        self.cbow = cbow

        # Input embedding.
        self.u_embeddings = nn.Embedding(num_vocab, embed_dimension, sparse=True)
        # Output embedding.
        self.v_embeddings = nn.Embedding(num_vocab, embed_dimension, sparse=True)

        initrange = 0.5 / self.embed_dimension
        self.u_embeddings.weight.data.uniform_(-initrange, initrange)
        self.v_embeddings.weight.data.uniform_(-0, 0)

    def forward(self, pos_u, pos_v, neg_v):
        if self.cbow:
            embed_u = self.u_embeddings(pos_v).sum(1, keepdim=True)  # (batch_size, 1, embed_size)
            embed_v = self.v_embeddings(pos_u)  # (batch_size, embed_size)
            score = torch.mul(embed_u, embed_v.unsqueeze(1)).squeeze()  # (batch_size, embed_size)
            score = score.sum(dim=-1)  # (batch_size)
            score = F.logsigmoid(score)

            neg_embed_v = self.v_embeddings(neg_v)  # (batch_size, num_neg, embed_size)
            neg_score = torch.mul(embed_u, neg_embed_v)  # (batch_size, num_neg, embed_size)
            neg_score = neg_score.sum(-1)  # (batch_size, num_neg)
            neg_score = F.logsigmoid(-1 * neg_score)
            return -1 * (torch.sum(score) + torch.sum(neg_score))
        else:
            embed_u = self.u_embeddings(pos_u)  # (batch_size, embed_size)
            embed_v = self.v_embeddings(pos_v)  # (batch_size, N, embed_size), N = 2*window_size
            score = torch.mul(embed_u.unsqueeze(1), embed_v).squeeze()  # (batch_size, N, embed_size)
            score = torch.sum(score, dim=-1)  # (batch_size, N)
            score = F.logsigmoid(score)

            neg_embed_v = self.v_embeddings(neg_v)  # (batch_size, num_neg, embed_size)
            neg_score = torch.bmm(neg_embed_v, embed_u.unsqueeze(2)).squeeze()
            neg_score = F.logsigmoid(-1 * neg_score)
            return -1 * (torch.sum(score) + torch.sum(neg_score))

    def train_ns(ns_model: NS, w2v_dataset: NSData,
                 window_size, num_neg, batch_size, num_epoch, init_lr, optim_class, device):
        ns_model = ns_model.to(device)
        optimizer = optim_class(ns_model.parameters(), lr=init_lr)

        pos_pairs = w2v_dataset.gen_pos_pairs(window_size)
        trained_batches = 0
        batch_count = math.ceil(num_epoch * len(pos_pairs) / batch_size)

        for epoch in range(num_epoch):
            loss_log = []
            for pair_batch in next_batch(shuffle(pos_pairs), batch_size):
                neg_v = w2v_dataset.get_neg_v_sampling(len(pair_batch), num_neg)
                neg_v = torch.tensor(neg_v).long().to(device)

                pos_u, pos_v = zip(*pair_batch)
                pos_u, pos_v = (torch.tensor(item).long().to(device)
                                for item in (pos_u, pos_v))

                optimizer.zero_grad()
                loss = ns_model(pos_u, pos_v, neg_v)
                loss.backward()
                optimizer.step()
                trained_batches += 1
                loss_log.append(loss.detach().cpu().numpy().tolist())

            if isinstance(optimizer, torch.optim.SGD):
                lr = init_lr * (1.0 - trained_batches / batch_count)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            print('Epoch %d avg loss: %.5f' % (epoch, np.mean(loss_log)))
        return ns_model.u_embeddings.weight.detach().cpu().numpy()

    def predict(self, batch):

        return

    def calculate_loss(self, batch):


        return


