import math
import numpy as np
from itertools import zip_longest
import torch
import torch.nn as nn
from sklearn.utils import shuffle

from libcity.model.poi_representation.Word2Vec import HS
from libcity.model.abstract_model import AbstractModel


def next_batch(data, batch_size):
    data_length = len(data)
    num_batches = math.ceil(data_length / batch_size)
    for batch_index in range(num_batches):
        start_index = batch_index * batch_size
        end_index = min((batch_index + 1) * batch_size, data_length)
        yield data[start_index:end_index]


class POI2Vec(HS, AbstractModel):
    def __init__(self, num_vocab, num_inner_nodes, embed_dimension):
        self.__dict__.update(locals())
        super().__init__(num_vocab, embed_dimension)

        self.w_embeddings = nn.Embedding(num_inner_nodes, embed_dimension, padding_idx=0, sparse=True)
        self.w_embeddings.weight.data.uniform_(-0, 0)

    def forward(self, pos_u, pos_w, neg_w, **kwargs):
        """
        :param pos_u: positive input tokens, shape (batch_size, window_size * 2)
        :param prob: probabilities of each route, shape (batch_size, 4)
        :return: loss value of this batch.
        """
        pos_score, neg_score = super().forward(pos_u, pos_w, neg_w, sum=False)
        pos_score, neg_score = (-1 * (item.sum(1) * kwargs['prop']).sum() for item in (pos_score, neg_score))
        return pos_score + neg_score

    def train_p2v(p2v_model, dataset, window_size, batch_size, num_epoch, init_lr, optim_class, device):
        p2v_model = p2v_model.to(device)
        optimizer = optim_class(p2v_model.parameters(), lr=init_lr)

        train_set = dataset.gen_path_pairs(window_size)
        trained_batches = 0
        batch_count = math.ceil(num_epoch * len(train_set) / batch_size)

        for epoch in range(num_epoch):
            loss_log = []
            for pair_batch in next_batch(shuffle(train_set), batch_size):
                flatten_batch = []
                for row in pair_batch:
                    flatten_batch += [[row[0], p, n, pr] for p, n, pr in zip(*row[1:])]

                context, pos_pairs, neg_pairs, prop = zip(*flatten_batch)
                context = torch.tensor(context).long().to(device)
                pos_pairs, neg_pairs = (
                torch.tensor(list(zip_longest(*item, fillvalue=0))).long().to(device).transpose(0, 1)
                for item in (pos_pairs, neg_pairs))  # (batch_size, longest)
                prop = torch.tensor(prop).float().to(device)

                optimizer.zero_grad()
                loss = p2v_model(context, pos_pairs, neg_pairs, prop=prop)
                loss.backward()
                optimizer.step()
                trained_batches += 1
                loss_log.append(loss.detach().cpu().numpy().tolist())

            if isinstance(optimizer, torch.optim.SGD):
                lr = init_lr * (1.0 - trained_batches / batch_count)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            print('Epoch %d avg loss: %.5f' % (epoch, np.mean(loss_log)))
        return p2v_model.u_embeddings.weight.detach().cpu().numpy()

    def predict(self, batch):



        return

    def calculate_loss(self, batch):


        return
