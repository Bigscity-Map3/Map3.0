import numpy as np
from libcity.model.poi_representation.w2v import *
from libcity.model.abstract_model import AbstractModel
from logging import getLogger
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

        self.u_embeddings = nn.Embedding(num_vocab + 1, embed_dimension, sparse=True)
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
        # print('=======')
        # print(pos_u.shape, pos_v.shape, neg_v.shape, user.shape, weekday.shape, neg_ne.shape, neg_nn.shape)
        embed_u = self.u_embeddings(pos_u)  # (batch_size, embed_size)
        embed_week = self.week_embeddings(weekday)  # (batch_size, week_embed_size)
        embed_cat = torch.cat([embed_u, embed_week], dim=-1)  # (batch_size, embed_size + week_embed_size)
        # print(embed_u.shape, embed_week.shape, embed_cat.shape)

        embed_v = self.v_embeddings(pos_v)  # (batch_size, window_size * 2, embed_size + week_embed_size)
        # print(embed_v.shape)
        score = torch.mul(embed_cat.unsqueeze(1), embed_v).squeeze()
        # (batch_size, window_size * 2, embed_size + week_embed_size)
        # print(score.shape)
        score = torch.sum(score, dim=-1)  # (batch_size, window_size * 2)
        # print(score.shape)
        score = F.logsigmoid(score)

        neg_embed_v = self.v_embeddings(neg_v)  # (batch_size, num_neg, embed_size + week_embed_size)
        # print(neg_embed_v.shape)
        neg_score = torch.bmm(neg_embed_v, embed_cat.unsqueeze(2)).squeeze()  # (batch_size, num_neg)
        # print(neg_score.shape)
        neg_score = F.logsigmoid(-1 * neg_score)

        embed_user = self.user_embeddings(user)  # (batch_size, embed_size + week_embed_size)
        neg_embed_ne = self.v_embeddings(neg_ne)  # (batch_size, N, embed_size + week_embed_size)
        neg_embed_nn = self.v_embeddings(neg_nn)
        # print(embed_user.shape, neg_embed_ne.shape, neg_embed_nn.shape)

        neg_ne_score = torch.bmm(embed_cat.unsqueeze(1) - neg_embed_ne, embed_user.unsqueeze(2)).squeeze()
        # (batch_size, N)
        neg_ne_score = F.logsigmoid(neg_ne_score)
        neg_nn_score = torch.bmm(embed_cat.unsqueeze(1) - neg_embed_nn, embed_user.unsqueeze(2)).squeeze()
        neg_nn_score = F.logsigmoid(neg_nn_score)
        # print(neg_ne_score.shape, neg_nn_score.shape)
        # print(torch.sum(score), torch.sum(neg_score), torch.sum(neg_ne_score), torch.sum(neg_nn_score))

        return -1 * (self.alpha * (torch.sum(score) + torch.sum(neg_score)) +
                     self.beta * (torch.sum(neg_ne_score) + torch.sum(neg_nn_score)))

    def static_embed(self):
        return self.u_embeddings.weight[:self.num_vocab].detach().cpu().numpy()

    def calculate_loss(self, batch):
        batch_count, pos_u, pos_v, neg_v, user, week, neg_ne, neg_nn = batch
        return self.forward(pos_u, pos_v, neg_v, user, week, neg_ne, neg_nn)


def dis(lat1, lng1, lat2, lng2):
    lng1, lat1, lng2, lat2 = map(radians, [lng1, lat1, lng2, lat2])
    dlng = lng2 - lng1

    c = sin(lat1) * sin(lat2) + cos(lat1) * cos(lat2) * cos(dlng)
    r = 6371  # km
    if c > 1:
        c = 1
    return int(r * acos(c))


class TeaserData(SkipGramData):
    def __init__(self, users, sentences, weeks, coordinates, num_ne, num_nn, indi_context, distance_threshold=5,
                 sample=1e-3):
        """
        @param sentences: all users' full trajectories, shape (num_users, seq_len)
        @param weeks: weekday indices corresponding to the sentences.
        @param coordinates: coordinates of all locations, shape (num_locations, 3), each row is (loc_index, lat, lng)
        """
        super().__init__(sentences, sample)
        self.num_ne = num_ne
        self.num_nn = num_nn
        self.indi_context = indi_context

        self.users = users
        self.weeks = weeks
        all_locations = set(coordinates[:, 0].astype(int).tolist())

        # A dict mapping one location index to its non-neighbor locations.
        self.non_neighbors = {}
        for coor_row in coordinates:
            loc_index = int(coor_row[0])

            # distance = coor_row[1:].reshape(1, 2) - coordinates[:, 1:]  # (num_loc, 2)
            # distance = np.sqrt(np.power(distance[:, 0], 2) + np.power(distance[:, 1], 2))  # (num_loc)

            distance = np.empty(len(coordinates))
            for i, coordinate in enumerate(coordinates):
                distance[i] = dis(coor_row[1], coor_row[2], coordinate[1], coordinate[2])

            non_neighbor_indices = coordinates[:, 0][np.argwhere(distance > distance_threshold)].reshape(-1).astype(int)
            if non_neighbor_indices.shape[0] == 0:
                non_neighbor_indices = np.array([len(all_locations)], dtype=int)
            self.non_neighbors[loc_index] = non_neighbor_indices

        # A dict mapping one user index to its all unvisited locations.
        self.unvisited = {}
        for user, visited in zip(users, sentences):
            user = int(user)
            user_unvisited = all_locations - set(visited)
            self.unvisited[user] = user_unvisited & self.unvisited.get(user, all_locations)

    def gen_pos_pairs(self, window_size):
        pos_pairs = []
        for user, sentence, week in zip(self.users, self.sentences, self.weeks):
            for i in range(0, len(sentence) - (2 * window_size + 1)):
                target = sentence[i + window_size]
                target_week = 0 if week[i + window_size] in range(5) else 1
                context = sentence[i:i + window_size] + sentence[i + window_size + 1:i + 2 * window_size + 1]
                sample_ne = self.sample_unvisited(user, num_neg=self.num_ne)
                sample_nn = self.sample_non_neighbor(target, num_neg=self.num_nn)
                if self.indi_context:
                    pos_pairs += [[user, target, target_week, [c], sample_ne, sample_nn] for c in context]
                else:
                    pos_pairs.append([user, target, target_week, context, sample_ne, sample_nn])
        return pos_pairs

    def sample_unvisited(self, user, num_neg):
        return np.random.choice(np.array(list(self.unvisited[user])), size=num_neg).tolist()

    def sample_non_neighbor(self, target, num_neg):
        return np.random.choice(self.non_neighbors[target], size=num_neg).tolist()
