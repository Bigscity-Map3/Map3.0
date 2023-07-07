import numpy as np
import os
import pandas as pd
from libcity.data.dataset.poi_representation_dataset import PoiRepresentationDataset


class GeoTeaserDataset(PoiRepresentationDataset):
    def __init__(self, config):
        self.config = config
        self.dataset = self.config.get('dataset', '')
        self.data_path = './raw_data/' + self.dataset + '/'
        self.geo_file = self.config.get('geo_file', self.dataset)
        self.rel_file = self.config.get('rel_file', self.dataset)
        self.dyna_file = self.config.get('dyna_file', self.dataset)
        if not os.path.exists('./libcity/cache/GEOTEASER_{}'.format(self.dataset)):
            os.mkdir('./libcity/cache/GEOTEASER_{}'.format(self.dataset))
        super().__init__(config)


        self.teaser_num_ne = self.config.get('teaser_num_ne',0)
        self.teaser_num_nn = self.config.get('teaser_num_nn',0)
        self.teaser_indi_context = self.config.get('teaser_indi_context', False)
        self.teaser_beta = self.config.get('teaser_beta', 0.0)
        self.teaser_week_embed_size = self.config.get('teaser_week_embed_size', 0)
        self.users = self.embed_train_entitys
        self.distance_threshold = self.config.get("distance_threshold", 0.2)
        self.sample = self.config.get("sample", 1e-3)

        self.coordinates  = self.traj_poi[['poi_id','lat','lng']].drop_duplicates('poi_id').to_numpy()
        self.all_locations = set(self.coordinates[:, 0].astype(int).tolist())
        self.gen_non_neighbor()
        self.gen_unvisited_loc()


    def gen_non_neighbor(self):
        self.non_neighbors = {}
        for coor_row in self.coordinates:
            loc_index = int(coor_row[0])
            distance = coor_row[1:].reshape(1, 2) - self.coordinates[:, 1:]  # (num_loc, 2)
            distance = np.sqrt(np.power(distance[:, 0], 2) + np.power(distance[:, 1], 2))  # (num_loc)
            non_neighbor_indices = self.coordinates[:, 0][np.argwhere(distance > self.distance_threshold)].reshape(-1).astype(int)
            if non_neighbor_indices.shape[0] == 0:
                non_neighbor_indices = np.array([len(self.all_locations)], dtype=int)
            self.non_neighbors[loc_index] = non_neighbor_indices

    def gen_unvisited_loc(self):
        self.unvisited = {}
        for user, visited in zip(self.users, self.sentences):
            user_unvisited = self.all_locations - set(visited)
            self.unvisited[user] = user_unvisited & self.unvisited.get(user, self.all_locations)


    def gen_pos_pairs(self, window_size):
        pos_pairs = []
        for user, sentence, week in zip(self.users, self.sentences, self.embed_train_weekdays):
            for i in range(0, len(sentence) - (2 * window_size + 1)):
                target = sentence[i+window_size]
                target_week = 0 if week[i+window_size] in range(5) else 1
                context = sentence[i:i+window_size] + sentence[i+window_size+1:i+2*window_size+1]
                sample_ne = self.sample_unvisited(user, num_neg=self.teaser_num_ne)
                sample_nn = self.sample_non_neighbor(target, num_neg=self.teaser_num_nn)
                if self.teaser_indi_context:
                    pos_pairs += [[user, target, target_week, [c], sample_ne, sample_nn] for c in context]
                else:
                    pos_pairs.append([user, target, target_week, context, sample_ne, sample_nn])
        return pos_pairs

    def sample_unvisited(self, user, num_neg):
        return np.random.choice(np.array(list(self.unvisited[user])), size=(num_neg)).tolist()

    def sample_non_neighbor(self, target, num_neg):
        return np.random.choice(self.non_neighbors[target], size=(num_neg)).tolist()

    def get_data_feature(self):
        self.user_num = self.traj_poi['entity_id'].value_counts().count()
        self.poi_num = self.traj_poi['poi_id'].value_counts().count()
        self.pos_pairs = self.gen_pos_pairs(self.w2v_window_size)
        return { "num_loc": self.poi_num,
                "user_num": self.user_num,
                "pos_pairs": self.pos_pairs,
                "sample_table": self.sample_table}
