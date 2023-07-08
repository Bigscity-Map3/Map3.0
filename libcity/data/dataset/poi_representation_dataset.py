import os
from libcity.data.dataset.traffic_representation_dataset import TrafficRepresentationDataset
import numpy as np
import pandas as pd
from collections import Counter

class PoiRepresentationDataset(TrafficRepresentationDataset):
    def __init__(self, config):
        self.config = config
        self.dataset = self.config.get('dataset', '')
        self.data_path = './raw_data/' + self.dataset + '/'
        self.label_data_path = './raw_data/' + self.dataset + '/label_data/'
        self.geo_file = self.config.get('geo_file', self.dataset)
        self.rel_file = self.config.get('rel_file', self.dataset)
        self.dyna_file = self.config.get('dyna_file', self.dataset)
        self.split_days= self.config.get('split_days', [[10,11,12], [13], [14]])
        self.w2v_window_size = self.config.get('w2v_window_size', 1)
        self.edge_index = []
        assert os.path.exists(self.data_path + self.geo_file + '.geo')
        assert os.path.exists(self.data_path + self.rel_file + '.rel')
        assert os.path.exists(self.data_path + self.dyna_file + '.dyna')
        super().__init__(config)


        self.traj_process_poi()

        self.embed_train_entitys, self.embed_train_sentences, self.embed_train_weekdays, \
        self.embed_train_timestamp, self._length = zip(*self.gen_sequence(min_len=self.w2v_window_size * 2 + 1))

        # Word2VecDataset and SkipgramDataset properties and functions
        self.word_freq = self.get_token_freq(self.embed_train_sentences)
        self.sentences = self.embed_train_sentences
        self.sample_table = self.gen_neg_sample_table(self.word_freq)





    def traj_process_poi(self):
        dynafile = pd.read_csv(self.data_path + self.dyna_file + '.dyna',nrows=64)
        traj_df = dynafile[['time', 'entity_id', 'geo_id']]

        def gen_map_index(df, column, offset=0):
            index_map = {origin: index + offset
                         for index, origin in enumerate(df[column].drop_duplicates())}
            return index_map

        def gen_map(x):
            map = {row['origin_id']: row['destination_id'] for index, row in x.iterrows()}
            return map

        self.user2index = gen_map_index(traj_df, 'entity_id')

        traj_df['entity_id'] = traj_df['entity_id'].map(self.user2index)
        traj_df['geo_id'] = traj_df['geo_id'].map(gen_map(self.road2poi))
        traj_df.rename(columns={'geo_id': 'poi_id'}, inplace=True)
        traj_df.dropna(inplace=True)
        self.poi2index = gen_map_index(traj_df, 'poi_id')
        traj_df['poi_id'] = traj_df['poi_id'].map(self.poi2index)
        self.pois_df = pd.DataFrame(self.geofile[self.geofile['traffic_type'] == 'poi'])
        self.pois_df = self.pois_df[['geo_id', 'coordinates']]
        self.pois_df['lng'] = self.pois_df['coordinates'].apply(lambda x: eval(x)[0])
        self.pois_df['lat'] = self.pois_df['coordinates'].apply(lambda x: eval(x)[1])
        self.pois_df.rename(columns={'geo_id': 'poi_id'}, inplace=True)
        self.pois_df['poi_id'] = self.pois_df['poi_id'].map(self.poi2index)
        self.traj_poi = pd.merge(traj_df, self.pois_df, on='poi_id', how='inner')



    def gen_sequence(self, min_len=0, select_days=None, include_delta=False):
        data = self.traj_poi.copy()

        data['datetime'] = pd.to_datetime(data['time'],format='%Y-%m-%d %H:%M:%S')
        data['day'] = data['datetime'].dt.day
        if select_days is not None:
            data = data[data['day'].isin(self.split_days[select_days])]
        data['weekday'] = data['datetime'].dt.weekday
        data['timestamp']  = data['datetime'].apply(lambda x: x.timestamp())

        if include_delta:
            data['time_delta'] = data['timestamp'].shift(-1) - data['timestamp']
            coor_delta = (data[['lng', 'lat']].shift(-1) - data[['lng', 'lat']]).to_numpy()
            data['dist'] = np.sqrt((coor_delta ** 2).sum(-1))

        seq_set = []
        for (entity, day), group in data.groupby(['entity_id','day']):
            if group.shape[0] < min_len:
                continue
            one_set =  [entity, group['poi_id'].tolist(), group['weekday'].astype(int).tolist(),
                       group['timestamp'].astype(int).tolist(), group.shape[0]]

            if include_delta:
                one_set += [[0] + group['time_delta'].iloc[:-1].tolist(),
                            [0] + group['dist'].iloc[:-1].tolist(),
                            group['lat'].tolist(),
                            group['lng'].tolist()]

            seq_set.append(one_set)
        return seq_set

    def gen_span(self, span_len, select_day=None):
        data = pd.DataFrame(self.traj_poi, copy=True)
        data['day'] = data['datetime'].dt.day
        if select_day is not None:
            data = data[data['day'].isin(select_day)]
        data['weekday'] = data['datetime'].dt.weekday
        data['timestamp'] = data['datetime'].apply(lambda x: x.timestamp())

        seq_set = []
        for (entity_id, day), group in data.groupby(['entity_id', 'day']):
            for i in range(group.shape[0] - span_len + 1):
                select_group = group.iloc[i:i + span_len]
                one_set = [entity_id, select_group['poi_id'].tolist(), select_group['weekday'].astype(int).tolist(),
                           select_group['timestamp'].astype(int).tolist()]
                seq_set.append(one_set)
        return seq_set


    def get_data(self):
        return None,None,None

    def get_token_freq(self, sentences):
        freq = Counter()
        for sentence in sentences:
            freq.update(sentence)
        freq = np.array(sorted(freq.items()))
        return freq

    def gen_neg_sample_table(self, freq_array, sample_table_size=1e8, clip_ratio=1e-3):
        sample_table = []
        pow_freq = freq_array[:, 1] ** 0.75

        words_pow = pow_freq.sum()
        ratio = pow_freq / words_pow
        ratio = np.clip(ratio, a_min=0, a_max=clip_ratio)
        ratio = ratio / ratio.sum()

        count = np.round(ratio * sample_table_size)
        for word_index, c in enumerate(count):
            sample_table += [freq_array[word_index, 0]] * int(c)
        sample_table = np.array(sample_table)
        return sample_table

    def gen_pos_pairs(self, window_size):
        pos_pairs = []
        for sentence in self.sentences:
            for i in range(0, len(sentence) - (2 * window_size + 1)):
                target = sentence[i + window_size]
                context = sentence[i:i + window_size] + sentence[i + window_size + 1:i + 2 * window_size + 1]
                pos_pairs.append([target, context])
        return pos_pairs

    def get_neg_v_sampling(self, batch_size, num_neg):
        neg_v = np.random.choice(self.sample_table, size=(batch_size, num_neg))
        return neg_v


