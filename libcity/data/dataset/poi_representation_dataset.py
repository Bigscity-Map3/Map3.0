import math
import os
import ast
import torch
import sklearn
import numpy as np
import pandas as pd
from logging import getLogger
from collections import Counter
from random import *
from itertools import zip_longest

from libcity.data.dataset import AbstractDataset
from libcity.model.poi_representation.utils import next_batch, init_seed
from libcity.model.poi_representation.tale import TaleData
from libcity.model.poi_representation.poi2vec import P2VData
from libcity.model.poi_representation.teaser import TeaserData
from libcity.model.poi_representation.w2v import SkipGramData


class POIRepresentationDataLoader:

    def __init__(self, config, data_feature, data):
        self.config = config
        self.data_feature = data_feature
        self.data = data
        self.batch_size = self.config.get('batch_size', 64)
        self.device = self.config.get('device', torch.device('cpu'))
        self.model_name = self.config.get('model')
        self.w2v_window_size = self.config.get('w2v_window_size', 1)
        self.skipgram_neg = self.config.get('skipgram_neg', 5)
        w2v_data = self.data_feature.get('w2v_data')
        self.embed_train_users, self.embed_train_sentences, self.embed_train_weekdays, \
        self.embed_train_timestamp, _length = zip(*w2v_data)
        seed = self.config.get('seed', 0)
        init_seed(seed)

    def next_batch(self):
        pass


class CTLEDataLoader(POIRepresentationDataLoader):

    def gen_random_mask(self, src_valid_lens, src_len, mask_prop):
        """
        @param src_valid_lens: valid length of sequence, shape (batch_size)
        """
        # all_index = np.arange((batch_size * src_len)).reshape(batch_size, src_len)
        # all_index = shuffle_along_axis(all_index, axis=1)
        # mask_count = math.ceil(mask_prop * src_len)
        # masked_index = all_index[:, :mask_count].reshape(-1)
        # return masked_index
        index_list = []
        for batch, l in enumerate(src_valid_lens):
            mask_count = torch.ceil(mask_prop * l).int()
            masked_index = torch.randperm(l)[:mask_count]
            masked_index += src_len * batch
            index_list.append(masked_index)
        return torch.cat(index_list).long().to(src_valid_lens.device)

    def gen_casual_mask(self, seq_len, include_self=True):
        """
        Generate a casual mask which prevents i-th output element from
        depending on any input elements from "the future".
        Note that for PyTorch Transformer model, sequence mask should be
        filled with -inf for the masked positions, and 0.0 else.

        :param seq_len: length of sequence.
        :return: a casual mask, shape (seq_len, seq_len)
        """
        if include_self:
            mask = 1 - torch.triu(torch.ones(seq_len, seq_len)).transpose(0, 1)
        else:
            mask = 1 - torch.tril(torch.ones(seq_len, seq_len)).transpose(0, 1)
        return mask.bool()

    def next_batch(self):
        mask_prop = self.config.get('mask_prop', 0.2)
        num_vocab = self.data_feature.get('num_loc')
        user_ids, src_tokens, src_weekdays, src_ts, src_lens = zip(*self.data)
        for batch in next_batch(sklearn.utils.shuffle(list(zip(src_tokens, src_weekdays, src_ts, src_lens))),
                                batch_size=self.batch_size):
            # Value filled with num_loc stands for masked tokens that shouldn't be considered.
            src_batch, _, src_t_batch, src_len_batch = zip(*batch)
            src_batch = np.transpose(np.array(list(zip_longest(*src_batch, fillvalue=num_vocab))))
            src_t_batch = np.transpose(np.array(list(zip_longest(*src_t_batch, fillvalue=0))))

            src_batch = torch.tensor(src_batch).long().to(self.device)
            src_t_batch = torch.tensor(src_t_batch).float().to(self.device)
            hour_batch = (src_t_batch % (24 * 60 * 60) / 60 / 60).long()

            batch_len, src_len = src_batch.size(0), src_batch.size(1)
            src_valid_len = torch.tensor(src_len_batch).long().to(self.device)

            mask_index = self.gen_random_mask(src_valid_len, src_len, mask_prop=mask_prop)

            src_batch = src_batch.reshape(-1)
            hour_batch = hour_batch.reshape(-1)
            origin_tokens = src_batch[mask_index]  # (num_masked)
            origin_hour = hour_batch[mask_index]

            # Value filled with num_loc+1 stands for special token <mask>.
            masked_tokens = \
                src_batch.index_fill(0, mask_index, num_vocab + 1).reshape(batch_len, -1)
            # (batch_size, src_len)

            yield origin_tokens, origin_hour, masked_tokens, src_t_batch, mask_index


class HierDataLoader(POIRepresentationDataLoader):

    def next_batch(self):
        user_ids, src_tokens, src_weekdays, src_ts, src_lens = zip(*self.data)
        for batch in next_batch(sklearn.utils.shuffle(list(zip(src_tokens, src_weekdays, src_ts, src_lens))),
                                batch_size=self.batch_size):
            src_token, src_weekday, src_t, src_len = zip(*batch)
            src_token, src_weekday = [
                torch.from_numpy(np.transpose(np.array(list(zip_longest(*item, fillvalue=0))))).long().to(self.device)
                for item in (src_token, src_weekday)]
            src_t = torch.from_numpy(np.transpose(np.array(list(zip_longest(*src_t, fillvalue=0))))).float().to(
                self.device)
            src_len = torch.tensor(src_len).long().to(self.device)

            src_hour = (src_t % (24 * 60 * 60) / 60 / 60).long()
            src_duration = ((src_t[:, 1:] - src_t[:, :-1]) % (24 * 60 * 60) / 60 / 60).long()
            src_duration = torch.clamp(src_duration, 0, 23)
            yield src_token, src_weekday, src_hour, src_duration, src_len


class TaleDataLoader(POIRepresentationDataLoader):

    def __init__(self, config, data_feature, data):
        super().__init__(config, data_feature, data)
        if self.model_name == 'Tale':
            tale_slice = config.get('slice', 60)
            tale_span = config.get('span', 0)
            tale_indi_context = config.get('indi_context', True)
            self.dataset = TaleData(self.embed_train_sentences, self.embed_train_timestamp, tale_slice, tale_span,
                                    indi_context=tale_indi_context)
        elif self.model_name == 'POI2Vec':
            poi2vec_theta = data_feature.get('theta', 0.1)
            poi2vec_indi_context = config.get('indi_context', False)
            id2coor_df = data_feature.get('id2coor_df')
            self.dataset = P2VData(self.embed_train_sentences, id2coor_df, theta=poi2vec_theta,
                                   indi_context=poi2vec_indi_context)
        self.train_set = self.dataset.get_path_pairs(self.w2v_window_size)

    def next_batch(self):
        embed_epoch = self.config.get('embed_epoch', 5)
        batch_count = math.ceil(embed_epoch * len(self.train_set) / self.batch_size)
        for pair_batch in next_batch(sklearn.utils.shuffle(self.train_set), self.batch_size):
            flatten_batch = []
            for row in pair_batch:
                flatten_batch += [[row[0], p, n, pr] for p, n, pr in zip(*row[1:])]

            context, pos_pairs, neg_pairs, prop = zip(*flatten_batch)
            context = torch.tensor(context).long().to(self.device)
            pos_pairs, neg_pairs = (
                torch.tensor(list(zip_longest(*item, fillvalue=0))).long().to(self.device).transpose(0, 1)
                for item in (pos_pairs, neg_pairs)
            )  # (batch_size, longest)
            prop = torch.tensor(prop).float().to(self.device)
            yield batch_count, context, pos_pairs, neg_pairs, prop


class TeaserDataLoader(POIRepresentationDataLoader):

    def __init__(self, config, data_feature, data):
        super().__init__(config, data_feature, data)
        teaser_num_ne = config.get('num_ne', 0)  # (number of unvisited locations)
        teaser_num_nn = config.get('num_nn', 0)  # (number of non-neighbor locations)
        teaser_indi_context = config.get('indi_context', False)
        coor_mat = data_feature.get('coor_mat')
        self.dataset = TeaserData(self.embed_train_users, self.embed_train_sentences, self.embed_train_weekdays,
                                  coor_mat,
                                  num_ne=teaser_num_ne, num_nn=teaser_num_nn,
                                  indi_context=teaser_indi_context)
        self.pos_pairs = self.dataset.gen_pos_pairs(self.w2v_window_size)

    def next_batch(self):
        num_neg = self.skipgram_neg
        embed_epoch = self.config.get('embed_epoch', 5)
        batch_count = math.ceil(embed_epoch * len(self.pos_pairs) / self.batch_size)
        for pair_batch in next_batch(sklearn.utils.shuffle(self.pos_pairs), self.batch_size):
            neg_v = self.dataset.get_neg_v_sampling(len(pair_batch), num_neg)
            neg_v = torch.tensor(neg_v).long().to(self.device)

            user, pos_u, week, pos_v, neg_ne, neg_nn = zip(*pair_batch)
            user, pos_u, week, pos_v, neg_ne, neg_nn = (torch.tensor(item).long().to(self.device)
                                                        for item in (user, pos_u, week, pos_v, neg_ne, neg_nn))
            yield batch_count, pos_u, pos_v, neg_v, user, week, neg_ne, neg_nn


class SkipGramDataLoader(POIRepresentationDataLoader):

    def __init__(self, config, data_feature, data):
        super().__init__(config, data_feature, data)
        self.dataset = SkipGramData(self.embed_train_sentences)
        window_size = self.w2v_window_size
        self.pos_pairs = self.dataset.gen_pos_pairs(window_size)

    def next_batch(self):
        num_neg = self.skipgram_neg
        embed_epoch = self.config.get('embed_epoch', 5)
        batch_count = math.ceil(embed_epoch * len(self.pos_pairs) / self.batch_size)
        for pair_batch in next_batch(sklearn.utils.shuffle(self.pos_pairs), self.batch_size):
            neg_v = self.dataset.get_neg_v_sampling(len(pair_batch), num_neg)
            neg_v = torch.tensor(neg_v).long().to(self.device)

            pos_u, pos_v = zip(*pair_batch)
            pos_u, pos_v = (torch.tensor(item).long().to(self.device)
                            for item in (pos_u, pos_v))
            yield batch_count, pos_u, pos_v, neg_v


def get_dataloader(config, data_feature, data):
    model_name = config.get('model')
    if model_name in ['CTLE']:
        return CTLEDataLoader(config, data_feature, data)
    if model_name in ['Hier']:
        return HierDataLoader(config, data_feature, data)
    if model_name in ['Tale', 'POI2Vec']:
        return TaleDataLoader(config, data_feature, data)
    if model_name in ['Teaser']:
        return TeaserDataLoader(config, data_feature, data)
    if model_name in ['SkipGram', 'CBOW']:
        return SkipGramDataLoader(config, data_feature, data)
    return None


class POIRepresentationDataset(AbstractDataset):

    def __init__(self, config):
        """
        @param raw_df: raw DataFrame containing all mobile signaling records.
            Should have at least three columns: user_id, latlng and datetime.
        @param coor_df: DataFrame containing coordinate information.
            With an index corresponding to latlng, and two columns: lat and lng.
        """
        self.config = config
        self._logger = getLogger()
        self._logger.info('Starting load data ...')
        self.dataset = self.config.get('dataset')
        self.test_scale = self.config.get('test_scale', 0.1)
        self.min_len = self.config.get('min_len', 5)  # 轨迹最短长度
        self.min_frequency = self.config.get('min_frequency', 10)  # POI 最小出现次数
        self.min_poi_cnt = self.config.get('min_poi_cnt', 5)  # 用户最少拥有 POI 数
        self.pre_len = self.config.get('pre_len', 3)  # 预测后 pre_len 个 POI
        self.w2v_window_size = self.config.get('w2v_window_size', 1)
        self.data_path = './raw_data/' + self.dataset + '/'
        if not os.path.exists(self.data_path):
            raise ValueError("Dataset {} not exist! Please ensure the path "
                             "'./raw_data/{}/' exist!".format(self.dataset, self.dataset))
        self.usr_file = self.config.get('usr_file', self.dataset)
        self.geo_file = self.config.get('geo_file', self.dataset)
        self.dyna_file = self.config.get('dyna_file', self.dataset)
        # if os.path.exists(os.path.join(self.data_path, self.usr_file + '.usr')):
        #     self._load_usr()
        # else:
        #     raise ValueError('Not found .usr file!')
        if os.path.exists(os.path.join(self.data_path, self.geo_file + '.geo')):
            self._load_geo()
        else:
            raise ValueError('Not found .geo file!')
        if os.path.exists(os.path.join(self.data_path, self.dyna_file + '.dyna')):
            self._load_dyna()
        else:
            raise ValueError('Not found .dyna file!')
        self._split_days()
        self._logger.info('User num: {}'.format(self.num_user))
        self._logger.info('Location num: {}'.format(self.num_loc))
        self.con = self.config.get('con', 7e8)
        self.theta = self.num_user * self.num_loc / self.con
        self._init_data_feature()

    def _load_geo(self):
        geo_df = pd.read_csv(os.path.join(self.data_path, self.geo_file + '.geo'))
        geo_df = geo_df[geo_df['type'] == 'Point']
        poi_list = geo_df['coordinates'].tolist()
        lng_list = []
        lat_list = []
        for s in poi_list:
            lng, lat = ast.literal_eval(s)
            lng_list.append(lng)
            lat_list.append(lat)
        lng_col = pd.Series(lng_list, name='lng')
        lat_col = pd.Series(lat_list, name='lat')
        idx_col = pd.Series(list(range(len(geo_df))), name='geo_id')
        type_name = self.config.get('poi_type_name', None)
        if type_name is not None:
            category_list=list(geo_df[type_name].drop_duplicates())
            c2i={name:i for i,name in enumerate(category_list)}
            cid_list=[]
            for name in list(geo_df[type_name]):
                cid_list.append(c2i[name])
            cid_list=pd.Series(cid_list,name='category')
            self.coor_df = pd.concat([idx_col, lat_col, lng_col, cid_list], axis=1)
        else:
            self.coor_df = pd.concat([idx_col, lat_col, lng_col], axis=1)

    def _load_dyna(self):
        dyna_df = pd.read_csv(os.path.join(self.data_path, self.dyna_file + '.dyna'))
        dyna_df = dyna_df[dyna_df['type'] == 'trajectory']
        dyna_df = dyna_df.merge(self.coor_df, left_on='location', right_on='geo_id', how='left')
        dyna_df.rename(columns={'time': 'datetime'}, inplace=True)
        dyna_df.rename(columns={'location': 'loc_index'}, inplace=True)
        dyna_df.rename(columns={'entity_id': 'user_index'}, inplace=True)
        self.df = dyna_df[['user_index', 'loc_index', 'datetime', 'lat', 'lng']]
        user_counts = self.df['user_index'].value_counts()
        self.df = self.df[self.df['user_index'].isin(user_counts.index[user_counts >= self.min_poi_cnt])]
        loc_counts = self.df['loc_index'].value_counts()
        self.coor_df = self.coor_df[self.coor_df['geo_id'].isin(loc_counts.index[loc_counts >= self.min_frequency])]
        self.df = self.df[self.df['loc_index'].isin(loc_counts.index[loc_counts >= self.min_frequency])]
        loc_index_map = self.gen_index_map(self.coor_df, 'geo_id')
        self.coor_df['geo_id'] = self.coor_df['geo_id'].map(loc_index_map)
        self.df['loc_index'] = self.df['loc_index'].map(loc_index_map)
        user_index_map = self.gen_index_map(self.df, 'user_index')
        self.df['user_index'] = self.df['user_index'].map(user_index_map)
        self.num_user = len(user_index_map)
        self.num_loc = self.coor_df.shape[0]

    def _split_days(self):
        data = pd.DataFrame(self.df, copy=True)
        data['datetime'] = pd.to_datetime(data["datetime"])
        data['day'] = data['datetime'].dt.day
        days = data['day'].drop_duplicates().to_list()
        if len(days) <= 1:
            raise ValueError('Dataset contains only one day!')
        days.sort()
        test_count = max(1, min(math.ceil(len(days) * self.test_scale), len(days)))
        self.split_days = [days[:-test_count], days[-test_count:]]
        self._logger.info('Days for train: {}'.format(self.split_days[0]))
        self._logger.info('Days for test: {}'.format(self.split_days[1]))

    def _load_usr(self):
        pass

    def gen_index_map(self, df, column, offset=0):
        index_map = {origin: index + offset
                     for index, origin in enumerate(df[column].drop_duplicates())}
        return index_map

    def _init_data_feature(self):
        self.max_seq_len = Counter(self.df['user_index'].to_list()).most_common(1)[0][1]
        self.train_set = self.gen_sequence(min_len=self.pre_len + 1, select_days=0, include_delta=True)
        self.test_set = self.gen_sequence(min_len=self.pre_len + 1, select_days=1, include_delta=True)
        self.w2v_data = self.gen_sequence(min_len=self.w2v_window_size * 2 + 1, select_days=0)
        self.coor_mat = self.df[['loc_index', 'lat', 'lng']].drop_duplicates('loc_index').to_numpy()
        self.id2coor_df = self.df[['loc_index', 'lat', 'lng']].drop_duplicates('loc_index'). \
            set_index('loc_index').sort_index()

    def get_data(self):
        """
        返回数据的DataLoader，包括训练数据、测试数据、验证数据

        Returns:
            tuple: tuple contains:
                train_dataloader: Dataloader composed of Batch (class) \n
                eval_dataloader: Dataloader composed of Batch (class) \n
                test_dataloader: Dataloader composed of Batch (class)
        """
        return get_dataloader(self.config, self.get_data_feature(), self.gen_sequence(select_days=0)), None, None

    def get_data_feature(self):
        """
        返回一个 dict，包含数据集的相关特征

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        return {
            "max_seq_len": self.max_seq_len,
            "num_loc": self.num_loc,
            "num_user": self.num_user,
            "train_set": self.train_set,
            "test_set": self.test_set,
            "w2v_data": self.w2v_data,
            "coor_mat": self.coor_mat,
            "id2coor_df": self.id2coor_df,
            "theta" : self.theta,
            "coor_df" : self.coor_df
        }

    def gen_sequence(self, min_len=None, select_days=None, include_delta=False):
        """
        Generate moving sequence from original trajectories.

        @param min_len: minimal length of sentences.
        @param select_day: list of day to select, set to None to use all days.
        """
        if min_len is None:
            min_len = self.min_len
        data = pd.DataFrame(self.df, copy=True)
        data['datetime'] = pd.to_datetime(data["datetime"])
        data['day'] = data['datetime'].dt.day
        if select_days is not None:
            data = data[data['day'].isin(self.split_days[select_days])]
        data['weekday'] = data['datetime'].dt.weekday
        data['timestamp'] = data['datetime'].apply(lambda x: x.timestamp())

        if include_delta:
            data['time_delta'] = data['timestamp'].shift(-1) - data['timestamp']
            coor_delta = (data[['lng', 'lat']].shift(-1) - data[['lng', 'lat']]).to_numpy()
            data['dist'] = np.sqrt((coor_delta ** 2).sum(-1))

        seq_set = []
        for (user_index, day), group in data.groupby(['user_index', 'day']):
            if group.shape[0] < min_len:
                continue
            one_set = [user_index, group['loc_index'].tolist(), group['weekday'].astype(int).tolist(),
                       group['timestamp'].astype(int).tolist(), group.shape[0]]

            if include_delta:
                one_set += [[0] + group['time_delta'].iloc[:-1].tolist(),
                            [0] + group['dist'].iloc[:-1].tolist(),
                            group['lat'].tolist(),
                            group['lng'].tolist()]

            seq_set.append(one_set)
        return seq_set
