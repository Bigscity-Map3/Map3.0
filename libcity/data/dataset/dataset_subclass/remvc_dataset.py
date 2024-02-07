import os
import pandas as pd
import numpy as np
from datetime import datetime
from libcity.data.dataset.traffic_representation_dataset import TrafficRepresentationDataset


def gen_index_map(df, column, offset=0):
    index_map = {origin: index
                 for index, origin in enumerate(df[column].drop_duplicates())}
    return index_map


def convert_to_seconds_in_day(time_string):
    # 解析时间字符串为datetime对象
    dt = datetime.strptime(time_string, "%Y-%m-%d %H:%M:%S")
    # 将小时、分钟、秒转换为秒数并相加
    seconds_in_day = dt.hour * 3600 + dt.minute * 60 + dt.second
    return seconds_in_day


class ReMVCDataset(TrafficRepresentationDataset):
    def __init__(self, config):
        self.config = config
        super().__init__(config)
        self.dataset = self.config.get('dataset', '')
        self.data_path = './raw_data/' + self.dataset + '/'
        self.geo_file = self.config.get('geo_file', self.dataset)
        self.rel_file = self.config.get('rel_file', self.dataset)
        self.dyna_file = self.config.get('dyna_file', self.dataset)
        assert os.path.exists(self.data_path + self.geo_file + '.geo')
        assert os.path.exists(self.data_path + self.rel_file + '.rel')
        assert os.path.exists(self.data_path + self.dyna_file + '.dyna')
        if not os.path.exists('./libcity/cache/ReMVC_{}'.format(self.dataset)):
            os.mkdir('./libcity/cache/ReMVC_{}'.format(self.dataset))

        self.get_region_dict()
        self.get_model_poi()
        self.get_model_flow()
        self.get_matrix_dict()

    def get_region_dict(self):
        self._logger.info('Begin get_region_dict')
        region_dict = {}
        time_slices_num = self.config.get('time_slices_num', 48)
        for i in range(self.num_regions):
            region_dict[i] = {
                'poi': [],
                'pickup_matrix': np.array([[0] * self.num_regions for _ in range(time_slices_num)]),
                'dropoff_matrix': np.array([[0] * self.num_regions for _ in range(time_slices_num)])
            }

        # poi
        poi_dict = {}
        poi_df = self.geofile[self.geofile['traffic_type'] == 'poi']
        poi_map = gen_index_map(poi_df, 'function')
        self.num_poi_types = len(poi_map)
        for _, row in poi_df.iterrows():
            poi_dict[row['geo_id']] = poi_map[row['function']]
        for _, row in self.region2poi.iterrows():
            region_id = row['origin_id']
            poi_id = row['destination_id']
            region_dict[region_id]['poi'].append(poi_dict[poi_id])

        # matrix
        dyna_df = pd.read_csv(self.data_path + self.dyna_file + '.dyna')
        lst_poi, lst_dyna = None, None
        # self._logger.info('Total {}'.format(len(dyna_df)))
        for _, row in dyna_df.iterrows():
            # if (_ + 1) % 2000 == 0:
            #     self._logger.info('Finish {}'.format(_ + 1))
            cur_poi, cur_dyna, t = row['geo_id'], row['traj_id'], convert_to_seconds_in_day(str(row['time']))
            time_slice = t // int(86400 / time_slices_num)
            if lst_dyna is not None and lst_dyna == cur_dyna:
                region_dict[lst_poi]['pickup_matrix'][time_slice][cur_poi] += 1
                region_dict[cur_poi]['dropoff_matrix'][time_slice][lst_poi] += 1
            lst_poi, lst_dyna = cur_poi, cur_dyna

        self.region_dict = region_dict

    def get_model_poi(self):
        self._logger.info('Start get model poi ...')
        poi_features = {}
        for i in range(self.num_regions):
            poi_features[i] = np.zeros(self.num_poi_types)
            for j in self.region_dict[i]['poi']:
                poi_features[i][j] += 1
        model_poi = {}
        for i in range(self.num_regions):
            ll = 0
            model_poi[i] = np.zeros(self.num_regions - 1)
            for j in range(self.num_regions):
                if i != j:
                    model_poi[i][ll] = np.sqrt(np.sum((poi_features[i] - poi_features[j]) ** 2))
                    ll += 1
            model_poi[i] = model_poi[i] / np.sum(model_poi[i])
        self.model_poi = model_poi
        self._logger.info('Finish get model poi ...')

    def get_model_flow(self):
        self._logger.info('Start get model flow ...')
        model_flow = {}
        for i in range(self.num_regions):
            ll = 0
            model_flow[i] = np.zeros(self.num_regions - 1)
            for j in range(self.num_regions):
                if i != j:
                    model_flow[i][ll] = \
                        np.sqrt(np.sum((self.region_dict[i]['pickup_matrix'][:, np.newaxis] -
                                        self.region_dict[j]['pickup_matrix']) ** 2, axis=2)) + \
                        np.sqrt(np.sum((self.region_dict[i]['dropoff_matrix'][:, np.newaxis] -
                                        self.region_dict[j]['dropoff_matrix']) ** 2, axis=2))
                    ll += 1
            model_flow[i] = model_flow[i] / np.sum(model_flow[i])
        self.model_flow = model_flow
        self._logger.info('Finish get model flow ...')

    def get_matrix_dict(self):
        matrix_dict = {}
        for idx in range(self.num_regions):
            pickup_matrix = self.region_dict[idx]["pickup_matrix"]
            dropoff_matrix = self.region_dict[idx]["dropoff_matrix"]

            pickup_matrix = pickup_matrix / pickup_matrix.sum()
            where_are_NaNs = np.isnan(pickup_matrix)
            pickup_matrix[where_are_NaNs] = 0

            dropoff_matrix = dropoff_matrix / dropoff_matrix.sum()
            where_are_NaNs = np.isnan(dropoff_matrix)
            dropoff_matrix[where_are_NaNs] = 0

            matrix_dict[idx] = pickup_matrix, dropoff_matrix
        self.matrix_dict = matrix_dict

    def get_data(self):
        return None, None, None

    def get_data_feature(self):
        function = np.zeros(self.num_regions)
        region_df = self.geofile[self.geofile['traffic_type'] == 'region']
        for i, row in region_df.iterrows():
            function[i] = row['function']
        return {
            'region_dict': self.region_dict,
            'model_poi': self.model_poi,
            'model_flow': self.model_flow,
            'matrix_dict': self.matrix_dict,
            'sampling_pool': [i for i in range(self.num_regions)],
            'num_pois': self.num_pois,
            'num_poi_types': self.num_poi_types,
            'num_regions': self.num_regions,
            'label': {
                'function_cluster': function
            }
        }
