import os
import json
import numpy as np
import pandas as pd
from logging import getLogger
from datetime import datetime
from tqdm import tqdm


cache_dir = os.path.join('libcity', 'cache', 'dataset_cache')


def str2timestamp(s):
    return int(datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timestamp())


class PreProcess():
    def __init__(self, config):
        self.dataset = config.get('dataset')
        self.data_dir = os.path.join(cache_dir, self.dataset)
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        self.geo_file = os.path.join('raw_data', self.dataset, config.get('geo_file', self.dataset) + '.geo')
        self.rel_file = os.path.join('raw_data', self.dataset, config.get('rel_file', self.dataset) + '.rel')
        self.dyna_file = os.path.join('raw_data', self.dataset, config.get('dyna_file', self.dataset) + '.dyna')


class preprocess_traj(PreProcess):
    # 在目前的代码中，该文件被命名为 traj_{dataset}_11.csv，但现在计划改名，将其与 region 区分开
    def __init__(self, config):
        super().__init__(config)
        file_name = 'traj_road.csv'
        self.data_file = os.path.join(self.data_dir, file_name)
        if not os.path.exists(self.data_file):
            logger = getLogger()
            logger.info('Start preprocess traj.')
            dyna_df = pd.read_csv(self.dyna_file)
            id = []
            path = []
            tlist = []
            length = []
            speed = []
            duration = []
            hop = []
            usr_id = []
            traj_id = []
            start_time = []
            lst_traj_id, lst_usr_id = None, None
            geo_df = pd.read_csv(self.geo_file)
            num_regions = geo_df[geo_df['traffic_type'] == 'region'].shape[0]
            for _, row in tqdm(dyna_df.iterrows(), total=dyna_df.shape[0]):
                idx = int(row['total_traj_id'])
                if lst_traj_id != row['traj_id'] or lst_usr_id != row['entity_id']:  # 轨迹划分依据还存疑，靠 traj_id 和 total_traj_id 都不行
                    idx = len(id)
                    id.append(idx)
                    path.append([])
                    tlist.append([])
                    length.append(0.)
                    speed.append(0.)
                    duration.append(0)
                    hop.append(0)
                    usr_id.append(row['entity_id'])
                    traj_id.append(row['traj_id'])
                    start_time.append(row['time'].split(' ')[0])
                tlist[idx].append(str2timestamp(row['time']))
                path[idx].append(row['geo_id'] - num_regions)
                lst_traj_id = row['traj_id']
                lst_usr_id = row['entity_id']
            for i in range(id[-1]):
                duration[i] = tlist[i][-1] - tlist[i][0]
                hop[i] = len(path[i])
            df = pd.concat(
                [
                    pd.Series(id, name='id'), 
                    pd.Series(path, name='path'), 
                    pd.Series(tlist, name='tlist'),
                    pd.Series(length, name='length'),
                    pd.Series(speed, name='speed'),
                    pd.Series(duration, name='duration'),
                    pd.Series(hop, name='hop'),
                    pd.Series(usr_id, name='usr_id'),
                    pd.Series(traj_id, name='traj_id'),
                    pd.Series(start_time, name='start_time')
                ], axis=1)
            df.to_csv(self.data_file, index=False)
            # 没有分验证集的必要 4:1
            train_file = os.path.join(self.data_dir, 'traj_road_train.csv')
            test_file = os.path.join(self.data_dir, 'traj_road_test.csv')
            train_df = df.sample(frac=4/5, random_state=1)
            test_df = df.drop(train_df.index)
            train_df.to_csv(train_file, index=False)
            test_df.to_csv(test_file, index=False)

            # region traj
            file_name = 'traj_region.csv'
            data_file = os.path.join(self.data_dir, file_name)
            rel_df = pd.read_csv(self.rel_file)
            road2region_df = rel_df[rel_df['rel_type'] == 'road2region']
            road2region = {}
            for _, row in road2region_df.iterrows():
                x = int(row['origin_id']) - num_regions
                y = int(row['destination_id'])
                road2region[x] = y
            region_paths = []
            region_tlists = []
            for i, road_path in enumerate(path):
                tmp1, tmp2 = [], []
                lst_region = None
                for j, road in enumerate(road_path):
                    region = road2region[road]
                    if region != lst_region:
                        tmp1.append(region)
                        tmp2.append(tlist[i][j])
                        lst_region = region
                region_paths.append(tmp1)
                region_tlists.append(tmp2)
            df = pd.concat(
                [
                    pd.Series(id, name='id'), 
                    pd.Series(region_paths, name='path'), 
                    pd.Series(tlist, name='tlist'),
                    pd.Series(usr_id, name='usr_id'),
                    pd.Series(traj_id, name='traj_id'),
                    pd.Series(start_time, name='start_time')
                ], axis=1)
            df.to_csv(data_file, index=False)
            # 没有分验证集的必要 4:1
            train_file = os.path.join(self.data_dir, 'traj_region_train.csv')
            test_file = os.path.join(self.data_dir, 'traj_region_test.csv')
            train_df = df.sample(frac=4/5, random_state=1)
            test_df = df.drop(train_df.index)
            train_df.to_csv(train_file, index=False)
            test_df.to_csv(test_file, index=False)
            logger.info('Finish preprocess traj.')
        

class preprocess_csv(PreProcess):
    def __init__(self, config):
        super().__init__(config)
        if not os.path.exists(os.path.join(self.data_dir, 'POI.csv')):
            logger = getLogger()
            logger.info('Start preprocess csv.')
            geo_df = pd.read_csv(self.geo_file)
            df_dict = {
                'poi': pd.DataFrame(),
                'region': pd.DataFrame(),
                'road': pd.DataFrame()
            }
            for key in geo_df.keys():
                for traffic_type in df_dict.keys():
                    if key.startswith(traffic_type):
                        df_dict[traffic_type][key[len(traffic_type) + 1:]] = geo_df[key].dropna()
            for key in ['id']:
                df_dict['poi'][key] = df_dict['poi'][key].astype(int)
            for key in ['id', 'PARCEL_ID', 'FUNCTION', 'BLD_Count', 'InCBD', 'FORM_TYPE']:
                df_dict['region'][key] = df_dict['region'][key].astype(int)
            for key in ['id', 'highway', 'lanes', 'tunnel', 'bridge', 'roundabout', 'oneway', 'maxspeed', 'u', 'v']:
                df_dict['road'][key] = df_dict['road'][key].astype(int)
            df_dict['poi'].to_csv(os.path.join(self.data_dir, 'POI.csv'), index=False)
            df_dict['region'].to_csv(os.path.join(self.data_dir, 'region.csv'), index=False)
            df_dict['road'].to_csv(os.path.join(self.data_dir, 'road.csv'), index=False)
            logger.info('Finish preprocess csv.')


def save_od_matrix(data_dir, file_name, n):
    traj_df = pd.read_csv(os.path.join(data_dir, file_name + '.csv'))
    file_path = os.path.join(data_dir, file_name + '_od.npy')
    if not os.path.exists(file_path):
        od_matrix = np.zeros((n, n))
        for _, row in traj_df.iterrows():
            tmp = row['path'].split(',')
            if len(tmp) == 1:
                origin = destination = int(tmp[0][1:-1])
            else:
                origin = int(tmp[0][1:])
                destination = int(tmp[-1][:-1])
            od_matrix[origin][destination] += 1
        np.save(file_path, od_matrix)


def save_in_avg(data_dir, file_name, num_days):
    file_path = os.path.join(data_dir, file_name + '_in_avg.npy')
    if not os.path.exists(file_path):
        od_file = np.load(os.path.join(data_dir, file_name + '_od.npy'))
        np.save(file_path, np.sum(od_file, axis=0) / num_days)


def save_out_avg(data_dir, file_name, num_days):
    file_path = os.path.join(data_dir, file_name + '_out_avg.npy')
    if not os.path.exists(file_path):
        od_file = np.load(os.path.join(data_dir, file_name + '_od.npy'))
        np.save(file_path, np.sum(od_file, axis=1) / num_days)


class preprocess_od(PreProcess):
    def __init__(self, config):
        super().__init__(config)
        preprocess_traj(config)
        geo_df = pd.read_csv(self.geo_file)
        num_regions = geo_df[geo_df['traffic_type'] == 'region'].shape[0]
        num_roads = geo_df[geo_df['traffic_type'] == 'road'].shape[0]
        traj_df = pd.read_csv(os.path.join(self.data_dir, 'traj_road.csv'))
        num_days = traj_df['start_time'].drop_duplicates().shape[0]
        save_od_matrix(self.data_dir, 'traj_region'      , num_regions)
        save_od_matrix(self.data_dir, 'traj_region_train', num_regions)
        save_od_matrix(self.data_dir, 'traj_region_test' , num_regions)
        save_od_matrix(self.data_dir, 'traj_road'        , num_roads  )
        save_od_matrix(self.data_dir, 'traj_road_train'  , num_roads  )
        save_od_matrix(self.data_dir, 'traj_road_test'   , num_roads  )
        save_in_avg(self.data_dir, 'traj_region_test', num_days)
        save_in_avg(self.data_dir, 'traj_road_test', num_days)
        save_out_avg(self.data_dir, 'traj_region_test', num_days)
        save_out_avg(self.data_dir, 'traj_road_test', num_days)


class preprocess_feature(PreProcess):
    def __init__(self, config):
        super().__init__(config)
        if not os.path.exists(os.path.join(self.data_dir, 'region_features.csv')):
            geo_df = pd.read_csv(self.geo_file)
            FUNCTION = geo_df['region_FUNCTION'].dropna().astype(int)
            InCBD = geo_df['region_InCBD'].dropna().astype(int)
            FORM_TYPE = geo_df['region_FORM_TYPE'].dropna().astype(int)
            highway = geo_df['road_highway'].dropna().astype(int)
            lanes = geo_df['road_lanes'].dropna().astype(int)
            maxspeed = geo_df['road_maxspeed'].dropna().astype(int)
            region_df = pd.concat(
                        [
                            pd.Series(FUNCTION, name='FUNCTION'), 
                            pd.Series(InCBD, name='InCBD'), 
                            pd.Series(FORM_TYPE, name='FORM_TYPE')
                        ], axis=1)
            road_df = pd.concat(
                        [
                            pd.Series(highway, name='highway'), 
                            pd.Series(lanes, name='lanes'),
                            pd.Series(maxspeed, name='maxspeed')
                        ], axis=1)
            region_df.to_csv(os.path.join(self.data_dir, 'region_features.csv'), index=False)
            road_df.to_csv(os.path.join(self.data_dir, 'road_features.csv'), index=False)


def save_neighbor(traffic_type, df, offset, data_dir, n):
    file_path = os.path.join(data_dir, traffic_type + '_neighbor.json')
    if not os.path.exists(file_path):
        df = df[df['rel_type'] == traffic_type + '2' + traffic_type]
        neighbor_dict = {}
        for i in range(n):
            neighbor_dict[i] = []
        for _, row in df.iterrows():
            x = row['origin_id'] - offset
            y = row['destination_id'] - offset
            neighbor_dict[x].append(y)
        with open(file_path, 'w') as f:
            json.dump(neighbor_dict, f)


class preprocess_neighbor(PreProcess):
    def __init__(self, config):
        super().__init__(config)
        rel_df = pd.read_csv(self.rel_file)
        geo_df = pd.read_csv(self.geo_file)
        num_regions = geo_df[geo_df['traffic_type'] == 'region'].shape[0]
        num_roads = geo_df[geo_df['traffic_type'] == 'road'].shape[0]
        save_neighbor('road', rel_df, num_regions, self.data_dir, num_roads)
        save_neighbor('region', rel_df, 0, self.data_dir, num_regions)


def preprocess_all(config):
    preprocess_csv(config)
    preprocess_feature(config)
    preprocess_neighbor(config)
    preprocess_traj(config)
    preprocess_od(config)
