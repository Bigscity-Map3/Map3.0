import os
from logging import getLogger
import numpy as np
import pandas as pd
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


class preprocess_traj_region_11(PreProcess):
    def __init__(self, config):
        super().__init__(config)
        file_name = 'traj_region_11.csv'  # 上级目录已包含数据集名称
        self.data_file = os.path.join(self.data_dir, file_name)
        if not os.path.exists(self.data_file):
            logger = getLogger()
            logger.info('Start preprocess traj region 11.')
            dyna_df = pd.read_csv(self.dyna_file)
            id = []
            path = []
            tlist = []
            usr_id = []
            traj_id = []
            start_time = []
            lst_traj_id, lst_usr_id = None, None
            for _, row in tqdm(dyna_df.iterrows(), total=dyna_df.shape[0]):
                if lst_traj_id != row['traj_id'] or lst_usr_id != row['entity_id']:  # 轨迹划分依据还存疑，靠 traj_id 和 total_traj_id 都不行
                    idx = len(id)
                    id.append(idx)
                    path.append([])
                    tlist.append([])
                    usr_id.append(row['entity_id'])
                    traj_id.append(row['traj_id'])
                    start_time.append(row['time'].split(' ')[0])
                tlist[idx].append(str2timestamp(row['time']))
                path[idx].append(row['geo_id'])
                lst_traj_id = row['traj_id']
                lst_usr_id = row['entity_id']
            df = pd.concat(
                [
                    pd.Series(id, name='id'), 
                    pd.Series(path, name='path'), 
                    pd.Series(tlist, name='tlist'),
                    pd.Series(usr_id, name='usr_id'),
                    pd.Series(traj_id, name='traj_id'),
                    pd.Series(start_time, name='start_time')
                ], axis=1)
            df.to_csv(self.data_file, index=False)
            logger.info('Finish preprocess traj region 11.')


class preprocess_traj_road_11(PreProcess):
    # 在目前的代码中，该文件被命名为 traj_{dataset}_11.csv，但现在计划改名，将其与 region 区分开
    def __init__(self, config):
        super().__init__(config)
        file_name = 'traj_road_11.csv'
        self.data_file = os.path.join(self.data_dir, file_name)
        if not os.path.exists(self.data_file):
            logger = getLogger()
            logger.info('Start preprocess traj road 11.')
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
                path[idx].append(row['geo_id'])
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
            logger.info('Finish preprocess traj road 11.')
        

if __name__ == '__main__':
    pd.set_option('display.max_rows', None)  # 设置行数为无限制
    pd.set_option('display.max_columns', None)  # 设置列数为无限制
    os.chdir('/home/tangyb/private/tyb/remote/representation')
    config = {'dataset': 'cd'}
    preprocess_traj_region_11(config)