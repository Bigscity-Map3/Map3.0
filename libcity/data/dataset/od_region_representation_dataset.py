from libcity.data.dataset import AbstractDataset
import os
from logging import getLogger
import geopandas as gpd
import numpy as np
import pandas as pd
from libcity.utils import ensure_dir
from libcity.utils import geojson2geometry
from tqdm import tqdm
import copy


class ODRegionRepresentationDataset(AbstractDataset):
    def __init__(self, config):
        self.config = config
        self.dataset = self.config.get('dataset', '')
        self.batch_size = self.config.get('batch_size', 64)
        self.cache_dataset = self.config.get('cache_dataset', True)
        self.num_workers = self.config.get('num_workers', 0)
        self.pad_with_last_sample = self.config.get('pad_with_last_sample', True)
        self.train_rate = self.config.get('train_rate', 0.7)
        self.eval_rate = self.config.get('eval_rate', 0.1)
        self.scaler_type = self.config.get('scaler', 'none')
        self.ext_scaler_type = self.config.get('ext_scaler', 'none')
        self.load_external = self.config.get('load_external', False)
        self.normal_external = self.config.get('normal_external', False)
        self.add_time_in_day = self.config.get('add_time_in_day', False)
        self.add_day_in_week = self.config.get('add_day_in_week', False)
        self.representation_object = self.config.get('representation_object', 'region')  #####
        self.input_window = self.config.get('input_window', 12)
        self.output_window = self.config.get('output_window', 12)
        self.region_geometry = None
        self.parameters_str = \
            str(self.dataset) + '_' + str(self.input_window) + '_' + str(self.output_window) + '_' \
            + str(self.train_rate) + '_' + str(self.eval_rate) + '_' + str(self.scaler_type) + '_' \
            + str(self.batch_size) + '_' + str(self.load_external) + '_' + str(self.add_time_in_day) + '_' \
            + str(self.add_day_in_week) + '_' + str(self.pad_with_last_sample)
        self.cache_file_name = os.path.join('./libcity/cache/dataset_cache/',
                                            'od_region_representation_{}.npz'.format(self.parameters_str))
        self.cache_file_folder = './libcity/cache/dataset_cache/'
        ensure_dir(self.cache_file_folder)
        self.data_path = './raw_data/' + self.dataset + '/'
        if not os.path.exists(self.data_path):
            raise ValueError("Dataset {} not exist! Please ensure the path "
                             "'./raw_data/{}/' exist!".format(self.dataset, self.dataset))
        ensure_dir('./libcity/cache/dataset_cache/{}'.format(self.dataset))
        self.od_label_path = './libcity/cache/dataset_cache/{}/od_mx.npy'.format(self.dataset)
        self.traj_road_path = './libcity/cache/dataset_cache/{}/traj_road.txt'.format(self.dataset)
        self.traj_time_path = './libcity/cache/dataset_cache/{}/traj_time.txt'.format(self.dataset)
        # 加载数据集的config.json文件
        self.weight_col = self.config.get('weight_col', '')
        self.data_col = self.config.get('data_col', '')
        self.ext_col = self.config.get('ext_col', '')
        self.geo_file = self.config.get('geo_file', self.dataset)
        self.rel_file = self.config.get('rel_file', self.dataset)
        self.dyna_file = self.config.get('dyna_file', self.dataset)
        self.od_file = self.config.get('od_file', self.dataset)
        self.data_files = self.config.get('data_files', self.dataset)
        self.ext_file = self.config.get('ext_file', self.dataset)
        self.output_dim = self.config.get('output_dim', 1)
        self.time_intervals = self.config.get('time_intervals', 300)  # s
        self.init_weight_inf_or_zero = self.config.get('init_weight_inf_or_zero', 'inf')
        self.set_weight_link_or_dist = self.config.get('set_weight_link_or_dist', 'dist')
        self.bidir_adj_mx = self.config.get('bidir_adj_mx', False)
        self.calculate_weight_adj = self.config.get('calculate_weight_adj', False)
        self.weight_adj_epsilon = self.config.get('weight_adj_epsilon', 0.1)
        self.distance_inverse = self.config.get('distance_inverse', False)
        self.remove_node_type = self.config.get("remove_node_type", "od")
        self.ind_to_geo_path = './libcity/cache/dataset_cache/{}/ind_to_geo_{}.npy'.format(self.dataset,
                                                                                           self.remove_node_type)
        self.func_label_path = './libcity/cache/dataset_cache/{}/func_label_{}.npy'.format(self.dataset,
                                                                                           self.remove_node_type)
        # 初始化
        self.data = None
        self.feature_name = {'X': 'float', 'y': 'float'}  # 此类的输入只有X和y
        self.adj_mx = None
        self.scaler = None
        self.ext_scaler = None
        self.feature_dim = 0
        self.ext_dim = 0
        self.num_nodes = 0
        self.num_batches = 0
        self._logger = getLogger()
        self.num_regions = 0
        self.num_roads = 0
        self.num_pois = 0
        self.traj_road = []
        self.traj_time = []
        self.road2region = None
        self.region2road = None
        self.poi2region = None
        self.region2poi = None
        self.poi2road = None
        self.road2poi = None
        if os.path.exists(self.data_path + self.geo_file + '.geo'):
            self._load_geo()
        else:
            raise ValueError('Not found .geo file!')
        if os.path.exists(self.data_path + self.rel_file + '.rel'):  # .rel file is not necessary
            self._load_rel()
        else:
            self.adj_mx = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        if os.path.exists(self.data_path + self.dyna_file + '.dyna'):
            self._load_dyna()
        else:
            raise ValueError('Not found .dyna file!')
        if os.path.exists(self.data_path + self.od_file + '.od'):
            self._load_od()
        else:
            self.construct_od_matrix()

    def _load_geo(self):
        """
        加载.geo文件，格式[geo_id,type,geometry,crime_count,traffic_type,venue_category_name]
        """
        geofile = pd.read_csv(self.data_path + self.geo_file + '.geo')
        self.geofile = geofile
        l = [geojson2geometry(coordinate) for coordinate in geofile[geofile['traffic_type'] == 'region']['coordinates']]
        self.region_geometry = gpd.GeoSeries.from_wkt(l)
        self.centroid = self.region_geometry.centroid
        self.geo_ids = list(geofile['geo_id'])
        self.region_ids = list(geofile[geofile['traffic_type'] == 'region']['geo_id'])
        self.num_regions = len(self.region_ids)
        self.poi_ids = list(geofile[geofile['traffic_type'] == 'poi']['geo_id'])
        self.num_pois = len(self.poi_ids)
        self.num_nodes = self.num_regions
        # self.crime_count = list(geofile[geofile['traffic_type'] == 'region']['crime_count'])
        # self.poi_function_class = list(geofile['venue_category_name'].astype('category').cat.codes)
        self._logger.info(
            "Loaded file " + self.geo_file + '.geo' + ',num_regions=' + str(self.num_regions) + ',num_roads=' + str(
                self.num_roads) + ',num_pois=' + str(self.num_pois) + ', num_nodes=' + str(self.num_nodes))
        return

    def _load_rel(self):
        """
        加载各个实体的联系，[rel_id,type,origin_id,destination_id,rel_type]
        将region2poi和poi2region保存
        把region2region直接做成self.adj_mx，邻接的保存为质心距离，不邻接的保存为0
        """
        import pdb 
        pdb.set_trace()
        relfile = pd.read_csv(self.data_path + self.rel_file + '.rel')
        self.poi2region = relfile[relfile['rel_type'] == 'poi2region']
        self.region2poi = relfile[relfile['rel_type'] == 'region2poi']
        self.region2region = relfile[relfile['rel_type'] == 'region2region']
        self.region2region = self.region2region.reset_index(drop=True)
        self.adj_mx = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        for i in range(len(self.region2region)):
            origin_region_id = self.region2region.loc[i, "origin_id"]
            destination_region_id = self.region2region.loc[i, "destination_id"]
            distance = self.centroid[origin_region_id].distance(self.centroid[destination_region_id])
            self.adj_mx[origin_region_id][destination_region_id] = distance
        self._logger.info("Loaded file " + self.rel_file + '.rel and finish constructing adj_mx')

    def _load_dyna(self):
        """
        加载轨迹数据，格式['dyna_id','type','time','entity_id','traj_id','geo_id','total_traj_id']
        目前将轨迹数据整理成若干[road0,road1...road]的集合,以及[time0,time1.....,time]的集合
        构造图在模型的dataset子类中实现
        """
        if os.path.exists(self.traj_road_path) and os.path.exists(self.traj_time_path):
            f1 = open(self.traj_road_path, 'r')
            f2 = open(self.traj_time_path, 'r')
            road_lines = f1.readlines()
            time_lines = f2.readlines()
            for line in road_lines:
                self.traj_road.append([int(road_str) for road_str in line.split(',')])
            for line in time_lines:
                line = line[:-1]
                self.traj_time.append(line.split(','))
            f1.close()
            f2.close()
            self._logger.info("Loaded file " + self.dyna_file + '.dyna')
        else:
            dynafile = pd.read_csv(self.data_path + self.dyna_file + '.dyna')
            traj_num = dynafile['total_traj_id'].max() + 1
            traj_road_str = ""
            traj_time_str = ""
            # 将traj_road存成road0,road1...road(一行一条轨迹）的格式
            # 将traj_time存成time0,time1.....,time（一行一条轨迹）的格式
            for i in tqdm(range(traj_num)):
                road_list = list(dynafile[dynafile['total_traj_id'] == i]['geo_id'])
                time_list = list(dynafile[dynafile['total_traj_id'] == i]['time'])
                self.traj_road.append(road_list)
                self.traj_time.append(time_list)
                for road in road_list:
                    traj_road_str += (str(road) + ',')
                traj_road_str = traj_road_str[:-1]
                traj_road_str += '\n'
                for time in time_list:
                    traj_time_str += (time + ',')
                traj_time_str = traj_time_str[:-1]
                traj_time_str += '\n'
            f1 = open(self.traj_road_path, 'w')
            f1.write(traj_road_str)
            f1.close()
            f2 = open(self.traj_time_path, 'w')
            f2.write(traj_time_str)
            f2.close()
            self._logger.info("Dyna file has been saved")
            self._logger.info("Loaded file " + self.dyna_file + '.dyna')

    def _load_od(self):
        """
        构造区域间的od矩阵 dyna_id,type,time,origin_id,destination_id,flow
        :return: od_mx
        """
        if os.path.exists(self.od_label_path):
            self.od_label = np.load(self.od_label_path)
            self._logger.info("finish construct od graph")
            return self.od_label
        assert self.representation_object == "region"
        odfile = pd.read_csv(self.data_path + self.od_file + '.od')
        total_flow = odfile['flow'].sum()
        self._logger.info("total_flow = {}".format(total_flow))
        self.od_label = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        pbar = tqdm(range(len(odfile)))
        pbar.set_description("construct od matrix")
        for i in pbar:
            origin_region = odfile.loc[i, "origin_id"]
            destination_region = odfile.loc[i, "destination_id"]
            flow = odfile.loc[i, "flow"]
            self.od_label[origin_region][destination_region] += flow
        np.save(self.od_label_path, self.od_label)
        self._logger.info("finish construct od graph")
        return self.od_label

    def construct_od_matrix(self):
        if os.path.exists(self.od_label_path):
            self.od_label = np.load(self.od_label_path)
            self._logger.info("finish construct od graph")
            return
        assert self.representation_object == "region"
        self.od_label = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        for traj in self.traj_road:
            origin_region = traj[0]
            destination_region = traj[-1]
            self.od_label[origin_region][destination_region] += 1
        np.save(self.od_label_path, self.od_label)
        self._logger.info("finish construct od graph")
        return self.od_label
