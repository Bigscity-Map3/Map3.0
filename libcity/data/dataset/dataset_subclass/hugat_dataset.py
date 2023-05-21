import os
from logging import getLogger
import geopandas as gpd
import numpy as np
import pandas as pd
from libcity.data.dataset import AbstractDataset
from libcity.utils import ensure_dir
from tqdm import tqdm
import dgl


class HUGATDataset(AbstractDataset):
    def __init__(self, config):
        # 配置文件和logger
        self.config = config
        self._logger = getLogger()

        # 数据集信息
        self.dataset = self.config.get('dataset', '')
        self.data_path = './raw_data/' + self.dataset + '/'
        if not os.path.exists(self.data_path):
            raise ValueError("Dataset {} not exist! Please ensure the path "
                             "'./raw_data/{}/' exist!".format(self.dataset, self.dataset))
        ensure_dir('./libcity/cache/dataset_cache/{}'.format(self.dataset))

        # 数据集的各个文件路径
        self.geo_file = self.config.get('geo_file', self.dataset)
        self.rel_file = self.config.get('rel_file', self.dataset)
        self.dyna_file = self.config.get('dyna_file',self.dataset)
        self.ext_file = self.config.get('ext_file', self.dataset)
        self.od_file = self.config.get('od_file', self.dataset)

        # od矩阵的缓存路径
        self.od_mx_path = './libcity/cache/dataset_cache/{}/od_label.npy'.format(self.dataset)

        # 目标节点的geo_id和下标的映射
        self.ind_to_geo_path = './libcity/cache/dataset_cache/{}/ind_to_geo_{}.npy'.format(self.dataset,self.remove_node_type)
        self.func_label_path = './libcity/cache/dataset_cache/{}/func_label_{}.npy'.format(self.dataset,self.remove_node_type)

        # 重要的原始数据
        self.geo_ids = None

        self.region_ids = None
        self.num_regions = None
        
        self.poi_ids = None
        self.num_pois = None

        self.num_nodes = None

        self.region_geometry = None
        self.centroid = None

        self.crime_count = None
        self.poi_category = None

        self.poi2region = None
        self.region2poi = None
        self.region2region = None
        self.adj_mx = None

        self.check_in_poi = None
        self.check_in_time = None

        self.land_usage_region = None
        self.land_usage_type = None

        self.od_label = None

        # 重要的中间变量
        # meta path R_R
        self.R_to_tmp = None
        self.tmp_to_R = None

        # meta path R_C_R
        self.R_to_C = None
        self.C_to_R = None

        # meta path R_TC_R
        self.R_to_TC = None
        self.TC_to_R = None

        # meta path R_TO_R
        self.R_to_TO = None
        self.TO_to_R = None

        # meta path R_TD_R
        self.R_to_TD = None
        self.TD_to_R = None

        # 数据处理结果
        self.g = None
        self.feature = None
        self.meta_path = [
            ["r_tmp", "tmp_r"], 
            ["r_c", "c_r"], 
            ["r_tc", "tc_r"], 
            ["r_to", "to_r"], 
            ["r_td", "td_r"]
        ]

        self.P_org_dst = None
        self.P_dst_org = None
        self.S_chk = None
        self.S_land = None

        self.crime_count = None
        
        # 加载.geo文件
        if os.path.exists(self.data_path + self.geo_file + '.geo'):
            self._load_geo()
        else:
            raise ValueError('Not found .geo file!')
        
        # 加载.rel文件
        if os.path.exists(self.data_path + self.rel_file + '.rel'):
            self._load_rel()
        else:
            raise ValueError('Not found .rel file!')

        # 加载.dyna文件
        if os.path.exists(self.data_path + self.dyna_file + '.dyna'):
            self._load_dyna()
        else:
            raise ValueError('Not found .dyna file!')
        
        # 加载.ext文件
        if os.path.exists(self.data_path + self.ext_file + '.ext'):
            self._load_ext()
        else:
            raise ValueError('Not found .ext file!')
        
        # 加载.od文件
        if os.path.exists(self.data_path + self.od_file + '.od'):
            self._load_od()
        else:
            raise ValueError('Not found .od file!')
        
        self.process_meta_path_R_R()
        self.process_meta_path_R_C_R()
        self.process_meta_path_R_TC_R()
        self.process_meta_path_R_TO_R()
        self.process_meta_path_R_TD_R()

        self.process_hetero_graph()


    # 读入各个region和POI
    def _load_geo(self):
        """
        加载.geo文件，格式[geo_id,type,geometry,crime_count,traffic_type,venue_category_name]
        """
        geofile = pd.read_csv(self.data_path + self.geo_file + '.geo')

        self.geo_ids = list(geofile['geo_id'])

        self.region_ids = list(geofile[geofile['traffic_type'] == 'region']['geo_id'])
        self.num_regions = len(self.region_ids)

        self.poi_ids = list(geofile[geofile['traffic_type'] == 'poi']['geo_id'])
        self.num_pois = len(self.poi_ids)

        self.num_nodes = self.num_regions

        l = [coordinate for coordinate in geofile[geofile['traffic_type'] == 'region']['coordinates']]
        self.region_geometry = gpd.GeoSeries.from_wkt(l)
        self.centroid = self.region_geometry.centroid

        self.crime_count = list(geofile[geofile['traffic_type'] == 'region']['crime_count'])
        self.poi_category = list(geofile[geofile['traffic_type'] == 'poi']['venue_category_name'])
        
        self._logger.info("Loaded file " + self.geo_file + '.geo' + ',num_regions=' + str(self.num_regions) + ',num_pois=' + str(self.num_pois) + ', num_nodes=' + str(self.num_nodes))

    def _load_rel(self):
        """
        加载POI和region的联系，格式[rel_id,type,origin_id,destination_id,rel_type]
        """
        relfile = pd.read_csv(self.data_path + self.rel_file + '.rel')
        self.poi2region = relfile[relfile['rel_type'] == 'poi2region']
        self.region2poi = relfile[relfile['rel_type'] == 'region2poi']
        self.region2region = relfile[relfile['rel_type'] == 'region2region']
        self.region2region = self.region2region.reset_index(drop = True)
        self.adj_mx = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        for i in range(len(self.region2region)):
            origin_region_id = self.region2region.loc[i,"origin_id"]
            destination_region_id = self.region2region.loc[i,"destination_id"]
            distance = self.centroid[origin_region_id].distance(self.centroid[destination_region_id])
            self.adj_mx[origin_region_id][destination_region_id] = distance
        self._logger.info("Loaded file " + self.rel_file + '.rel and finish constructing adj_mx')

    def _load_dyna(self):
        """
        加载签到数据，格式[dyna_id,type,time,geo_id]
        """
        dynafile = pd.read_csv(self.data_path + self.dyna_file + '.dyna')
        self.check_in_poi = dynafile['geo_id']
        self.check_in_time = dynafile['time']

    def _load_ext(self):
        """
        加载土地使用数据，格式[ext_id,geo_id,landuse_type]
        """
        extfile = pd.read_csv(self.data_path + self.ext_file + '.ext')
        self.land_usage_region = extfile['geo_id']
        self.land_usage_type = extfile['landuse_type']
        
    def _load_od(self):
        """
        构造区域间的od矩阵 dyna_id,type,time,origin_id,destination_id,flow
        :return: od_mx
        """
        if os.path.exists(self.od_mx_path):
            self.od_label = np.load(self.od_mx_path)
            self._logger.info("finish construct od graph")
            return self.od_label
        odfile = pd.read_csv(self.data_path + self.od_file + '.od')
        self.od_label = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        pbar = tqdm(range(len(odfile)))
        pbar.set_description("construct od matrix")
        for i in pbar:
            origin_region = odfile.loc[i,"origin_id"]
            destination_region = odfile.loc[i,"destination_id"]
            self.od_label[origin_region][destination_region] += 1
        np.save(self.od_mx_path, self.od_label)
        self._logger.info("finish construct od graph")
        return self.od_label
    
    def process_meta_path_R_R(self):
        tmp_id = 0
        region_to_tmp = []
        tmp_to_region = []

        for i in range(len(self.adj_mx)):
            region_to_tmp.append([i, tmp_id])
            tmp_to_region.append([tmp_id, i])
            tmp_id += 1
            for j in range(i + 1, len(self.adj_mx)):
                if (self.adj_mx[i][j] != 0 and self.adj_mx[j][i] != 0) or (self.adj_mx[i][j] != 0):
                    region_to_tmp.append([i, tmp_id])
                    tmp_to_region.append([tmp_id, j])
                    tmp_id += 1
                elif self.adj_mx[j][i] != 0:
                    region_to_tmp.append([j, tmp_id])
                    tmp_to_region.append([tmp_id, i])
                    tmp_id += 1

        region_to_tmp = np.array(region_to_tmp).transpose()
        region_to_tmp = (region_to_tmp[0], region_to_tmp[1])
        
        tmp_to_region = np.array(tmp_to_region).transpose()
        tmp_to_region = (tmp_to_region[0], tmp_to_region[1])

        self.R_to_tmp = region_to_tmp
        self.tmp_to_R = tmp_to_region

    def process_meta_path_R_C_R(self):
        pass

    def process_meta_path_R_TC_R(self):
        pass

    def process_meta_path_R_TO_R(self):
        pass

    def process_meta_path_R_TD_R(self):
        pass

    def process_hetero_graph(self):
        self.g = dgl.heterograph(
            {
                ("region", "r_tmp", "tmp"): self.R_to_tmp,
                ("tmp", "tmp_r", "region"): self.tmp_to_R,

                ("region", "r_c", "category"): self.R_to_C,
                ("category", "c_r", "region"): self.C_to_R,

                ("region", "r_tc", "TC"): self.R_to_TC,
                ("TC", "tc_r", "region"): self.TC_to_R,

                ("region", "r_to", "TO"): self.R_to_TO,
                ("TO", "to_r", "region"): self.TO_to_R,

                ("region", "r_td", "TD"): self.R_to_TD,
                ("TD", "td_r", "region"): self.TD_to_R,
            }
        )

    def get_data(self):
        return None, None, None

    def get_data_feature(self):
        """
        返回一个 dict，包含数据集的相关特征

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        return {'num_nodes': self.num_nodes, 'g': self.g, 'feature': self.feature, 'meta_path': self.meta_path,
                'P_org_dst': self.P_org_dst, 'P_dst_org': self.P_dst_org, 'S_chk': self.S_chk, 'S_land': self.S_land,
                'crime_count_predict': self.crime_count}
