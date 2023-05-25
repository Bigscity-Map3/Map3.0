import os
from datetime import datetime

from logging import getLogger
import geopandas as gpd
import numpy as np
import pandas as pd
from libcity.data.dataset import AbstractDataset
from libcity.utils import ensure_dir
from tqdm import tqdm
import torch
import dgl


def transform_mapping_relation(mapping_set):
    """
    将记录了不同节点映射关系的集合转化为nparray的格式并转置
    :param mapping_set: 记录了不同节点映射关系的集合
    :return: mapping_set转置后的nparray的格式
    """
    mapping_set = list(mapping_set)
    for i in range(len(mapping_set)):
        mapping_set[i] = list(mapping_set[i])
    mapping_set = np.array(mapping_set).transpose()
    mapping_set = (mapping_set[0], mapping_set[1])
    return mapping_set


def hellinger_distance(dim, P):
    """
    利用Hellinger distance对P进行处理新的矩阵
    :param dim: 输出矩阵的维度，同时也是P矩阵的第二个维度
    :param P: 待处理的矩阵
    :return: 处理后的新矩阵
    """
    res = np.zeros((dim, dim), dtype=np.float32)
    for i in range(dim):
        for j in range(i + 1, dim):
            tmp = (1 / np.sqrt(2)) * np.sqrt(np.power(P[:, i] - P[:, j], 2).sum())
            res[i][j] = res[j][i] = tmp
    return res


def judge_POI_validity(poi_id, poi2region):
    """
    判断POI是否在某个region当中，若在返回该region的id，不在返回None
    :param poi_id: POI的id
    :param poi2region: POI和region的对应关系
    :return: region的id或None
    """
    region_id = list(poi2region[poi2region['origin_id'] == poi_id]['destination_id'])
    if len(region_id) == 0:
        return None
    return region_id[0]


def get_time_stamp_node(time, time_format):
    """
    按预设的格式生成时间节点
    :param time: 时间的字符串表示
    :param time_format: 预设的格式
    :return: 时间节点
    """

    # 得到标准时间
    standard_time = datetime.strptime(time, '%Y-%m-%dT%H:%M:%SZ')

    # 按预设的格式生成时间节点
    if time_format == 'hour':
        t = (standard_time.hour)
    elif time_format == 'weekday:hour':
        t = (standard_time.weekday(), standard_time.hour)
    else:
        t = (1 if standard_time.weekday() in range(5) else 0, standard_time.hour)

    return t


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

        # od数据的缓存路径
        self.od_data_path = './libcity/cache/dataset_cache/{}/od_data.npy'.format(self.dataset)
        self.od_mx_path = './libcity/cache/dataset_cache/{}/od_label.npy'.format(self.dataset)

        # 重要的原始数据
        self.feature_dim = self.config.get('feature_dim', 256)
        self.geo_ids = None

        self.region_ids = None
        self.num_regions = None
        
        self.poi_ids = None
        self.num_pois = None

        self.num_nodes = None

        self.region_geometry = None
        self.centroid = None

        self.crime_count = None
        self.poi_function_class = None

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

        # TC、TO、TD的格式
        self.time_format = self.config.get('time_format', 'weekday-type:hour')

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
        self.process_meta_path_R_TO_R_and_R_TD_R()

        self.process_hetero_graph()

        self.generate_node_feature()

        self.process_P_org_dst_and_P_dst_org()
        self.process_S_chk()
        self.process_S_land()

        pass

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

        l = [geometry for geometry in geofile[geofile['traffic_type'] == 'region']['geometry']]
        self.region_geometry = gpd.GeoSeries.from_wkt(l)
        self.centroid = self.region_geometry.centroid

        self.crime_count = list(geofile[geofile['traffic_type'] == 'region']['crime_count'])
        self.poi_function_class = list(geofile['venue_category_name'].astype('category').cat.codes)
        
        self._logger.info("Loaded file " + self.geo_file + '.geo' + ', num_regions=' + str(self.num_regions) + 
                          ', num_pois=' + str(self.num_pois) + ', num_nodes=' + str(self.num_nodes))

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
        self.check_in_poi = list(dynafile['geo_id'])
        self.check_in_time = list(dynafile['time'])
        self._logger.info("Loaded file " + self.dyna_file + '.dyna')

    def _load_ext(self):
        """
        加载土地使用数据，格式[ext_id,geo_id,landuse_type]
        """
        extfile = pd.read_csv(self.data_path + self.ext_file + '.ext')
        self.land_usage_region = list(extfile['geo_id'])
        self.land_usage_type = list(extfile['landuse_type'])
        self._logger.info("Loaded file " + self.ext_file + '.ext')
        
    def _load_od(self):
        """
        加载出租车轨迹数据，构造区域间的od矩阵 dyna_id,type,time,origin_id,destination_id,flow
        """
        if os.path.exists(self.od_mx_path) and os.path.exists(self.od_data_path):
            self.od_label = np.load(self.od_mx_path)
            self.od_data = np.load(self.od_data_path, allow_pickle=True)
            self._logger.info("Loaded file " + self.od_file + '.od' + " and finish construct od graph")
            return

        odfile = pd.read_csv(self.data_path + self.od_file + '.od',nrows=500)
        self.od_label = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
        pbar = tqdm(range(len(odfile)))
        pbar.set_description("construct od matrix")
        for i in pbar:
            origin_region = odfile.loc[i,"origin_id"]
            destination_region = odfile.loc[i,"destination_id"]
            self.od_label[origin_region][destination_region] += 1
        np.save(self.od_mx_path, self.od_label)

        self.od_data = odfile[['origin_id', 'destination_id', 'time']]
        self.od_data = self.od_data.to_numpy()
        np.save(self.od_data_path, self.od_data)

        self._logger.info("Loaded file " + self.od_file + '.od' + " and finish construct od graph")
    
    def process_meta_path_R_R(self):
        """
        构造meta path: RR，为了方便调用dgl的接口，这里的思路是增加一个不存在的虚拟节点tmp，
        若区域i和j之间有邻接关系，则新建一个中间节点tmp，使用i-tmp-j作为meta path
        """

        # 从0开始对tmp节点编号
        tmp_id = 0

        # 区域region和tmp节点的双向对应关系集合
        region_to_tmp = set()
        tmp_to_region = set()

        # 构造对应关系
        for i in range(len(self.adj_mx)):
            # 每个region和自己必然相邻
            region_to_tmp.add((i, tmp_id))
            tmp_to_region.add((tmp_id, i))
            tmp_id += 1

            for j in range(i + 1, len(self.adj_mx)):
                if self.adj_mx[i][j] != 0:
                    region_to_tmp.add((i, tmp_id))
                    tmp_to_region.add((tmp_id, j))
                    tmp_id += 1
                elif self.adj_mx[j][i] != 0:
                    region_to_tmp.add((j, tmp_id))
                    tmp_to_region.add((tmp_id, i))
                    tmp_id += 1

        # 将对应关系转化为需要的格式
        self.R_to_tmp = transform_mapping_relation(region_to_tmp)
        self.tmp_to_R = transform_mapping_relation(tmp_to_region)

        self._logger.info("finish process meta path RR")

    def process_meta_path_R_C_R(self):
        """
        构造meta path: RCR，注意有些POI并不对应任何region
        """

        # 区域region和POI类型节点的双向对应关系集合，采用集合的形式是为了不出现大量的重复边
        region_to_category = set()
        category_to_region = set()

        # 构造对应关系
        for poi_id in self.poi_ids:
            # 首先判断当前POI是否有对应的region
            region_id = judge_POI_validity(poi_id, self.poi2region)
            if region_id is None:
                continue

            # 得到POI类别
            poi_category = self.poi_function_class[poi_id]

            region_to_category.add((region_id, poi_category))
            category_to_region.add((poi_category, region_id))

        # 将对应关系转化为需要的格式
        self.R_to_C = transform_mapping_relation(region_to_category)
        self.C_to_R = transform_mapping_relation(category_to_region)

        self._logger.info("finish process meta path RCR")

    def process_meta_path_R_TC_R(self):
        """
        构造meta path: RTCR，注意有些POI并不对应任何region
        """

        # 从0开始对check-in-time节点编号
        all_tc_id = 0
        tc_to_id = dict()

        # 区域region和check-in-time的双向对应关系集合，采用集合的形式是为了不出现大量的重复边
        region_to_tc = set()
        tc_to_region = set()

        # 构造对应关系
        for poi_id, time_check_in in zip(self.check_in_poi, self.check_in_time):
            # 首先判断当前POI是否有对应的region
            region_id = judge_POI_validity(poi_id, self.poi2region)
            if region_id is None:
                continue

            # 按预设的格式生成check-in-time节点
            tc = get_time_stamp_node(time_check_in, self.time_format)

            # 得到check-in-time节点id
            if tc_to_id.get(tc) is None:
                tc_to_id[tc] = all_tc_id
                all_tc_id += 1
            tc_id = tc_to_id.get(tc)

            region_to_tc.add((region_id, tc_id))
            tc_to_region.add((tc_id, region_id))

        # 将对应关系转化为需要的格式
        self.R_to_TC = transform_mapping_relation(region_to_tc)
        self.TC_to_R = transform_mapping_relation(tc_to_region)

        self._logger.info("finish process meta path RTCR")

    def process_meta_path_R_TO_R_and_R_TD_R(self):
        """
        构造meta path: RTOR、RTDR
        """

        # 从0开始对轨迹时间节点编号，这里本应有TO和TD两个时间，但是由于每条轨迹只有一个时间戳
        # 故把TO和TD看做同一个时间戳节点，
        all_t_id = 0
        t_to_id = dict()

        # 区域region和TO、TD的双向对应关系集合，采用集合的形式是为了不出现大量的重复边
        region_to_to = set()
        to_to_region = set()
        region_to_td = set()
        td_to_region = set()

        for i in tqdm(range(len(self.od_data))):
            # 得到起点和终点
            origin_region_id = self.od_data[i][0]
            destination_region_id = self.od_data[i][1]

            # 按预设的格式生成TO和TD节点
            t = get_time_stamp_node(self.od_data[i][2], self.time_format)

            # 得到TO和TD节点id
            if t_to_id.get(t) is None:
                t_to_id[t] = all_t_id
                all_t_id += 1
            td_id = to_id = t_to_id.get(t)

            region_to_to.add((origin_region_id, to_id))
            to_to_region.add((to_id, origin_region_id))

            region_to_td.add((destination_region_id, td_id))
            td_to_region.add((td_id, destination_region_id))

        # 将对应关系转化为需要的格式
        self.R_to_TO = transform_mapping_relation(region_to_to)
        self.TO_to_R = transform_mapping_relation(to_to_region)

        self.R_to_TD = transform_mapping_relation(region_to_td)
        self.TD_to_R = transform_mapping_relation(td_to_region)

        self._logger.info("finish process meta path RTOR and RTDR")

    def process_hetero_graph(self):
        """
        根据meta path构造异构图
        """
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

        self._logger.info("finish process hetero graph")

    def generate_node_feature(self):
        """
        随机采样初始化节点特征向量
        """
        self.feature = np.random.uniform(-1, 1, size=(self.num_nodes, self.feature_dim))
        self.feature = torch.from_numpy(self.feature)

        self._logger.info("finish generate node feature")

    def process_P_org_dst_and_P_dst_org(self):
        """
        得到P_org_dst和P_dst_org两个矩阵，注意nan的处理
        """
        self.P_org_dst = np.copy(self.od_label)
        self.P_org_dst = self.P_org_dst / self.P_org_dst.sum(axis=0, keepdims=True)
        self.P_org_dst = np.nan_to_num(self.P_org_dst)

        self.P_dst_org = np.copy(self.od_label).transpose()
        self.P_dst_org = self.P_dst_org / self.P_dst_org.sum(axis=0, keepdims=True)
        self.P_dst_org = np.nan_to_num(self.P_dst_org)

        self._logger.info("finish process P_org_dst and P_dst_org")
        
    def process_S_chk(self):
        """
        得到S_chk矩阵，注意nan的处理
        """

        # 得到P_cat_reg矩阵
        poi_category_num = max(self.poi_function_class) + 1
        P_cat_reg = np.zeros((poi_category_num, self.num_nodes), dtype=np.float32)
        
        for poi_id in self.check_in_poi:
            # 首先判断当前POI是否有对应的region
            region_id = judge_POI_validity(poi_id, self.poi2region)
            if region_id is None:
                continue

            poi_category = self.poi_function_class[poi_id]
            P_cat_reg[poi_category][region_id] += 1

        P_cat_reg = P_cat_reg / P_cat_reg.sum(axis=0, keepdims=True)
        P_cat_reg = np.nan_to_num(P_cat_reg)

        P_cat_reg = np.sqrt(P_cat_reg)

        # 利用Hellinger distance得到S_chk矩阵
        self.S_chk = hellinger_distance(self.num_nodes, P_cat_reg)

        self._logger.info("finish process S_chk")

    def process_S_land(self):
        """
        得到S_land矩阵，注意nan的处理
        """

        # 得到P_type_regg矩阵
        land_usage_type_num = len(set(self.land_usage_type))
        P_type_reg = np.zeros((land_usage_type_num, self.num_nodes), dtype=np.float32)
        for region_id, type_land_usage in zip(self.land_usage_region, self.land_usage_type):
            if region_id != 'None':
                region_id = int(eval(region_id))
                type_land_usage = int(type_land_usage)
                P_type_reg[type_land_usage - 1][region_id] += 1

        P_type_reg = P_type_reg / P_type_reg.sum(axis=0, keepdims=True)
        P_type_reg = np.nan_to_num(P_type_reg)

        P_type_reg = np.sqrt(P_type_reg)

        # 利用Hellinger distance得到S_land矩阵
        self.S_land = hellinger_distance(self.num_nodes, P_type_reg)

        self._logger.info("finish process S_land")

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
                'label':{'crime_count_predict': np.array(self.crime_count, dtype=np.float64)}}
