import os
from datetime import datetime

import numpy as np
import geatpy as ea

from libcity.data.dataset import TrafficRepresentationDataset


class ZEMobDataset(TrafficRepresentationDataset):
    def __init__(self, config):
        super().__init__(config)

        # 求解ppmi_matrix，三个字典分别存储了event的出现次数，zone的出现次数，zone和event的共现次数
        self.mobility_events = dict()
        self.zones = dict()
        self.co_occurs = dict()
        self.co_occurs_num = 0

        # 求解G_matrix，三组变量分别对应文章中的A、P、T
        self.max_gen = config.get('MaxGen')
        self.NIND = config.get('NIND')
        self.arrive_num_weekday = np.zeros(len(self.region_ids))
        self.arrive_num_weekend = np.zeros(len(self.region_ids))
        self.leave_num_weekday = np.zeros(len(self.region_ids))
        self.leave_num_weekend = np.zeros(len(self.region_ids))
        self.T_weekday = np.zeros((len(self.region_ids), len(self.region_ids)), dtype=np.float32)
        self.T_weekend = np.zeros((len(self.region_ids), len(self.region_ids)), dtype=np.float32)
        self.distance = np.zeros((len(self.region_ids), len(self.region_ids)), dtype=np.float32)

        # ppmi_matrix为二维矩阵，维度为z*e
        self.ppmi_matrix = []

        # G_matrix为二维矩阵，维度为z*e
        self.G_matrix = []

        # zone的数量
        self.zone_num = 0

        # mobility_event的数量
        self.mobility_event_num = 0

        # 设置中间变量的缓存路径
        if not os.path.exists('./libcity/cache/ZEMob_{}'.format(self.dataset)):
            os.mkdir('./libcity/cache/ZEMob_{}'.format(self.dataset))
        self.distance_matrix_path = './libcity/cache/ZEMob_{}/distance_matrix.npy'.format(self.dataset)
        self.ppmi_matrix_path = './libcity/cache/ZEMob_{}/ppmi_matrix.npy'.format(self.dataset)

        self.construct_mobility_data()
        self.construct_ppmi_matrix()
        self.construct_gravity_matrix()

    def get_data(self):
        return None, None, None

    def construct_mobility_data(self):
        """
        统计mobility event和co-occurs的信息，用于创建ppmi matrix和gravity matrix

        :return:
        """
        mobility_event_index = 0
        for traj, time_list in zip(self.traj_road, self.traj_time):
            # 得到起始zone和起始zone对应的mobility_event
            origin_region = int(list(self.road2region[self.road2region['origin_id'] == traj[0]]['destination_id'])[0])
            origin_date = datetime.strptime(time_list[0], '%Y-%m-%d %H:%M:%S')
            origin_hour = origin_date.hour
            origin_date_type = 1 if origin_date.weekday() in range(5) else 0
            origin_mobility_event = (origin_region, origin_hour, origin_date_type, 'o')

            # 得到目的zone和目的zone对应的mobility_event
            destination_region = int(
                list(self.road2region[self.road2region['origin_id'] == traj[-1]]['destination_id'])[0]
            )
            destination_date = datetime.strptime(time_list[-1], '%Y-%m-%d %H:%M:%S')
            destination_hour = destination_date.hour
            destination_date_type = 1 if destination_date.weekday() in range(5) else 0
            destination_mobility_event = (destination_region, destination_hour, destination_date_type, 'd')

            # 计算每个mobility_event出现的次数，同时给每个mobility一个index
            # 即一个一维字典，key是mobility_event，value是一个二元列表，第一个值是co-occur的次数，第二个值是index
            if self.mobility_events.get(origin_mobility_event) is None:
                self.mobility_events[origin_mobility_event] = [1, mobility_event_index]
                mobility_event_index += 1
            else:
                self.mobility_events[origin_mobility_event][0] += 1
            if self.mobility_events.get(destination_mobility_event) is None:
                self.mobility_events[destination_mobility_event] = [1, mobility_event_index]
                mobility_event_index += 1
            else:
                self.mobility_events[destination_mobility_event][0] += 1

            # 计算每个zone出现的次数，即一个一维字典，key是zone，value是co-occur的次数
            if self.zones.get(origin_region) is None:
                self.zones[origin_region] = 1
            else:
                self.zones[origin_region] += 1
            if self.zones.get(destination_region) is None:
                self.zones[destination_region] = 1
            else:
                self.zones[destination_region] += 1

            # 计算每种co-occur的次数，即一个二维字典，第一维的key是zone，第二维的key是mobility_event，value是co-occur的次数
            if self.co_occurs.get(origin_region) is None:
                event_dict = dict()
                event_dict[destination_mobility_event] = 1
                self.co_occurs[origin_region] = event_dict
            else:
                event_dict = self.co_occurs[origin_region]
                if event_dict.get(destination_mobility_event) is None:
                    event_dict[destination_mobility_event] = 1
                else:
                    event_dict[destination_mobility_event] += 1
            if self.co_occurs.get(destination_region) is None:
                event_dict = dict()
                event_dict[origin_mobility_event] = 1
                self.co_occurs[destination_region] = event_dict
            else:
                event_dict = self.co_occurs[destination_region]
                if event_dict.get(origin_mobility_event) is None:
                    event_dict[origin_mobility_event] = 1
                else:
                    event_dict[origin_mobility_event] += 1

            # 统计总的co-occur的数量
            self.co_occurs_num += 2

            # 计算A、P、T
            if origin_date_type == 1:
                self.arrive_num_weekday[destination_region] += 1
                self.leave_num_weekday[origin_region] += 1
                self.T_weekday[origin_region][destination_region] += 1
            else:
                self.arrive_num_weekend[destination_region] += 1
                self.leave_num_weekend[origin_region] += 1
                self.T_weekend[origin_region][destination_region] += 1

        # 统计zone和mobility_event的数量
        self.zone_num = len(self.region_ids)
        self.mobility_event_num = len(self.mobility_events)

        self._logger.info("finish constructing mobility basic data")

    def construct_ppmi_matrix(self):
        """
        创建ppmi矩阵

        :return:
        """
        if os.path.exists(self.ppmi_matrix_path):
            self.ppmi_matrix = np.load(self.ppmi_matrix_path)
            self._logger.info("finish constructing ppmi matrix")
            return
        self.ppmi_matrix = np.zeros((len(self.region_ids), len(self.mobility_events)), dtype=np.float32)
        for region_id in self.co_occurs.keys():
            tmp_mobility_events = self.co_occurs[region_id].keys()
            for mobility_event in tmp_mobility_events:
                mb_id = self.mobility_events[mobility_event][1]
                self.ppmi_matrix[region_id][mb_id] = max(0, np.log2(
                    (self.co_occurs[region_id][mobility_event] * self.co_occurs_num) /
                    (self.zones[region_id] * self.mobility_events[mobility_event][0])
                ))
        np.save(self.ppmi_matrix_path, self.ppmi_matrix)
        self._logger.info("finish constructing ppmi matrix")

    def construct_gravity_matrix(self):
        """
        创建gravity matrix

        :return:
        """
        self.construct_distance_matrix()

        problem = GravityMatrix(self.zone_num, self.arrive_num_weekday, self.leave_num_weekday, self.T_weekday, self.distance)
        algorithm = ea.soea_SEGA_templet(
            problem,
            ea.Population(Encoding='RI', NIND=self.NIND),
            MAXGEN=self.max_gen,
            trappedValue=1e-6,
            maxTrappedCount=10,
            logTras=0,
        )
        res = ea.optimize(
            algorithm,
            verbose=True,
            drawing=0,
            outputMsg=False,
            drawLog=False,
            saveFlag=False
        )
        beta_weekday = res['Vars'][0][0]

        problem = GravityMatrix(self.zone_num, self.arrive_num_weekend, self.leave_num_weekend, self.T_weekend, self.distance)
        algorithm = ea.soea_SEGA_templet(
            problem,
            ea.Population(Encoding='RI', NIND=self.NIND),
            MAXGEN=self.max_gen,
            trappedValue=1e-6,
            maxTrappedCount=10,
            logTras=0,
        )
        res = ea.optimize(
            algorithm,
            verbose=True,
            drawing=0,
            outputMsg=False,
            drawLog=False,
            saveFlag=False
        )
        beta_weekend = res['Vars'][0][0]

        F = np.exp((-beta_weekday) * self.distance)
        G_weekday = (F * self.arrive_num_weekday) / np.matmul(F, self.arrive_num_weekday.reshape(self.zone_num, 1))

        F = np.exp((-beta_weekend) * self.distance)
        G_weekend = (F * self.arrive_num_weekend) / np.matmul(F, self.arrive_num_weekend.reshape(self.zone_num, 1))

        self.G_matrix = np.zeros((self.zone_num, self.mobility_event_num))
        for zone_id in range(self.zone_num):
            for mobility_event in self.mobility_events.keys():
                mb_id = self.mobility_events[mobility_event][1]
                mb_type = mobility_event[2]
                mb_zone_id = mobility_event[0]
                if mb_type == 1:
                    self.G_matrix[zone_id][mb_id] = G_weekday[zone_id][mb_zone_id]
                else:
                    self.G_matrix[zone_id][mb_id] = G_weekend[zone_id][mb_zone_id]
        self._logger.info("finish constructing gravity matrix, beta_wd is {}, beta_we is {}".format(str(beta_weekday), str(beta_weekend)))

    def construct_distance_matrix(self):
        if os.path.exists(self.distance_matrix_path):
            self.distance = np.load(self.distance_matrix_path)
            self._logger.info("finish constructing distance matrix")
            return
        centroid = self.region_geometry.centroid
        for i in range(self.zone_num):
            for j in range(i, self.zone_num):
                distance = centroid[i].distance(centroid[j])
                self.distance[i][j] = distance
                self.distance[j][i] = distance
        np.save(self.distance_matrix_path, self.distance)
        self._logger.info("finish constructing distance matrix")

    def get_data_feature(self):
        """
        返回一个 dict，包含数据集的相关特征

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        return {'ppmi_matrix': self.ppmi_matrix, 'G_matrix': self.G_matrix, 
                'region_num': self.zone_num, 'mobility_event_num': self.mobility_event_num,
                'label': {"function_cluster": np.array(self.function)}}


# 遗传算法求解gravity matrix
class GravityMatrix(ea.Problem):
    def __init__(self, zone_num, A, P, T, D):
        ea.Problem.__init__(
            self, name='GravityMatrix', M=1, maxormins=[1], Dim=1,
            varTypes=[0], lb=[-50], ub=[50], lbin=[1], ubin=[1]
        )
        self.zone_num = zone_num
        self.A = A
        self.P = P
        self.T = T
        self.D = D

    def evalVars(self, betas):
        res = np.zeros_like(betas)
        for i in range(len(res)):
            beta = betas[i][0]
            F = np.exp((-beta) * self.D)
            G_denominator = np.matmul(F, self.A.reshape(self.zone_num, 1))
            G_numerator = np.matmul(self.P.reshape(self.zone_num, 1), self.A.reshape(1, self.zone_num)) * F
            T_roof = G_numerator / G_denominator
            res[i][0] = np.sum((self.T - T_roof) ** 2)
        return res
