from datetime import datetime

import numpy as np
import geatpy as ea
import torch

from libcity.data.dataset import TrafficRepresentationDataset


class ZEMobDataset(TrafficRepresentationDataset):
    def __init__(self, config):
        super().__init__(config)
        # 求解ppmi_matrix
        self.mobility_events = dict()
        self.zones = dict()
        self.co_occurs = dict()
        self.co_occurs_num = 0

        # 求解G_matrix
        self.arrive_num = np.zeros(len(self.region_ids))
        self.leave_num = np.zeros(len(self.region_ids))
        self.T = np.zeros(len(self.region_ids), len(self.region_ids))
        self.T_bar = np.zeros(len(self.region_ids), len(self.region_ids))
        self.distance = np.zeros(len(self.region_ids), len(self.region_ids))

        # ppmi_matrix为二维矩阵，维度为z*e
        self.ppmi_matrix = []

        # G_matrix为二维矩阵，维度为z*e
        self.G_matrix = []

        # zone的数量
        self.zone_num = 0

        # mobility_event的数量
        self.mobility_event_num = 0

        self.construct_mobility_data()
        self.construct_ppmi_matrix()
        self.construct_gravity_matrix()

    def get_data(self):
        return None, None, None

    def construct_mobility_data(self):
        """
        返回值如下mobility_event、zone、co_occur的列表

        :return: (mobility_pattern, mobility_event, zone)
        """
        mobility_event_index = 0
        for traj, time_list in zip(self.traj_road, self.traj_time):
            # 得到起始zone和起始zone对应的mobility_event
            origin_region = int(list(self.road2region[self.road2region['origin_id'] == traj[0]]['destination_id'])[0])
            origin_date = datetime.strptime(time_list[0], '%Y-%m-%d %H:%M:%S')
            origin_hour = origin_date.hour
            origin_date_type = 'weekday' if origin_date.weekday() in range(5) else 'weekend'
            origin_mobility_event = (origin_region, origin_hour, origin_date_type, 'o')

            # 得到目的zone和目的zone对应的mobility_event
            destination_region = int(
                list(self.road2region[self.road2region['origin_id'] == traj[-1]]['destination_id'])[0]
            )
            destination_date = datetime.strptime(time_list[-1], '%Y-%m-%d %H:%M:%S')
            destination_hour = destination_date.hour
            destination_date_type = 'weekday' if destination_date.weekday() in range(5) else 'weekend'
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

        # 统计zone和mobility_event的数量
        self.zone_num = len(self.region_ids)
        self.mobility_event_num = len(self.mobility_events)

        return

    def construct_ppmi_matrix(self):
        """
        返回ppmi矩阵

        :return: ppmi_matrix
        """
        self.ppmi_matrix = np.zeros((len(self.region_ids), len(self.mobility_events)), dtype=np.float32)
        for region_id in self.co_occurs.keys():
            tmp_mobility_events = self.co_occurs[region_id].keys()
            for mobility_event in tmp_mobility_events:
                mb_id = self.mobility_events[mobility_event][1]
                self.ppmi_matrix[region_id][mb_id] = max(0, np.log2(
                    (self.co_occurs[region_id][mobility_event] * self.co_occurs_num) /
                    (self.zones[region_id] * self.mobility_events[mobility_event][0])
                ))

        return

    def construct_gravity_matrix(self):
        """
        返回G*矩阵

        :return: G_matrix
        """
        self.construct_distance_matrix()

        problem = GravityMatrix()
        algorithm = ea.soea_SEGA_templet(
            problem,
            ea.Population(Encoding='RI', NIND=20),
            MAXGEN=50,
            logTras=1,
            trappedValue=1e-6,
            maxTrappedCount=10
        )
        res = ea.optimize(
            algorithm,
            verbose=True,
            drawing=1,
            outputMsg=False,
            drawLog=False,
            saveFlag=True
        )
        beta = res['Vars'][0][0]

        self.G_matrix = np.ones((len(self.region_ids), len(self.region_ids)), dtype=np.float32)

        return

    def construct_distance_matrix(self):
        centroid = self.region_geometry.centroid
        for i in range(self.zone_num):
            for j in range(self.zone_num):
                self.distance[i][j] = centroid[i].distance(centroid[j])

    def get_data_feature(self):
        """
        返回一个 dict，包含数据集的相关特征

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        return {'ppmi_matrix': self.ppmi_matrix, 'G_matrix': self.G_matrix,
                'region_num': self.zone_num, 'mobility_event_num': self.mobility_event_num}


class GravityMatrix(ea.Problem):
    def __init__(self, mobility_patterns):
        ea.Problem.__init__(
            self, name='GravityMatrix', M=1, maxormins=[-1], Dim=1,
            varTypes=[0], lb=[-10], ub=[10], lbin=[1], ubin=[1]
        )
        self.mobility_patterns = mobility_patterns
        # self.zones

    def evalVars(self, beta):
        pass
        # f = x * np.sin(10 * np.pi * x) + 2.0
        # return f