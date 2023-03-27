import os
import datetime
from datetime import datetime
from geopy.distance import distance
import json
from enum import Enum
import numpy as np
import random
from deap import base, creator, tools, algorithms

from libcity.data.dataset.traffic_representation_dataset import TrafficRepresentationDataset
class ZEMobDataset(TrafficRepresentationDataset):
    def __init__(self,config):
        self.config = config
        self.dataset = self.config.get('dataset', '')
        self.data_path = './raw_data/' + self.dataset + '/'
        self.geo_file = self.config.get('geo_file', self.dataset)
        self.rel_file = self.config.get('rel_file', self.dataset)
        self.dyna_file = self.config.get('dyna_file',self.dataset)
        assert os.path.exists(self.data_path + self.geo_file + '.geo')
        assert os.path.exists(self.data_path + self.rel_file + '.rel')
        assert os.path.exists(self.data_path + self.dyna_file + '.dyna')
        super().__init__(config)
        self.getZones()
        self.process_patterns()
        self.process_events()
        self.getZE()
        self.getM()
        self.getDistanceGraph()
        self.beta_we,self.Gwe=self.getG(TimeSlot.DAYTYPE.WEEKEND)
        self.beta_wd,self.Gwd=self.getG(TimeSlot.DAYTYPE.WEEKDAY)




    def get_data(self):
        return None,None,None


    def getZones(self):
        """
        加载Zone Set
        :return:
        """
        assert self.representation_object == "region"
        self.zones=list(self.region_ids)
        self.z_num=len(self.zones)

    def process_patterns(self):
        """
        加载 Human mobility pattern Set

        :return:
        """
        assert self.representation_object == "region"
        self.patterns=[]
        for i in range(len(self.traj_road)):
            traj=self.traj_road[i]
            zO = list(self.road2region[self.road2region['origin_id'] == traj[0]]['destination_id'])[0]
            zD = list(self.road2region[self.road2region['origin_id'] == traj[-1]]['destination_id'])[0]
            # '2015-11-01 09:35:48'
            tO=TimeSlot(self.traj_time[0])
            tD =TimeSlot(self.traj_time[-1])
            self.patterns.append(Pattern(zO,zD,tO,tD))

    def process_events(self):
        """
        加载Human mobility event Set
        :return:
        """
        assert self.representation_object == "region"
        self.events=[]
        for p in self.patterns:
            self.events.append(Event(p.zO,p.tO,Event.STA.LEAVE))
            self.events.append(Event(p.zD,p.tD,Event.STA.ARRIVE))
        self.e_num=len(self.events)

    def getZE(self):
        """
        计算全部的#(z,e)
        计算gama
        需要已知events的数量，所以必须先生成events
        events的数目一定是偶数
        :return:
        """
        self.ZE=np.zeros((len(self.zones), len(self.events)), dtype=np.int)
        idx_e=0
        for p in self.patterns:
            # 相同顺序遍历patterns保证events顺序一致
            idx_z=self.zones.index(p.zO)
            # zO~Event(p.zD,p.tD,Event.STA.ARRIVE)
            self.ZE[idx_z][idx_e+1]+=1
            idx_z = self.zones.index(p.zD)
            # zD~Event(p.zO,p.tO,Event.STA.LEAVE)
            self.ZE[idx_z][idx_e]+=1
            idx_e+=2
        self.gama=np.sum(self.ZE)




    def getM(self):
        self.M = np.dot(self.ZE,self.gama/(self.z_num*self.e_num))
        self.M = np.log2(self.M)
        self.M = np.maximum(self.M,0)
        self._logger.info("finish consturcting M")




    def getDistanceGraph(self):
        """
        构建距离矩阵distance_graph
        :return:distance_graph
        """
        self.distance_graph=np.zeros((self.z_num, self.z_num), dtype=np.float32)
        self.centroid = self.region_geometry.centroid
        for i in range(self.z_num):
            for j in range(i, self.z_num):
                distance = self.centroid[i].distance(self.centroid[j])
                self.distance_graph[i][j] = distance
                self.distance_graph[j][i] = distance
        self._logger.info("finish consturcting distance graph")


    def getG(self,day_type):
        """
        计算G矩阵
        :param day_type: Event.STA.ARRIVE
        :return:beta,G
        """
        # A:total number of mobility patterns arrive at zone z
        # T:observed mobility pattern number from zO to zD
        # P:total number of mobility patterns leave zO
        G = np.zeros((self.z_num, self.z_num), dtype=np.float32)
        A = np.zeros(self.z_num, dtype=np.float32)
        P = np.zeros(self.z_num, dtype=np.float32)
        T = np.zeros((self.z_num, self.z_num), dtype=np.float32)
        beta = np.random.rand()
        # 初始化A,P,T
        for pattern in self.patterns:
            # 根据出发时刻判断是weekday还是weekend
            if pattern.tO.day_type==day_type:

                A[self.zones.index(pattern.zD)] += 1
                T[self.zones.index(pattern.zO)][self.zones.index(pattern.zD)] += 1
                P[self.zones.index(pattern.zO)] += 1
        def cal_G(beta):
            """
            根据beta计算G
            :return:G
            """
            F = np.exp(-1.0 * beta * self.distance_graph)
            G = F * A[:np.newaxis]
            G = G / np.sum(G, axis=1, keepdims=True)
            return G
        def NSGA2():
            """
            使用NSGA2优化beta
            :return:beta
            """
            def evaluate(individual):
                """
                评估beta
                :param beta:
                :return:
                """
                G = cal_G(individual[0])
                T_hat = P[:np.newaxis] * G
                mse = np.sum((T - T_hat) ** 2)
                return mse,

            # 定义单目标问题的创建器
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMin)

            # 初始化DEAP的工具箱
            toolbox = base.Toolbox()
            toolbox.register("attr_bool", random.randint, 0, 1)
            toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, 10)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)
            toolbox.register("evaluate", evaluate)
            toolbox.register("mate", tools.cxTwoPoint)
            toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
            toolbox.register("select", tools.selNSGA2)

            # 定义算法参数
            POP_SIZE = 50
            GENERATIONS = 50
            CXPB = 0.9
            MUTPB = 0.1

            # 初始化种群并进行进化
            pop = toolbox.population(n=POP_SIZE)
            for gen in range(GENERATIONS):
                offspring = algorithms.varAnd(pop, toolbox, CXPB, MUTPB)
                fits = toolbox.map(toolbox.evaluate, offspring)
                for fit, ind in zip(fits, offspring):
                    ind.fitness.values = fit
                pop = toolbox.select(offspring, k=len(pop))

            # 输出最优解
            best_ind = tools.selBest(pop, k=1)[0]
            return best_ind
        beta=NSGA2()
        G=cal_G(beta)
        return beta,G

    def getG_star(self):
        """
        计算G_star,将原文中G_star转换为一个Z*E的矩阵
        :return:
        """
        self.G_star=np.zeros((self.z_num, self.e_num), dtype=np.float32)
        for i in range(self.z_num):
            for j in range(self.e_num):
                if self.ZE[i][j]==1:
                    zD=self.events[j].z
                    if self.events[j].t.sta==TimeSlot.DAYTYPE.WEEKEND:
                        self.G_star[i][j]=self.Gwe[i][self.zones.index(zD)]
                    else:
                        self.G_star[i][j] = self.Gwd[i][self.zones.index(zD)]


    def get_data_feature(self):
        """
        返回一个 dict，包含数据集的相关特征

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        return {
            "M": self.M, "z_num": self.z_num,"e_num":self.e_num,"G_star":self.G_star,
        }

    """
        对应关系：
        zone:geo_id
        p(Human mobility pattern):Pattern
        e(Human mobility event):Event
    """

    class Pattern():

        def __init__(self, zO, zD, tO, tD):
            """
            构建一个Human mobility pattern实例
            :param zO:geo_id
            :param zD:geo_id
            :param tO:TimeSlot
            :param tD:TimeSlot
            :return:
            """
            self.zO = zO
            self.zD = zD
            self.tO = tO
            self.tD = tD

    class Event():
        class STA(Enum):
            LEAVE = 0
            ARRIVE = 1

        def __init__(self, z, t, sta):
            """
            构建一个Human mobility event
            :param z:geo_id
            :param t:TimeSlot
            :param sta:0-leave,1-arrive
            """
            self.z = z
            self.t = t
            self.sta = sta

    class TimeSlot():
        class DAYTYPE(Enum):
            WEEKDAY = 0
            WEEKEND = 1

        def __init__(self, date_str):
            """

            :param date_str: '2015-11-01 09:35:48'
            :return:
            """
            date_obj = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
            weekday = date_obj.weekday()
            if weekday < 5:
                self.day_type = self.DAYTYPE.WEEKDAY  # 工作日
            else:
                self.day_type = self.DAYTYPE.WEEKEND  # 周末
            self.t = int(date_obj.hour)







