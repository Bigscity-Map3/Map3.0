import os
import datetime
from datetime import datetime
from enum import Enum
import numpy as np
import random
from deap import base, creator, tools, algorithms
from tqdm import tqdm
from libcity.data.dataset.traffic_representation_dataset import TrafficRepresentationDataset


class ZEMobDataset(TrafficRepresentationDataset):
    def __init__(self, config):
        self.config = config
        self.dataset = self.config.get('dataset', '')
        self.data_path = './raw_data/' + self.dataset + '/'
        self.geo_file = self.config.get('geo_file', self.dataset)
        self.rel_file = self.config.get('rel_file', self.dataset)
        self.dyna_file = self.config.get('dyna_file', self.dataset)
        if not os.path.exists('./libcity/cache/ZEMob_{}'.format(self.dataset)):
            os.mkdir('./libcity/cache/ZEMob_{}'.format(self.dataset))
        self.distance_graph_path = './libcity/cache/ZEMob_{}/distance_graph.npy'.format(self.dataset)
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
        self.beta_we, self.Gwe = self.getG(TimeSlot.DAYTYPE.WEEKEND)
        self.beta_wd, self.Gwd = self.getG(TimeSlot.DAYTYPE.WEEKDAY)

    def get_data(self):
        return None, None, None

    def getZones(self):
        """
        加载Zone Set
        :return:
        """
        assert self.representation_object == "region"
        self.zones = list(self.region_ids)
        self.z_num = len(self.zones)
        self._logger.info("finish loading zones, zone total num {}".format(self.z_num))

    def process_patterns(self):
        """
        加载 Human mobility pattern Set

        :return:
        """
        assert self.representation_object == "region"
        self.patterns = []
        for i in range(len(self.traj_road)):
            traj = self.traj_road[i]
            zO = list(self.road2region[self.road2region['origin_id'] == traj[0]]['destination_id'])[0]
            zD = list(self.road2region[self.road2region['origin_id'] == traj[-1]]['destination_id'])[0]
            # '2015-11-01 09:35:48'
            tO = TimeSlot(self.traj_time[i][0])
            tD = TimeSlot(self.traj_time[i][-1])
            self.patterns.append(Pattern(zO, zD, tO, tD))
        self._logger.info("finish processing patterns, patterns total num {}".format( len(self.patterns)))
    def process_events(self):
        """
        加载Human mobility event Set
        :return:
        """
        assert self.representation_object == "region"
        self.events = []
        for p in self.patterns:
            self.events.append(Event(p.zO, p.tO, Event.STA.LEAVE))
            self.events.append(Event(p.zD, p.tD, Event.STA.ARRIVE))
        self.e_num = len(self.events)
        self._logger.info("finish processing events, events total num {}".format(len(self.events)))

    def getZE(self):
        """
        计算全部的#(z,e)
        计算gama
        需要已知events的数量，所以必须先生成events
        events的数目一定是偶数
        :return:
        """
        self.ZE = np.zeros((len(self.zones), len(self.events)), dtype=np.int)
        idx_e = 0
        for p in self.patterns:
            # 相同顺序遍历patterns保证events顺序一致
            idx_z = self.zones.index(p.zO)
            # zO~Event(p.zD,p.tD,Event.STA.ARRIVE)
            self.ZE[idx_z][idx_e + 1] += 1
            idx_z = self.zones.index(p.zD)
            # zD~Event(p.zO,p.tO,Event.STA.LEAVE)
            self.ZE[idx_z][idx_e] += 1
            idx_e += 2
        self.gama = np.sum(self.ZE)
        self._logger.info("finish building ZE matrix, shape is {} ,gama is {}".format(self.ZE.shape,self.gama))

    def getM(self):
        self.M = np.dot(self.ZE, self.gama / (self.z_num * self.e_num))
        self.M = np.log2(self.M+ 1e-10) # 防止log(0)
        self.M = np.maximum(self.M, 0)
        self._logger.info("finish consturcting M")

    def getDistanceGraph(self):
        """
        构建距离矩阵distance_graph
        :return:distance_graph
        """
        if os.path.exists(self.distance_graph_path):
            self.distance_graph = np.load(self.distance_graph_path)
            self._logger.info("finish consturcting distance graph")
            return
        self.distance_graph = np.zeros((self.z_num, self.z_num), dtype=np.float32)
        self.centroid = self.region_geometry.centroid
        with tqdm(total=self.z_num*self.z_num, desc="consturcting distance graph") as pbar:
            for i in range(self.z_num):
                for j in range(i, self.z_num):
                    distance = self.centroid[i].distance(self.centroid[j])
                    self.distance_graph[i][j] = distance
                    self.distance_graph[j][i] = distance
                    pbar.update(2)
        np.save(self.distance_graph_path, self.distance_graph)
        self._logger.info("finish consturcting distance graph")

    def getG(self, day_type):
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
        self._logger.info("begin initializing A,T,P matrix")
        # 初始化A,P,T
        for pattern in self.patterns:
            # 根据出发时刻判断是weekday还是weekend
            if pattern.tO.day_type == day_type:
                A[self.zones.index(pattern.zD)] += 1
                T[self.zones.index(pattern.zO)][self.zones.index(pattern.zD)] += 1
                P[self.zones.index(pattern.zO)] += 1
        self._logger.info("finish initializing A: {},T:{},P:{}".format(A.shape,T.shape,P.shape))

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

            POPULATION_SIZE = 10
            NUM_GENERATIONS = 10
            # 创建问题的优化器和个体的数据结构
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
            creator.create("Individual", float, fitness=creator.FitnessMin)

            toolbox = base.Toolbox()
            toolbox.register("attr_float", random.uniform, -5.0, 5.0)
            toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=1)
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)

            # 注册评价函数
            toolbox.register("evaluate", evaluate)

            # 注册交叉和变异操作
            toolbox.register("mate", tools.cxTwoPoint)
            toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)

            # 注册选择操作
            toolbox.register("select", tools.selNSGA2)
            pop = list(toolbox.population(n=POPULATION_SIZE))
            # 迭代优化
            with tqdm(total=NUM_GENERATIONS, desc="NSGA2 GENERATION: ") as pbar:
                for gen in range(NUM_GENERATIONS):
                    # 评价种群中的个体
                    fitnesses = list(map(toolbox.evaluate, pop))
                    for ind, fit in zip(pop, fitnesses):
                        ind.fitness.values = fit

                    # 选择下一代
                    pop = toolbox.select(pop, len(pop))

                    # 交叉和变异
                    offspring = tools.selTournamentDCD(pop, len(pop))
                    offspring = [toolbox.clone(ind) for ind in offspring]

                    for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
                        if random.random() < 0.5:
                            toolbox.mate(ind1, ind2)
                            del ind1.fitness.values
                            del ind2.fitness.values

                    for ind in offspring:
                        if random.random() < 0.2:
                            toolbox.mutate(ind)
                            del ind.fitness.values

                    # 更新种群
                    pop = offspring
                    pbar.update(1)

            # 输出最优个体
            best_ind = tools.selBest(pop, 1)[0]
            self._logger.info("Best individual is: %s\nwith fitness: %s" % (best_ind, best_ind.fitness.values))
            return best_ind


        beta = NSGA2()
        self._logger.info("finish caculating beta , value is {}".format(beta))
        G = cal_G(beta)
        self._logger.info("finish caculating G:{}".format(G.shape))
        return beta, G

    def getG_star(self):
        """
        计算G_star,将原文中G_star转换为一个Z*E的矩阵
        :return:
        """
        self.G_star = np.zeros((self.z_num, self.e_num), dtype=np.float32)
        for i in range(self.z_num):
            for j in range(self.e_num):
                zD = self.events[j].z
                if self.events[j].t.sta == TimeSlot.DAYTYPE.WEEKEND:
                    self.G_star[i][j] = self.Gwe[i][self.zones.index(zD)]
                else:
                    self.G_star[i][j] = self.Gwd[i][self.zones.index(zD)]



    def get_data_feature(self):
        """
        返回一个 dict，包含数据集的相关特征

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        return {
            "M": self.M, "z_num": self.z_num, "e_num": self.e_num, "G_star": self.G_star,
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
