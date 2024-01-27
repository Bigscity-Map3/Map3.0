import numpy as np 
import time
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import SGD, Adam, ASGD, RMSprop
from torch.utils.data import DataLoader
from torch.nn.functional import log_softmax, softmax
import torch.nn.functional as F
import pickle

import math


# region_dict_path = "./clear_data/training_dict.pickle"
# region2poi_path = "./clear_data/region2poi.pickle"
# rs_ratio_path = "./clear_data/rs_ratio.pickle"


class SSLData:

    def __init__(self, region_dict_path, region2poi_path, rs_ratio_path):

        self.region_dict = pickle.load(open(region_dict_path, "rb"))
        self.sampling_pool = [_i for _i in range(len(self.region_dict))]

        region2poi = pickle.load(open(region2poi_path, "rb"))
        self.level_c2p = region2poi["level_c2p"]
        self.level_p2c = region2poi["level_p2c"]
        self.poi_num = len(region2poi["node_dict"])

        self.rs_ratio = pickle.load(open(rs_ratio_path, "rb"))

    def get_region(self, idx):
        pois = self.region_dict[idx]["poi"]

        poi_set = []
        for poi in pois:
            _id = poi[1]
            l_vector = [poi[2].index(1), poi[3].index(1)]
            poi_set.append([_id, l_vector])

        pickup_matrix = self.region_dict[idx]["pickup_matrix"]
        dropoff_matrix = self.region_dict[idx]["dropoff_matrix"]

        pickup_matrix = pickup_matrix / pickup_matrix.sum()
        where_are_NaNs = np.isnan(pickup_matrix)
        pickup_matrix[where_are_NaNs] = 0

        dropoff_matrix = dropoff_matrix / dropoff_matrix.sum()
        where_are_NaNs = np.isnan(dropoff_matrix)
        dropoff_matrix[where_are_NaNs] = 0

        flow_matrix = [pickup_matrix, dropoff_matrix]

        return poi_set, flow_matrix

class SAEncoder(nn.Module):

    def __init__(self, d_input, d_model, n_head, _type):
        super(SAEncoder, self).__init__()

        self.d_input = d_input
        self.d_model = d_model
        self.n_head = n_head

        self.linear_k = nn.Linear(self.d_input, self.d_model * self.n_head) 
        self.linear_v = nn.Linear(self.d_input, self.d_model * self.n_head) 
        self.linear_q = nn.Linear(self.d_input, self.d_model * self.n_head)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self._type=_type

    def fusion(self, v, f_type):
        if f_type == "concat":
            output = v.view(-1, self.d_model * self.n_head)
        if f_type == "avg":
            output = torch.mean(v, dim=0)
        return output
    
    def forward(self, x):
        q = self.linear_q(x) 
        k = self.linear_k(x)
        v = self.linear_v(x)

        q_ = q.view(self.n_head, self.d_model) 
        k_ = k.view(self.n_head, self.d_model)
        v_ = v.view(self.n_head, self.d_model)

        head, d_tensor = k_.size()
        score = (q_.matmul(k_.transpose(0, 1))) / math.sqrt(d_tensor)
        score = self.softmax(score)

        v_ = self.relu(v_)
        v = score.matmul(v_)

        output = self.fusion(v, self._type)
        return output


class CroSAEncoder(nn.Module):

    def __init__(self, d_input_query, d_input_kv, d_model, n_head, func):
        super(SAEncoder, self).__init__()

        self.d_input_query = d_input_query
        self.d_input_kv = d_input_kv
        self.d_model = d_model
        self.n_head = n_head

        self.linear_k = nn.Linear(self.d_input_kv, self.d_model * self.n_head) 
        self.linear_v = nn.Linear(self.d_input_kv, self.d_model * self.n_head) 
        self.linear_q = nn.Linear(self.d_input_query, self.d_model * self.n_head)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.func=func

    def fusion(self, v, f_type):
        if f_type == "concat":
            output = v.view(-1, self.d_model * self.n_head)
        if f_type == "avg":
            output = torch.mean(v, dim=0)
        return output
    
    def forward(self, q, kv):
        q = self.linear_q(q) 
        k = self.linear_k(kv)
        v = self.linear_v(kv)

        q_ = q.view(self.n_head, self.d_model) 
        k_ = k.view(self.n_head, self.d_model)
        v_ = v.view(self.n_head, self.d_model)

        head, d_tensor = k_.size()
        score = (q_.matmul(k_.transpose(0, 1))) / math.sqrt(d_tensor)
        score = self.softmax(score)
       
        v_ = self.relu(v_)
        v = score.matmul(v_)

        output = self.fusion(v, _type)
        if self.func == "relu":
            output = self.relu(output)

        return output

rs_type = "random"

class Flatten(torch.nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
class POI_SSL(torch.nn.Module):

    def __init__(self, ssl_data, neg_size, emb_size, attention_size, temp, extractor,device): 
        super(POI_SSL,self).__init__()

        self.ssl_data = ssl_data
        self.device = device
        self.init_basic_conf(neg_size, emb_size, attention_size, temp, extractor)

        
    def init_basic_conf(self, neg_size, emb_size, attention_size, temp, extractor):
        self.neg_size = neg_size
        self.emb_size = emb_size
        self.attention_size = attention_size
        self.bin_num = 10

        self.poi_num = self.ssl_data.poi_num
        self.temp = temp

        self.extractor = extractor

        self.W_poi = None

        if self.extractor == "CNN":
            self.poi_net = torch.nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=4, kernel_size=7),
                Flatten(),
                nn.Linear(964, self.emb_size)
            ).to(self.device)

        if self.extractor == "MLP":
            self.poi_net = torch.nn.Sequential(
                nn.Linear(247, self.emb_size),
                nn.ReLU(),
            ).to(self.device)

        if self.extractor == "SA":
            self.poi_net = SAEncoder(d_input=247, d_model=16, n_head=3).to(self.device)


    def generate_attention(self, W_parents, W_children, mask):
        _input = torch.cat([W_parents, W_children], dim=2)
        output = self.tree_att_net(_input)
        pre_attention = torch.matmul(output, self.v_attention)

        pre_attention = pre_attention + mask
        attention = torch.softmax(pre_attention, dim=1)
        
        return attention

    def tree_gcn(self):
        parentsList, childrenList, maskList = [], [], []

        for i, p2c_i in enumerate(self.ssl_data.level_p2c[::-1]):
            children = list(p2c_i.values())
            max_n_children = max(len(x) for x in children)
            mask = []

            for k in children:
                cur_mask = [0.0] * len(k)
                if len(k) < max_n_children:
                    cur_mask += [-10**13] * (max_n_children - len(k))
                    k += [0] * (max_n_children - len(k))
                mask.append(cur_mask)

            parents = []
            for p in p2c_i.keys():
                parents.append([p] * max_n_children)

            children = torch.Tensor(children).type(LType)
            parents = torch.Tensor(parents).type(LType)
            mask = torch.Tensor(mask).type(FType)
            
            parentsList.append(parents)
            childrenList.append(children)
            maskList.append(mask)

        W_emb_temp = self.W_poi.clone() + 0.
        for i, (parents, children, mask) in enumerate(zip(parentsList, childrenList, maskList)):
            W_parents = self.W_poi[parents]
            if i == 0:
                W_children = self.W_poi[children]
            else:
                W_children = W_emb_temp[children]

            tempAttention = self.generate_attention(W_parents, W_children, mask)
            tempEmb = (W_children * tempAttention[:,:,None]).sum(dim=1)

            W_emb_temp = torch.index_copy(W_emb_temp, 0, parents[:, 0], tempEmb)

        parentsList, childrenList, maskList = [], [], []
        for i, c2p_i in enumerate(self.ssl_data.level_c2p[::-1]):
            parents = list(c2p_i.values())
            max_n_parents = max(len(x) for x in parents)
            mask = []

            for k in parents:
                cur_mask = [0.0] * len(k)
                if len(k) < max_n_parents:
                    cur_mask += [-10**13] * (max_n_parents - len(k))
                    k += [0] * (max_n_parents - len(k))
                mask.append(cur_mask)

            children = []
            for c in c2p_i.keys():
                children.append([c] * max_n_parents)

            children = torch.Tensor(children).type(LType)
            parents = torch.Tensor(parents).type(LType)
            mask = torch.Tensor(mask).type(FType)
            
            parentsList.append(parents)
            childrenList.append(children)
            maskList.append(mask)

        for i, (parents, children, mask) in enumerate(zip(parentsList, childrenList, maskList)):
            W_children, W_parents = W_emb_temp[children], W_emb_temp[parents]

            tempAttention = self.generate_attention(W_children, W_parents, mask)
            # tempEmb = (W_parents * tempAttention[:,:,None]).sum(axis=1)
            tempEmb = (W_parents * tempAttention[:,:,None]).sum(dim=1)

            # W_emb_temp[children[:, 0]] = tempEmb
            W_emb_temp = torch.index_copy(W_emb_temp, 0, children[:, 0], tempEmb)

        self.W_emb_temp = W_emb_temp

        return W_emb_temp

    def location_attention(self, loc_emb_one, loc_emb_two):
        _input = torch.cat([loc_emb_one, loc_emb_two], axis=1)

        output = self.location_att_net(_input)
        pre_attention = torch.matmul(output, self.l_attention)

        attention = torch.softmax(pre_attention, dim=0)
        return attention

    def agg_region_emb(self, poi_set, W_emb_temp):
        p_node_dict = self.ssl_data.region_dict[0]["node_dict"] 
        poi_f = np.zeros(len(p_node_dict))
        for poi in poi_set:
            poi_id = poi[0]
            poi_f[poi_id] += 1

        if np.sum(poi_f) != 0:
            poi_f = poi_f / np.sum(poi_f)
        poi_f = torch.Tensor(poi_f).type(FType).to(device)

        if self.extractor == "CNN":
            poi_f = poi_f.unsqueeze(0)
            poi_f = poi_f.unsqueeze(0)
            temp_emb = self.poi_net(poi_f)

        if self.extractor == "MLP":
            temp_emb = self.poi_net(poi_f)

        if self.extractor == "SA":
            temp_emb = self.poi_net(poi_f)

        region_emb = temp_emb
        region_emb = region_emb.squeeze()

        return region_emb

    def add_aug(self, poi_set, _ratio):
        add_poi_set = []
        for poi in poi_set:
            add_poi_set.append(poi)
            ratio = random.random()
            if ratio < _ratio:
                add_poi_set.append(poi)
        return add_poi_set

    def delete_aug(self, poi_set, _ratio):
        de_poi_set = []
        for poi in poi_set:
            ratio = random.random()
            if ratio > _ratio:
                de_poi_set.append(poi)
        if not de_poi_set:
            de_poi_set = [poi_set[0]]
        return de_poi_set

    def replace_aug(self, poi_set, _ratio):
        replace_poi_set = []
        for poi in poi_set:
            new_poi = poi
            ratio = random.random()
            if ratio < _ratio:
                new_poi[0] = random.randint(0, self.ssl_data.poi_num-1)
            replace_poi_set.append(new_poi)
        return replace_poi_set

    def positive_sampling(self, region_id):
        pos_poi_sets = []
        poi_set, _ = self.ssl_data.get_region(region_id)

        de_poi_set = []
        for ratio in [0.1]:
            de_poi_set.append(self.delete_aug(poi_set, ratio))

        add_poi_set = []
        for ratio in [0.1]:
            add_poi_set.append(self.add_aug(poi_set, ratio))

        re_poi_set = []
        for ratio in [0.1]:
            re_poi_set.append(self.replace_aug(poi_set, ratio))
        
        pos_poi_sets = de_poi_set + add_poi_set + re_poi_set

        return pos_poi_sets

    def negative_sampling(self, region_id):
        sampling_pool = []
        for _id in self.ssl_data.sampling_pool:
            if _id == region_id:
                continue
            sampling_pool.append(_id)

        p = self.ssl_data.rs_ratio["model_poi"][region_id]
        neg_region_ids = np.random.choice(sampling_pool, self.neg_size, replace=False, p=p)

        neg_poi_sets = []
        for neg_region_id in neg_region_ids:
            poi_set, _ = self.ssl_data.get_region(neg_region_id)
            neg_poi_sets.append(poi_set)

        return neg_poi_sets

    def forward(self, poi_set, pos_poi_sets, neg_poi_sets):
        
        # W_emb_temp = self.tree_gcn()
        W_emb_temp = self.W_poi

        base_region_emb = self.agg_region_emb(poi_set, W_emb_temp)

        pos_region_emb_list = []
        for pos_poi_set in pos_poi_sets:
            pos_region_emb = self.agg_region_emb(pos_poi_set, W_emb_temp)
            pos_region_emb_list.append(pos_region_emb.unsqueeze(0))
        pos_region_emb = torch.cat(pos_region_emb_list, dim=0)

        neg_region_emb_list = []
        for neg_poi_set in neg_poi_sets:
            neg_region_emb = self.agg_region_emb(neg_poi_set, W_emb_temp)
            neg_region_emb_list.append(neg_region_emb.unsqueeze(0))
        neg_region_emb = torch.cat(neg_region_emb_list, dim=0)
        
        pos_scores = torch.matmul(pos_region_emb, base_region_emb)
        pos_label = torch.Tensor([1 for _ in range(pos_scores.size(0))]).type(FType).to(device)
        
        neg_scores = torch.matmul(neg_region_emb, base_region_emb)
        neg_label = torch.Tensor([0 for _ in range(neg_scores.size(0))]).type(FType).to(device)

        scores = torch.cat([pos_scores, neg_scores])
        labels = torch.cat([pos_label, neg_label])
        scores /= self.temp

        loss = -(F.log_softmax(scores, dim=0) * labels).sum() / labels.sum()

        return loss, base_region_emb, neg_region_emb

    def model_train(self, region_id):
        poi_set, _ = self.ssl_data.get_region(region_id)
        pos_poi_sets = self.positive_sampling(region_id)
        neg_poi_sets = self.negative_sampling(region_id)

        poi_loss, base_region_emb, neg_region_emb = self.forward(poi_set, pos_poi_sets, neg_poi_sets)

        return poi_loss, base_region_emb, neg_region_emb 

    def get_emb(self):
        output = []
        for region_id in self.ssl_data.sampling_pool:
            poi_set, _ = self.ssl_data.get_region(region_id)
            region_emb = self.agg_region_emb(poi_set, self.W_poi)

            output.append(region_emb.detach().cpu().numpy())
        return np.array(output)


class FLOW_SSL(torch.nn.Module):

    def __init__(self, ssl_data, neg_size, emb_size, temp, time_zone, extractor): 
        super(FLOW_SSL,self).__init__()

        self.ssl_data = ssl_data
        self.init_basic_conf(neg_size, emb_size, temp, time_zone, extractor)
        
    def init_basic_conf(self, neg_size, emb_size, temp, time_zone, extractor):
        self.neg_size = neg_size
        self.emb_size = emb_size
        self.temp = temp
        self.time_zone = time_zone

        self.extractor = extractor

        if self.extractor == "CNN":
            self.pickup_net = torch.nn.Sequential(
                nn.Conv1d(in_channels=self.time_zone, out_channels=4, kernel_size=7),
                Flatten(),
                nn.Linear(1056, self.emb_size)
            ).to(device)

            self.dropoff_net = torch.nn.Sequential(
                nn.Conv1d(in_channels=self.time_zone, out_channels=4, kernel_size=7),
                Flatten(),
                nn.Linear(1056, self.emb_size)
            ).to(device)

        if self.extractor == "MLP":
            self.pickup_net = torch.nn.Sequential(
                nn.Linear(270, self.emb_size),
                nn.ReLU(),
            ).to(device)

            self.dropoff_net = torch.nn.Sequential(
                nn.Linear(270, self.emb_size),
                nn.ReLU(),
            ).to(device)

        if self.extractor == "SA":
            self.pickup_net = SAEncoder(d_input=270, d_model=16, n_head=3).to(device)
            self.dropoff_net = SAEncoder(d_input=270, d_model=16, n_head=3).to(device)


    def gaussian_noise(self, matrix, mean=0, sigma=0.03):
        matrix = matrix.copy()
        noise = np.random.normal(mean, sigma, matrix.shape)
        mask_overflow_upper = matrix+noise >= 1.0
        mask_overflow_lower = matrix+noise < 0
        noise[mask_overflow_upper] = 1.0
        noise[mask_overflow_lower] = 0
        matrix += noise
        return matrix

    def positive_sampling(self, region_id):
        pos_flow_sets = []
        _, flow_matrix = self.ssl_data.get_region(region_id)
        pickup_matrix, dropoff_matrix = flow_matrix

        for sigma in [0.0001, 0.0001, 0.0001, 0.0001]:
            pickup_matrix = self.gaussian_noise(pickup_matrix, sigma=sigma)
            dropoff_matrix = self.gaussian_noise(dropoff_matrix, sigma=sigma)
            pos_flow_sets.append([pickup_matrix, dropoff_matrix])

        return pos_flow_sets

    def negative_sampling(self, region_id):
        sampling_pool = []
        for _id in self.ssl_data.sampling_pool:
            if _id == region_id:
                continue
            sampling_pool.append(_id)

        p = self.ssl_data.rs_ratio["model_flow"][region_id]
        neg_region_ids = np.random.choice(sampling_pool, self.neg_size, replace=False, p=p)

        neg_flow_sets = []
        for neg_region_id in neg_region_ids:
            _, flow_matrix = self.ssl_data.get_region(neg_region_id)
            neg_flow_sets.append(flow_matrix)


        return neg_flow_sets

    def agg_region_emb(self, flow_matrix):
        pickup_matrix = flow_matrix[0]
        dropoff_matrix = flow_matrix[1]

        if self.extractor == "CNN":
            pickup_matrix = torch.from_numpy(pickup_matrix).type(FType).to(device)
            pickup_matrix = pickup_matrix.unsqueeze(0)
            pickup_emb = self.pickup_net(pickup_matrix)

            dropoff_matrix = torch.from_numpy(dropoff_matrix).type(FType).to(device)
            dropoff_matrix = dropoff_matrix.unsqueeze(0)
            dropoff_emb = self.dropoff_net(dropoff_matrix)

        if self.extractor == "MLP":
            pickup_matrix = np.sum(pickup_matrix, axis=0)
            pickup_matrix = torch.from_numpy(pickup_matrix).type(FType).to(device)
            pickup_emb = self.pickup_net(pickup_matrix)

            dropoff_matrix = np.sum(dropoff_matrix, axis=0)
            dropoff_matrix = torch.from_numpy(dropoff_matrix).type(FType).to(device)
            dropoff_emb = self.dropoff_net(dropoff_matrix)

        if self.extractor == "SA":
            pickup_matrix = np.sum(pickup_matrix, axis=0)
            pickup_matrix = torch.from_numpy(pickup_matrix).type(FType).to(device)
            pickup_emb = self.pickup_net(pickup_matrix)

            dropoff_matrix = np.sum(dropoff_matrix, axis=0)
            dropoff_matrix = torch.from_numpy(dropoff_matrix).type(FType).to(device)
            dropoff_emb = self.dropoff_net(dropoff_matrix)

        # region_emb = torch.cat([pickup_emb, dropoff_emb], dim=1).squeeze()
        region_emb = (pickup_emb + dropoff_emb) / 2
        region_emb = region_emb.squeeze()

        return region_emb

    def forward(self, flow_matrix, pos_flow_sets, neg_flow_sets):
        base_region_emb = self.agg_region_emb(flow_matrix)

        pos_region_emb_list = []
        for pos_flow_matrix in pos_flow_sets:
            pos_region_emb = self.agg_region_emb(pos_flow_matrix)
            pos_region_emb_list.append(pos_region_emb.unsqueeze(0))
        pos_region_emb = torch.cat(pos_region_emb_list, dim=0)

        neg_region_emb_list = []
        for neg_flow_matrix in neg_flow_sets:
            neg_region_emb = self.agg_region_emb(neg_flow_matrix)
            neg_region_emb_list.append(neg_region_emb.unsqueeze(0))
        neg_region_emb = torch.cat(neg_region_emb_list, dim=0)

        pos_scores = torch.matmul(pos_region_emb, base_region_emb)
        pos_label = torch.Tensor([1 for _ in range(pos_scores.size(0))]).type(FType).to(device)
        
        neg_scores = torch.matmul(neg_region_emb, base_region_emb)
        neg_label = torch.Tensor([0 for _ in range(neg_scores.size(0))]).type(FType).to(device)

        scores = torch.cat([pos_scores, neg_scores])
        labels = torch.cat([pos_label, neg_label])
        scores /= self.temp

        loss = -(F.log_softmax(scores, dim=0) * labels).sum() / labels.sum()
        return loss, base_region_emb, neg_region_emb

    def model_train(self, region_id):

        _, flow_matrix = self.ssl_data.get_region(region_id)
        pos_flow_sets = self.positive_sampling(region_id)
        neg_flow_sets = self.negative_sampling(region_id)

        flow_loss, base_region_emb, neg_region_emb = self.forward(flow_matrix, pos_flow_sets, neg_flow_sets)

        return flow_loss, base_region_emb, neg_region_emb

    def get_emb(self):
        output = []
        for region_id in self.ssl_data.sampling_pool:
            _, flow_matrix = self.ssl_data.get_region(region_id)
            region_emb = self.agg_region_emb(flow_matrix)

            output.append(region_emb.detach().cpu().numpy())
        return np.array(output)

class Model_SSL():

    def __init__(self): 
        super(Model_SSL,self).__init__()
        
        config = ConfigParser()
        config.read('conf', encoding='UTF-8')
        GPU_DEVICE = config['DEFAULT'].get("GPU_DEVICE")
        device = torch.device("cuda:"+GPU_DEVICE if torch.cuda.is_available() else "cpu")
        extractor = config['DEFAULT'].get("EXTRACTOR")

        size = int(config['DEFAULT'].get("EMB"))
        mutual_reg = float(config['DEFAULT'].get("MUTUAL"))
        poi_reg = float(config['DEFAULT'].get("REG"))

        self.ssl_data = SSLData()
        self.poi_model = POI_SSL(self.ssl_data, neg_size=10, emb_size=size, attention_size=16, temp=0.08, extractor=extractor).to(device)
        self.flow_model = FLOW_SSL(self.ssl_data, neg_size=150, emb_size=size, temp=0.08, time_zone=48, extractor=extractor).to(device)

        self.epoch = 200
        self.learning_rate = 0.001

        self.mutual_reg = mutual_reg
        self.poi_reg = poi_reg

        self.mutual_neg_size = 5
        self.emb_size = size
        self.init_basic_conf()

        self.opt = Adam(lr=self.learning_rate, params=[{"params":self.poi_model.poi_net.parameters()},\
            {"params":self.flow_model.pickup_net.parameters()}, \
            {"params":self.flow_model.dropoff_net.parameters()}, {"params":self.mutual_net.parameters()}], weight_decay=1e-5)

    def init_basic_conf(self):
        self.mutual_net = torch.nn.Sequential(
                    nn.Linear(self.emb_size*2, 1)).to(device)

    def forward(self, base_poi_emb, base_flow_emb, neg_poi_emb, neg_flow_emb):
        pos_emb = torch.cat([base_poi_emb, base_flow_emb])
        pos_scores = self.mutual_net(pos_emb)
        pos_label = torch.Tensor([1 for _ in range(pos_scores.size(0))]).type(FType).to(device)

        weights = torch.ones(neg_poi_emb.size()[0])
        _indexs = torch.multinomial(weights, self.mutual_neg_size)
        neg_poi_emb = neg_poi_emb[_indexs]
        base_flow_emb = base_flow_emb.repeat(self.mutual_neg_size, 1)
        neg_emb_p = torch.cat([neg_poi_emb, base_flow_emb], dim=1)

        weights = torch.ones(neg_flow_emb.size()[0])
        _indexs = torch.multinomial(weights, self.mutual_neg_size)
        neg_flow_emb = neg_flow_emb[_indexs]
        base_poi_emb = base_poi_emb.repeat(self.mutual_neg_size, 1)
        neg_emb_f = torch.cat([base_poi_emb, neg_flow_emb], dim=1)

        neg_emb = torch.cat([neg_emb_p, neg_emb_f], dim=0)
        neg_scores = self.mutual_net(neg_emb).squeeze()
        neg_label = torch.Tensor([0 for _ in range(neg_scores.size(0))]).type(FType).to(device)

        scores = torch.cat([pos_scores, neg_scores])
        labels = torch.cat([pos_label, neg_label])

        loss = -(F.log_softmax(scores, dim=0) * labels).sum() / labels.sum()

        return loss

    def model_train(self):
        for epoch in range(self.epoch):
            self.loss =  0.0

            for region_id in self.ssl_data.sampling_pool: 
                poi_loss, base_poi_emb, neg_poi_emb = self.poi_model.model_train(region_id)
                flow_loss, base_flow_emb, neg_flow_emb =  self.flow_model.model_train(region_id)
                mutual_loss = self.forward(base_poi_emb, base_flow_emb, neg_poi_emb, neg_flow_emb)

                loss = flow_loss + self.poi_reg * poi_loss + self.mutual_reg * mutual_loss

                self.opt.zero_grad()
                self.loss += loss
                loss.backward()
                self.opt.step()

            print("=============================> iter epoch", epoch)
            print("avg loss = " + str(self.loss))

            if epoch >= 150:
                self.test()
                fw.write("=============================> iter epoch " + str(epoch) + "\n")
                fw.write("avg loss = " + str(self.loss) + "\n")

    def get_emb(self):
        output_flow = self.flow_model.get_emb()
        output_poi = self.poi_model.get_emb()
        output = np.concatenate((output_flow, output_poi), axis=1)
        return output

    def test(self):
        output = self.get_emb()
        lu_classify(output, fw, _type="con")
        predict_popus(output, fw)
        fw.flush()