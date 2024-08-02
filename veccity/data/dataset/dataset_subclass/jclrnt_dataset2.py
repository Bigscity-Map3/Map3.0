from veccity.data.dataset.lineseq_dataset import LineSeqDataset
import torch
from veccity.data.dataset.dataset_subclass.bert_base_dataset import padding_mask
import numpy as np
import os
import json
import pandas as pd
from tqdm import tqdm


class JCLRNTDataset(LineSeqDataset):
    def __init__(self,config):
        super().__init__(config)
        
        self.geo_to_ind=self.vocab.loc2index
        self.ind_to_geo=self.vocab.index2loc
        self.construct_road_edge()
        self.construct_od_graph()
        self.collate_fn=collate_nomask_seq

    def construct_road_edge(self):
        self.road_adj = np.zeros(shape=[self.num_nodes,self.num_nodes])
        #构建路网的邻接关系
        self.struct_edge_index=[]
        with open(self.adj_json_path,'r',encoding='utf-8') as fp:
            road_adj_data = json.load(fp)
        for road in range(self.vocab.specials_num,self.num_nodes):
            geo_id=self.ind_to_geo[road]
            for neighbor in road_adj_data[str(geo_id)]:
                if neighbor not in self.geo_to_ind:
                    continue
                n_id=self.geo_to_ind[neighbor]
                self.struct_edge_index.append((road,n_id))
                self.struct_edge_index.append((n_id,road))
        self.struct_edge_index=torch.Tensor(list(set(self.struct_edge_index))).long().transpose(1,0).to(self.device)
        
    def construct_od_graph(self):
        # jclrnt build contrastive graph
        self.edge_index=[]
        od_path=os.path.join(self.data_path,f"{self.dataset}.od")
        self.od_matrix=np.zeros((self.num_nodes,self.num_nodes),dtype=np.float32)
        
        if not os.path.exists(od_path):
            traj_df = pd.read_csv(self.traj_path)
            traj_list = []
            for i in tqdm(range(len(traj_df))):
                path = traj_df.loc[i, 'path']
                path = path[1:len(path) - 1].split(',')
                path = [int(s) for s in path]
                origin_road=path[0]
                destination_road=path[-1]
                if origin_road in self.geo_to_ind and destination_road in self.geo_to_ind:
                    o_id=self.geo_to_ind[origin_road]
                    d_id=self.geo_to_ind[destination_road]
                    self.od_matrix[o_id][d_id]+=1
                    self.edge_index.append((o_id,d_id))
        else:
            od_data=pd.read_csv(od_path)
        
            for i in range(od_data.shape[0]):
                origin_road=od_data['origin_id'][i]
                destination_road=od_data['destination_id'][i]
                if origin_road in self.geo_to_ind and destination_road in self.geo_to_ind:
                    o_id=self.geo_to_ind[origin_road]
                    d_id=self.geo_to_ind[destination_road]
                    self.od_matrix[o_id][d_id]=od_data['flow'][i]
                    self.edge_index.append((o_id,d_id))

        self.edge_index = torch.Tensor(list(self.edge_index)).transpose(1,0).to(self.device)

        self.tran_matrix = self.od_matrix / (self.od_matrix.max(axis=1, keepdims=True, initial=0.) + 1e-9)
        row, col = np.diag_indices_from(self.tran_matrix)
        self.tran_matrix[row, col] = 0
        self.tran_matrix_b = (self.tran_matrix > self.edge_threshold)
        self.edge_index_aug = [(i // self.num_nodes, i % self.num_nodes) for i, n in
                               enumerate(self.tran_matrix_b.flatten()) if n]
        self.edge_index_aug = np.array(self.edge_index_aug, dtype=np.int32).transpose()
        self.edge_index_aug = torch.Tensor(self.edge_index_aug).int().to(self.device)
        self_loop = torch.tensor([[self.num_nodes - 1], [self.num_nodes - 1]]).to(self.device)
        self.edge_index = torch.cat((self.edge_index, self_loop), axis=1)
        self.edge_index_aug = torch.cat((self.edge_index_aug, self_loop), axis=1)

    
    def get_data_feature(self):
        """
        返回一个 dict，包含数据集的相关特征

        Returns:
            dict: 包含数据集的相关特征的字典
        """
        return {'num_nodes':self.num_nodes,"struct_edge_index":self.struct_edge_index,"trans_edge_index":self.edge_index_aug,"vocab":self.vocab}
    
# collate for general
def collate_nomask_seq(data, max_len=None, vocab=None, add_cls=True):
    batch_size = len(data)
    features, temporal_mat = zip(*data)  # list of (seq_length, feat_dim)

    # Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)
    lengths = [X.shape[0] for X in features]  # original sequence length for each time series
    if max_len is None:
        max_len = max(lengths)
    X = torch.zeros(batch_size, max_len, features[0].shape[-1], dtype=torch.long)  # (batch_size, padded_length, feat_dim)
    for i in range(batch_size):
        end = min(lengths[i], max_len)
        X[i, :end, :] = features[i][:end, :]

    padding_masks = padding_mask(torch.tensor(lengths, dtype=torch.int16), max_len=max_len)

    return X.long(),  padding_masks