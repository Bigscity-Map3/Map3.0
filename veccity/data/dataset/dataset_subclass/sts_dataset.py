import datetime
import torch
from veccity.data.dataset.dataset_subclass.bert_base_dataset import padding_mask
from veccity.data.dataset.lineseq_dataset import LineSeqDataset
from veccity.data.preprocess import preprocess_all, cache_dir
from torch.utils.data import Dataset,DataLoader
import numpy as np
from tqdm import tqdm


class STSDataset(LineSeqDataset):
    def __init__(self, config):
        super().__init__(config)
        self.collate_fn = collate_unsuperv_down
        self.ori_path= cache_dir+'/{}/ori_trajs.npz'.format(self.dataset)
        self.qry_path= cache_dir+'/{}/query_trajs.npz'.format(self.dataset)
        self.process_trajs()


    def process_trajs(self):
        ori_data=np.load(self.ori_path,allow_pickle=True)
        qry_data=np.load(self.qry_path,allow_pickle=True)
        ori_traj=ori_data['trajs'].tolist()[:2000]
        ori_tlist=ori_data['tlist'].tolist()[:2000]
        ori_len=ori_data['lengths'].tolist()[:2000]

        qry_traj=qry_data['trajs'].tolist()[:2000]
        qry_tlist=qry_data['tlist'].tolist()[:2000]
        qry_len=qry_data['lengths'].tolist()[:2000]

        ori_dataset=List_Dataset(ori_traj,ori_tlist,ori_len,self.seq_len,self.vocab,self.add_cls)
        qry_dataset=List_Dataset(qry_traj,qry_tlist,qry_len,self.seq_len,self.vocab,self.add_cls)
        self.ori_dataloader=DataLoader(ori_dataset,self.batch_size,collate_fn=collate_superv_sts)
        self.qry_dataloader=DataLoader(qry_dataset,self.batch_size,collate_fn=collate_superv_sts)


class List_Dataset(Dataset):
    def __init__(self,traj,tlist,lens,seq_len,vocab,add_cls):
        self.traj=traj
        self.tlist=tlist
        self.lens=lens
        self.seq_len=seq_len
        self.add_cls=add_cls
        self.vocab=vocab
        self.temporal_mat_list,self.traj_list = self.datapropocess()
    
    def datapropocess(self):
        temporal_mat_list=[]
        traj_list=[]
        for i in tqdm(range(len(self.traj))):
            loc_list = self.traj[i]
            tim_list = self.tlist[i]
            if type(tim_list) == type("str"):
                tim_list=eval(tim_list)

            new_loc_list = [self.vocab.loc2index.get(loc, self.vocab.unk_index) for loc in loc_list]
            new_tim_list = [datetime.datetime.fromtimestamp(tim) for tim in tim_list]
            minutes = [new_tim.hour * 60 + new_tim.minute + 1 for new_tim in new_tim_list]
            weeks = [new_tim.weekday() + 1 for new_tim in new_tim_list]
            usr_list = [self.vocab.unk_index] * len(new_loc_list)
            if self.add_cls:
                new_loc_list = [self.vocab.sos_index] + new_loc_list
                minutes = [self.vocab.pad_index] + minutes
                weeks = [self.vocab.pad_index] + weeks
                usr_list = [usr_list[0]] + usr_list
                tim_list = [tim_list[0]] + tim_list
            temporal_mat = self._cal_mat(tim_list)
            temporal_mat_list.append(temporal_mat)
            traj_feat = np.array([new_loc_list, tim_list, minutes, weeks, usr_list]).transpose((1, 0))
            traj_list.append(traj_feat)
        return temporal_mat_list,traj_list
        

    def _cal_mat(self, tim_list):
        # calculate the temporal relation matrix
        seq_len = len(tim_list)
        mat = np.zeros((seq_len, seq_len))
        for i in range(seq_len):
            for j in range(seq_len):
                off = abs(tim_list[i] - tim_list[j])
                mat[i][j] = off
        return mat  # (seq_len, seq_len)


    def __getitem__(self, index):
        return torch.LongTensor(self.traj_list[index]),self.lens[index],torch.LongTensor(self.temporal_mat_list[index])

    def __len__(self):
        return len(self.traj)



def collate_superv_sts(data, max_len=None, vocab=None, add_cls=True):
    batch_size = len(data)
    features, lengths, temporal_mat = zip(*data)  # list of (seq_length, feat_dim)

    # Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)
    if max_len is None:
        max_len = max(lengths)
        
    X = torch.zeros(batch_size, max_len, features[0].shape[-1], dtype=torch.long)  # (batch_size, padded_length, feat_dim)
    batch_temporal_mat = torch.zeros(batch_size, max_len, max_len,
                                     dtype=torch.long)  # (batch_size, padded_length, padded_length)

    for i in range(batch_size):
        end = min(lengths[i], max_len)
        X[i, :end, :] = features[i][:end]
        batch_temporal_mat[i, :end, :end] = temporal_mat[i][:end, :end]


    padding_masks = padding_mask(torch.tensor(lengths, dtype=torch.int16), max_len=max_len)

    batch={}
    batch['seq']=X.long()
    batch['length'] = lengths
    batch['padding_masks']=padding_masks
    batch['batch_temporal_mat'] = batch_temporal_mat.long()
    return batch

def collate_unsuperv_down(data, max_len=None, vocab=None, add_cls=True):
    batch_size = len(data)
    features, temporal_mat = zip(*data)  # list of (seq_length, feat_dim)

    # Stack and pad features and masks (convert 2D to 3D tensors, i.e. add batch dimension)
    lengths = [X.shape[0] for X in features]  # original sequence length for each time series
    if max_len is None:
        max_len = max(lengths)
    X = torch.zeros(batch_size, max_len, features[0].shape[-1], dtype=torch.long)
    batch_temporal_mat = torch.zeros(batch_size, max_len, max_len,
                                     dtype=torch.long)

    for i in range(batch_size):
        end = min(lengths[i], max_len)
        X[i, :end, :] = features[i][:end, :]
        batch_temporal_mat[i, :end, :end] = temporal_mat[i][:end, :end]

    padding_masks = padding_mask(torch.tensor(lengths, dtype=torch.int16), max_len=max_len)
    batch={}
    batch['seq']=X.long()
    batch['length'] = lengths
    batch['padding_masks']=padding_masks
    batch['batch_temporal_mat'] = batch_temporal_mat.long()
    return batch