import torch
import torch.nn as nn
from logging import getLogger
from sklearn.metrics import  mean_squared_error, mean_absolute_error
import numpy as np
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import os
import pandas as pd
from tqdm import trange
import pickle

from libcity.evaluator.downstream_models.abstract_model import AbstractModel
from torch.utils.data import Dataset, DataLoader
def _cal_mat(tim_list,seq_len):
    # calculate the temporal relation matrix
    mat = np.zeros((seq_len, seq_len))
    for i in range(seq_len):
        for j in range(seq_len):
            off = abs(tim_list[i] - tim_list[j])
            mat[i][j] = off
    return mat  # (seq_len, seq_len)

class TrajEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, embedding, device):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.embedding = embedding
        self.n_layers = n_layers
        self.device = device
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, dropout=0.1 if n_layers > 1 else 0.0, batch_first=True)

    def forward(self, path, valid_len):
        original_shape = path.shape  # [batch_size, traj_len]
        
        full_embed = [torch.from_numpy(self.embedding[int(i)]).to(torch.float32) for i in path.view(-1)]
        full_embed = torch.stack(full_embed)
        full_embed = full_embed.view(*original_shape, self.input_dim)  # [batch_size, traj_len, embed_size]
        pack_x = pack_padded_sequence(full_embed, lengths=valid_len, batch_first=True, enforce_sorted=False).to(self.device)
        h0 = torch.zeros(self.n_layers, full_embed.size(0), self.hidden_dim).to(self.device)
        c0 = torch.zeros(self.n_layers, full_embed.size(0), self.hidden_dim).to(self.device)
        out, _ = self.lstm(pack_x, (h0, c0))
        out, _ = pad_packed_sequence(out, batch_first=True)
        out = torch.stack([out[i, int(ind - 1), :] for i, ind in enumerate(valid_len)])  # [batch_size, hidden_dim]
        return out


class MLPReg(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, activation, embedding, is_static,device, max_len):
        super(MLPReg, self).__init__()

        self.num_layers = num_layers
        self.activation = activation

        if not is_static:
            self.lstm=embedding.encode_sequence
        else:    
            self.lstm = TrajEncoder(input_dim, hidden_dim, 1, embedding, device)

        self.mlp = nn.Linear(max_len * input_dim, hidden_dim).to(device)
        self.embedding = embedding
        self.device = device
        self.max_len = max_len
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.is_static = is_static

        self.layers = []
        for _ in range(self.num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, 1))
        self.layers = nn.ModuleList(self.layers)

    def forward(self, path, valid_len,**kwargs):
        if self.is_static:
            t = torch.from_numpy(self.embedding[path.cpu()])
            x = self.mlp(t.view(-1, self.input_dim * self.max_len).float().to(self.device))
        else:
            x=self.lstm(path,valid_len,**kwargs)
        for i in range(self.num_layers - 1):
            x = self.activation(self.layers[i](x))
        return self.layers[-1](x).squeeze(1)


class TimeEstimationDataset(Dataset):
    def __init__(self, data_x,data_lens, data_y,time_arr):
        self.x = data_x
        self.lens = data_lens
        self.y = data_y
        self.time_arr = time_arr
        assert len(self.x) == len(self.y)

    def __getitem__(self, item):
        return self.x[item],self.lens[item], self.y[item],self.time_arr[item]

    def __len__(self):
        return len(self.x)

class TravelTimeEstimationModel(AbstractModel):
    def __init__(self, config):
        self.config=config
        self._logger = getLogger()
        self.alpha = config.get('alpha', 1)
        self.n_split = config.get('n_split', 5)
        self.exp_id = config.get('exp_id', None)
        self.result_path = './libcity/cache/{}/evaluate_cache/regression_{}_{}.npy'. \
            format(self.exp_id, self.alpha, self.n_split)
        self.dataset=config.get('dataset')
        self.cache_data_path = os.path.join('libcity', 'cache', 'dataset_cache', self.dataset)

    def run(self, embedding_vector, label, embed_model=None,**kwargs):
        self._logger.info("--- Time Estimation ---")
        min_len, max_len = self.config.get("tte_min_len", 1), self.config.get("tte_max_len", 100)
        dfs = label['time']
        num_samples = int(len(dfs) * 0.001)
        padding_id = embedding_vector.shape[0]

        tlist_path=self.cache_data_path + f'/tte_{max_len}.pickle'
        if os.path.exists(tlist_path):
            with open(tlist_path, 'rb') as f:
                time_arr = pickle.load(f)
                x_arr = pickle.load(f)
                lens_arr = pickle.load(f)
                y_arr = pickle.load(f)

        else:
            traj_arr = pd.read_csv(self.cache_data_path + '/traj_road.csv')
            time_arr=[]
            for i in trange(traj_arr.shape[0]):
                tlist=eval(traj_arr.iloc[i]['tlist'])
                tlist=[tlist[0]]+tlist+[tlist[-1]]*(max_len-len(tlist)-1)
                tlist=_cal_mat(tlist,max_len)
                time_arr.append(tlist)
            
            x_arr = np.zeros([num_samples, max_len],dtype=np.int64)
            lens_arr = np.zeros([num_samples], dtype=np.int64)
            y_arr = np.zeros([num_samples], dtype=np.float32)
            for i in range(num_samples):
                row = dfs.iloc[i]
                path = [int(x) for x in row['path']]
                lens = len(path)

                if len(row['path']) < max_len:
                    temp = np.array([padding_id] * (max_len - len(row['path'])))
                    path = np.append(path, temp)

                path = np.array(path)
                x_arr[i,:] = path
                lens_arr[i] = lens
                y_arr[i] = row['time']

                with open(tlist_path, 'wb') as f:
                    pickle.dump(time_arr, f)
                    pickle.dump(x_arr,f)
                    pickle.dump(lens_arr,f)
                    pickle.dump(y_arr,f)

        x_arr = torch.Tensor(x_arr).long()
        lens_arr = torch.Tensor(lens_arr).long()
        y_arr = torch.Tensor(y_arr)
        time_arr = torch.Tensor(time_arr)
            
        embedding_vector = np.concatenate((embedding_vector, np.zeros((1, embedding_vector.shape[1]))), axis=0)
        

        device=self.config.get('device','cpu')
        input_dim=self.config.get('embed_size',128)
        hidden_dim=self.config.get('d_model', 128)
        max_epoch = self.config.get('task_epochs',100) 
        is_static = self.config.get('is_static',True)
        train_size = int(num_samples * 0.8)

        test_size = num_samples - train_size
        train_data_X ,train_lens,train_data_y,train_time = x_arr[:train_size],lens_arr[:train_size],y_arr[:train_size],time_arr[:train_size]
        test_data_X ,test_lens, test_data_y,test_time = x_arr[train_size:],lens_arr[train_size:],y_arr[train_size:],time_arr[train_size:]
        train_dataset = TimeEstimationDataset(train_data_X,train_lens,train_data_y,train_time)
        test_dataset = TimeEstimationDataset(test_data_X,test_lens,test_data_y,test_time)
        train_dataloader= DataLoader(train_dataset,batch_size=64,shuffle=True)
        test_dataloader= DataLoader(test_dataset,batch_size=64,shuffle=True)
        if is_static:
            model = MLPReg(input_dim, hidden_dim, 3, nn.ReLU(),embedding_vector,is_static,device,max_len).to(device)
        else:
            model = MLPReg(input_dim, hidden_dim, 3, nn.ReLU(), embed_model,is_static,device,max_len).to(device)
        opt = torch.optim.Adam(model.parameters(),lr=1e-2)
        loss_fn=nn.MSELoss()
        patience = 5

        best = {"best epoch": 0, "mae": 1e9, "rmse": 1e9}

        for epoch in range(max_epoch):
            model.train()
            for batch_x,batch_lens,batch_y,batch_time in train_dataloader:
                opt.zero_grad()

                batch_x = batch_x.to(device)
                batch_lens = batch_lens.to(device)
                batch_y = batch_y.to(device)
                batch_time = batch_time.to(device)
                preds=model(batch_x,batch_lens,tlist=batch_time,**kwargs)
                loss = loss_fn(preds, batch_y)
                loss.backward()
                opt.step()

            model.eval()
            y_preds = []
            y_trues = []
            for batch_x, batch_lens,batch_y,batch_time in test_dataloader:
                batch_x = batch_x.to(device)
                batch_lens = batch_lens.to(device)
                batch_y = batch_y.to(device)
                y_preds.append(model(batch_x,batch_lens,tlist=batch_time,**kwargs).detach().cpu())
                y_trues.append(batch_y.detach().cpu())

            y_preds = torch.cat(y_preds, dim=0)
            y_trues = torch.cat(y_trues, dim=0)

            mae = mean_absolute_error(y_trues, y_preds)
            rmse = mean_squared_error(y_trues, y_preds) ** 0.5
            self._logger.info(f'Epoch: {epoch}, MAE: {mae.item():.4f}, RMSE: {rmse.item():.4f}')
            if mae < best["mae"]:
                best = {"best epoch": epoch, "mae": mae, "rmse": rmse}
                patience = 5
            else:
                if epoch > 10:
                    patience -= 1
                if not patience:
                    self._logger.info("Best epoch: {}, MAE:{}, RMSE:{}".format(best['best epoch'], best['mae'], best["rmse"]))
                    break
        return best

    def clear(self):
        pass

    def save_result(self, save_path, filename=None):
        pass