import torch
import torch.nn as nn
from logging import getLogger
from sklearn.metrics import  mean_squared_error, mean_absolute_error
import numpy as np
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from libcity.evaluator.downstream_models.abstract_model import AbstractModel
from torch.utils.data import Dataset, DataLoader

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
    def __init__(self, input_dim, hidden_dim, num_layers, activation, embedding, is_static,device):
        super(MLPReg, self).__init__()

        self.num_layers = num_layers
        self.activation = activation

        if not is_static:
            self.lstm=embedding.encode_sequence
        else:    
            self.lstm = TrajEncoder(input_dim, hidden_dim, 1, embedding, device)

        self.layers = []
        for _ in range(self.num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, 1))
        self.layers = nn.ModuleList(self.layers)

    def forward(self, path, valid_len,**kwargs):
        x=self.lstm(path,valid_len,**kwargs)
        for i in range(self.num_layers - 1):
            x = self.activation(self.layers[i](x))
        return self.layers[-1](x).squeeze(1)


class TimeEstimationDataset(Dataset):
    def __init__(self, data_x,data_lens, data_y):
        self.x = data_x
        self.lens = data_lens
        self.y = data_y
        assert len(self.x) == len(self.y)

    def __getitem__(self, item):
        return self.x[item],self.lens[item], self.y[item]

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

    def run(self, embedding_vector, label, embed_model=None,**kwargs):
        x=embedding_vector
        self._logger.info("--- Time Estimation ---")
        min_len, max_len = self.config.get("tte_min_len", 1), self.config.get("tte_max_len", 100)
        dfs = label['time']
        num_samples = int(len(dfs) * 0.001)
        num_regions=1056
        padding_id = 0
        x_arr = np.zeros([num_samples, max_len],dtype=np.int64)
        lens_arr = np.zeros([num_samples], dtype=np.int64)
        y_arr = np.zeros([num_samples], dtype=np.float32)
        for i in range(num_samples):
            row = dfs.iloc[i]
            path = [int(x)-num_regions for x in row['path']]
            lens = len(path)

            if len(row['path']) < max_len:
                temp = np.array([padding_id] * (max_len - len(row['path'])))
                path = np.append(path, temp)

            path = np.array(path)
            x_arr[i,:] = path
            lens_arr[i] = lens
            y_arr[i] = row['time']
        x_arr = torch.Tensor(x_arr).long()
        lens_arr = torch.Tensor(lens_arr).long()
        y_arr = torch.Tensor(y_arr)

        device=self.config.get('device','cpu')
        input_dim=self.config.get('embed_size',128)
        hidden_dim=self.config.get('d_model', 128)
        max_epoch = self.config.get('task_epochs',100) 
        is_static = self.config.get('is_static',True)
        train_size = int(num_samples * 0.8)
        test_size = num_samples - train_size
        train_data_X ,train_lens,train_data_y = x_arr[:train_size],lens_arr[:train_size],y_arr[:train_size]
        test_data_X ,test_lens, test_data_y = x_arr[train_size:],lens_arr[train_size:],y_arr[train_size:]
        train_dataset = TimeEstimationDataset(train_data_X,train_lens,train_data_y)
        test_dataset = TimeEstimationDataset(test_data_X,test_lens,test_data_y)
        train_dataloader= DataLoader(train_dataset,batch_size=64,shuffle=True)
        test_dataloader= DataLoader(test_dataset,batch_size=64,shuffle=True)
        if is_static:
            model = MLPReg(input_dim, hidden_dim, 3, nn.ReLU(),embedding_vector,is_static,device).to(device)
        else:
            model = MLPReg(input_dim, hidden_dim, 3, nn.ReLU(), embed_model,is_static,device).to(device)
        opt = torch.optim.Adam(model.parameters())
        loss_fn=nn.MSELoss()
        patience = 5

        best = {"best epoch": 0, "mae": 1e9, "rmse": 1e9}
        for epoch in range(0, max_epoch):
            model.train()
            for batch_x,batch_lens,batch_y in train_dataloader:
                opt.zero_grad()
                batch_x = batch_x.cuda()
                batch_lens = batch_lens.cuda()
                batch_y = batch_y.cuda()
                preds=model(batch_x,batch_lens,**kwargs)
                loss = loss_fn(preds, batch_y)
                loss.backward()
                opt.step()

            model.eval()
            y_preds = []
            y_trues = []
            for batch_x, batch_lens,batch_y in test_dataloader:
                batch_x = batch_x.cuda()
                batch_lens = batch_lens.cuda()
                batch_y = batch_y.cuda()
                y_preds.append(model(batch_x,batch_lens,**kwargs).detach().cpu())
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