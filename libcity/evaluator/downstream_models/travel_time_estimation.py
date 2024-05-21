import torch
import torch.nn as nn
from logging import getLogger
from sklearn.metrics import  mean_squared_error, mean_absolute_error
import numpy as np

from libcity.evaluator.downstream_models.abstract_model import AbstractModel
from torch.utils.data import Dataset, DataLoader



class MLPReg(nn.Module):
    def __init__(self, input_size, num_layers, activation):
        super(MLPReg, self).__init__()

        self.num_layers = num_layers
        self.activation = activation

        self.layers = []
        for _ in range(self.num_layers - 1):
            self.layers.append(nn.Linear(input_size, input_size))
        self.layers.append(nn.Linear(input_size, 1))
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        for i in range(self.num_layers - 1):
            x = self.activation(self.layers[i](x))
        return self.layers[-1](x).squeeze(1)


class TimeEstimationDataset(Dataset):
    def __init__(self, data_x, data_y):
        self.x = data_x
        self.y = data_y
        assert len(self.x) == len(self.y)

    def __getitem__(self, item):
        return self.x[item], self.y[item]

    def __len__(self):
        return len(self.x)

class TravelTimeEstimationModel(AbstractModel):
    def __init__(self, config):
        self.config = config
        self._logger = getLogger()
        self.alpha = config.get('alpha', 1)
        self.n_split = config.get('n_split', 5)
        self.exp_id = config.get('exp_id', None)
        self.result_path = './libcity/cache/{}/evaluate_cache/regression_{}_{}.npy'. \
            format(self.exp_id, self.alpha, self.n_split)

    def run(self, x, label):
        self._logger.info("--- Time Estimation ---")
        min_len, max_len = self.config.get("tte_min_len", 1), self.config.get("tte_max_len", 100)
        dfs = label['time']
        num_samples = int(len(dfs) * 0.001)
        num_road = len(x)
        embed_len = len(x[0])
        padding_id = label['padding_id']
        x_arr = np.zeros([num_samples, max_len * embed_len],dtype=np.float32)
        y_arr = np.zeros([num_samples], dtype=np.float32)
        for i in range(num_samples):
            row = dfs.iloc[i]
            path_rep = []
            path = [int(x) for x in row['path']]
            if len(row['path']) < max_len:
                temp = np.array([padding_id] * (max_len - len(row['path'])))
                path = np.append(path, temp)
            for index in path:
                embed = x[index-num_road]
                path_rep.append(embed)

            path_rep = np.array(path_rep)
            path_rep = np.concatenate(path_rep,axis=0)
            x_arr[i,:] = path_rep
            y_arr[i] = row['time']
        x_arr = torch.Tensor(x_arr)
        y_arr = torch.Tensor(y_arr)

        train_size = int(num_samples * 0.8)
        train_data_X ,train_data_y = x_arr[:train_size],y_arr[:train_size]
        test_data_X ,test_data_y = x_arr[train_size:],y_arr[train_size:]
        train_dataset = TimeEstimationDataset(train_data_X,train_data_y)
        test_dataset = TimeEstimationDataset(test_data_X,test_data_y)
        train_dataloader= DataLoader(train_dataset,batch_size=64,shuffle=True)
        test_dataloader= DataLoader(test_dataset,batch_size=64,shuffle=True)
        model = MLPReg(x_arr.shape[1], 3, nn.ReLU()).cuda()
        opt = torch.optim.Adam(model.parameters())

        patience = 5
        best = {"best epoch": 0, "mae": 1e9, "rmse": 1e9}
        for epoch in range(1, 101):
            model.train()
            for batch_x, batch_y in train_dataloader:
                opt.zero_grad()
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()
                loss = nn.MSELoss()(model(batch_x), batch_y)
                loss.backward()
                opt.step()

            model.eval()
            y_preds = []
            y_trues = []
            for batch_x, batch_y in test_dataloader:
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()
                y_preds.append(model(batch_x).detach().cpu())
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