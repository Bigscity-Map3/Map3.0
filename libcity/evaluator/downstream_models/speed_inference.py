import torch
import torch.nn as nn
from logging import getLogger
from sklearn.metrics import  mean_squared_error, mean_absolute_error
import numpy as np

from libcity.evaluator.downstream_models.abstract_model import AbstractModel


class Regressor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.fc(x).squeeze(1)


class SpeedInferenceModel(AbstractModel):
    def __init__(self, config):
        self._logger = getLogger()
        self.alpha = config.get('alpha', 1)
        self.n_split = config.get('n_split', 5)
        self.exp_id = config.get('exp_id', None)
        self.result_path = './libcity/cache/{}/evaluate_cache/regression_{}_{}.npy'. \
            format(self.exp_id, self.alpha, self.n_split)

    def run(self, x, label):
        self._logger.info("--- Speed Inference ---")
        split = x.shape[0] // self.n_split
        y_preds = []
        y_trues = []
        x_ = []
        y = torch.tensor(label['speed']['speed'].tolist()).cuda()
        index = label['speed']['index']
        num_road = len(x)
        for i in index:
            x_.append(x[i-num_road])
        x = torch.tensor(np.array(x_)).cuda()
        x.detach()

        for _ in range(self.n_split):
            x = torch.cat((x[split:], x[:split]), 0)
            y = torch.cat((y[split:], y[:split]), 0)
            x_train, x_eval = x[split:], x[:split]
            y_train, y_eval = y[split:], y[:split]
            model = Regressor(x.shape[1]).cuda()
            opt = torch.optim.Adam(model.parameters())

            best_mae = 1e9
            for e in range(1, 101):
                model.train()
                opt.zero_grad()
                loss = nn.MSELoss()(model(x_train), y_train)
                loss.backward()
                opt.step()

                model.eval()
                y_pred = model(x_eval).detach().cpu()
                mse = mean_squared_error(y_eval.cpu(), y_pred)
                if mse < best_mae:
                    best_mae = mse
                    best_pred = y_pred
            y_preds.append(best_pred)
            y_trues.append(y_eval.cpu())
        y_preds = torch.cat(y_preds, dim=0)
        y_trues = torch.cat(y_trues, dim=0)
        y_preds[y_preds < 0] = 0
        mae = mean_absolute_error(y_trues, y_preds)
        rmse = mean_squared_error(y_trues, y_preds) ** 0.5
        mse = mean_squared_error(y_trues, y_preds)

        self.save_result(y_preds, self.result_path)

        result = {'mae': mae, 'mse': mse, 'rmse': rmse}
        return result

    def clear(self):
        pass

    def save_result(self, predict, save_path, filename=None):
        np.save(save_path, predict)
        self._logger.info('Speed Inference Model is saved at {}'.format(save_path))

