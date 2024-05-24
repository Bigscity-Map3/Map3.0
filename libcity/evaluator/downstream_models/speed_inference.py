import torch
import torch.nn as nn
from logging import getLogger
from sklearn.metrics import  mean_squared_error, mean_absolute_error
import numpy as np

from libcity.evaluator.downstream_models.abstract_model import AbstractModel
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error



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
        x_ = []
        index = label['speed']['index']-1056
        for i in index:
            x_.append(x[i])
        x = np.array(x_)
        y = np.array(label['speed']['speed'])
        kf = KFold(n_splits=self.n_split)
        y_preds = []
        y_truths = []
        for train_index, test_index in kf.split(x):
            train_index = list(train_index)
            test_index = list(test_index)
            X_train, X_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            reg = linear_model.Ridge(alpha=self.alpha)
            X_train = np.array(X_train, dtype=float)
            y_train = np.array(y_train, dtype=float)
            reg.fit(X_train, y_train)
            y_pred = reg.predict(X_test)
            y_preds.append(y_pred)
            y_truths.append(y_test)

        y_pred[y_pred<0] = 0
        y_truths = np.concatenate(y_truths)
        y_preds = np.concatenate(y_preds)
        mae = mean_absolute_error(y_truths, y_preds)
        mse = mean_squared_error(y_truths, y_preds)
        rmse = mse**0.5
        r2 = r2_score(y_truths, y_preds)

        self.result={'mae':mae, 'mse':mse, 'r2':r2, 'rmse': rmse}
        self._logger.info(self.result)
        self.save_result(self.result_path)
        return self.result

    def clear(self):
        pass

    def save_result(self, save_path, filename=None):
        # np.save(save_path, predict)
        self._logger.info('Speed Inference Model is saved at {}'.format(save_path))

