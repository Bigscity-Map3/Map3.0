
from logging import getLogger
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

from libcity.evaluator.downstream_models.abstract_model import AbstractModel


class RegressionModel(AbstractModel):

    def __init__(self, config):
        self._logger = getLogger()
        self.alpha = config.get('alpha', 1)
        self.n_split = config.get('n_split', 5)
        self.exp_id = config.get('exp_id', None)
        self.result_path = './libcity/cache/{}/evaluate_cache/regression_{}_{}.npy'.\
            format(self.exp_id, self.alpha, self.n_split)
        
        self.regression_type = config.get('regression_type', 'Ridge')

    def run(self, x, label):
        kf = KFold(n_splits=self.n_split)
        y_preds = []
        y_truths = []

        mae_list = []
        mse_list = []
        r2_list = []

        for train_index, test_index in kf.split(x):
            X_train, X_test = x[train_index], x[test_index]
            y_train, y_test = label[train_index], label[test_index]

            if self.regression_type == 'LinearRegression':
                reg = linear_model.LinearRegression(alpha=self.alpha)
            elif self.regression_type == 'Lasso':
                reg = linear_model.Lasso(alpha=self.alpha)
            else:
                reg = linear_model.Ridge(alpha=self.alpha)

            X_train = np.array(X_train, dtype=float)
            y_train = np.array(y_train, dtype=float)

            reg.fit(X_train, y_train)

            y_pred = reg.predict(X_test)

            y_preds.append(y_pred)
            y_truths.append(y_test)

            mae_list.append(mean_absolute_error(y_test, y_pred))
            mse_list.append(mean_squared_error(y_test, y_pred))
            r2_list.append(r2_score(y_test, y_pred))

        y_preds = np.concatenate(y_preds)
        y_truths = np.concatenate(y_truths)

        self.save_result(y_preds, self.result_path)

        mae = mean_absolute_error(y_truths, y_preds)
        mse = mean_squared_error(y_truths, y_preds)
        r2 = r2_score(y_truths, y_preds)

        mean_mae = np.mean(mae_list)
        mean_mse = np.mean(mse_list)
        mean_r2 = np.mean(r2_list)

        result={'mae': mean_mae, 'mse': mean_mse, 'r2': mean_r2}    
        return result
    
    def clear(self):
        pass
    
    def save_result(self, predict,save_path, filename=None):
        np.save(save_path,predict)
        self._logger.info('the regression prediction results are saved in {}'.format(save_path))