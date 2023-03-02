
from logging import getLogger
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

from libcity.evaluator.downstream_models.abstract_model import AbstractModel


class RegressionModel(AbstractModel):

    def __init__(self, config):
        self._logger = getLogger()
        self.alpha = config.get('alpha',1)
        self.n_split = config.get('n_split',5)
        self.exp_id = config.get('exp_id', None)
        self.result_path = './libcity/cache/{}/evaluate_cache/regression_{}_{}.npy'.\
            format(self.exp_id,self.alpha,self.n_split)

    def run(self,x,label):
        kf = KFold(n_splits=self.n_split)
        y_preds = []
        y_truths = []
        for train_index, test_index in kf.split(x):
            X_train, X_test = x[train_index], x[test_index]
            y_train, y_test = label[train_index], label[test_index]
            reg = linear_model.Ridge(alpha=self.alpha)
            X_train = np.array(X_train, dtype=float)
            y_train = np.array(y_train, dtype=float)
            reg.fit(X_train, y_train)
            y_pred = reg.predict(X_test)
            y_preds.append(y_pred)
            y_truths.append(y_test)

        self.save_result(y_preds,self.result_path)
        y_pred[y_pred<0] = 0
        mae = mean_absolute_error(y_truths, y_preds)
        mse = mean_squared_error(y_truths, y_preds)
        r2 = r2_score(y_truths, y_preds)

        result={'mae':mae,'mse':mse,'r2':r2}    
        return result
    
    def clear(self):
        pass
    
    def save_result(self, predict,save_path, filename=None):
        np.save(save_path,predict)
        self._logger.info('Kmeans category is saved at {}'.format(save_path))