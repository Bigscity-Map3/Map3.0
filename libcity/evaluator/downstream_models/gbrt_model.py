from logging import getLogger
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from libcity.evaluator.downstream_models.abstract_model import AbstractModel
def construct_feat_from_srcemb_dstemb_dist(triplets, src_emb, dst_emb, dist):
    triplets = triplets.astype(int)
    feat_src = src_emb[triplets[:, 0]]
    feat_dst = dst_emb[triplets[:, 1]]
    feat_dist = dist[triplets[:, 0], triplets[:, 1]].reshape(-1, 1)
    X = np.concatenate([feat_src, feat_dst, feat_dist], axis=1)
    y = triplets[:, 2]
    return X, y


def RMSE(y_hat, y):
    '''
    Root Mean Square Error Metric
    '''
    return np.sqrt(np.mean((y_hat - y) ** 2))


def CPC(y_hat, y):
    '''
    Common Part of Commuters Metric
    '''
    common = np.min(np.stack((y_hat, y), axis=1), axis=1)
    return 2 * np.sum(common) / (np.sum(y_hat) + np.sum(y))


def CPL(y_hat, y):
    '''
    Common Part of Links Metric.

    Check the topology.
    '''
    yy_hat = y_hat > 0
    yy = y > 0
    return 2 * np.sum(yy_hat * yy) / (np.sum(yy_hat) + np.sum(yy))


def MAPE(y_hat, y):
    '''
    Mean Absolute Percentage Error Metric
    '''
    abserror = np.abs(y_hat - y)
    return np.mean(abserror / y)


def MAE(y_hat, y):
    '''
    Mean Absolute Error Metric
    '''
    abserror = np.abs(y_hat - y)
    return np.mean(abserror)


def evaluate(y_hat, y):
    '''
    Evaluate the error in different metrics
    '''
    # metric
    rmse = RMSE(y_hat, y)
    mae = MAE(y_hat, y)
    mape = MAPE(y_hat, y)
    cpc = CPC(y_hat, y)
    cpl = CPL(y_hat, y)
    # return
    return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'CPC': cpc, 'CPL': cpl}
#only for GMEL
class GBRT_Predictor(AbstractModel):
    def __init__(self, config):
        self._logger = getLogger()
        self.exp_id = config.get('exp_id', None)
        self.dataset = config.get('dataset','')
        self.model = config.get('model','')
        self.output_dim = config.get('output_dim', 96)
        self.data_path = './raw_data/' + self.dataset + '/'
        self.geo_file = config.get('geo_file', self.dataset)

    def run(self,src_emb,dst_emb,distm,train_data,valid_data,test_data):
        scaled_distm = distm / distm.max() * np.max([src_emb.max(), dst_emb.max()])
        X_train, y_train = construct_feat_from_srcemb_dstemb_dist(train_data, src_emb, dst_emb, scaled_distm)
        X_valid, y_valid = construct_feat_from_srcemb_dstemb_dist(valid_data, src_emb, dst_emb, scaled_distm)
        X_test, y_test = construct_feat_from_srcemb_dstemb_dist(test_data, src_emb, dst_emb, scaled_distm)
        self._logger.info("start train GBRT_Predictor")
        gbrt = GradientBoostingRegressor(max_depth=2, random_state=2019, n_estimators=100)
        gbrt.fit(X_train, y_train)
        self._logger.info('finish training')
        # test
        y_gbrt = gbrt.predict(X_test)
        res = evaluate(y_gbrt, y_test)
        self._logger.info('test result: {}'.format(res))