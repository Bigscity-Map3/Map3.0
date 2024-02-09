import math
import json
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from logging import getLogger
from sklearn import manifold
from sklearn.cluster import KMeans
from sklearn import metrics
from libcity.evaluator.abstract_evaluator import AbstractEvaluator
from sklearn.metrics import accuracy_score, f1_score
from dis import dis
import json
import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import linear_model
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import argparse
from sklearn.ensemble import RandomForestRegressor
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import manifold
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


class Classifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.fc(x)

class Bilinear_Module(nn.Module):
    def __init__(self, dim):
        super(Bilinear_Module, self).__init__()
        self.regressor = nn.Bilinear(dim,dim,1)

    def forward(self, x):
        #x为batch_size×2×dim
        return self.regressor(x[:, 0, :],x[:, 1, :])

def metrics_local(y_truths, y_preds):
    y_preds[y_preds < 0] = 0
    mae = mean_absolute_error(y_truths, y_preds)
    rmse = mean_squared_error(y_truths, y_preds, squared=False)
    mape = mean_absolute_percentage_error(y_truths, y_preds) * 100
    r2 = r2_score(y_truths, y_preds)
    return mae, rmse, mape, r2


def evaluation_classify(X, y, kfold=5, num_classes=5, seed=42, output_dim=128):
    KF = StratifiedKFold(n_splits=kfold, random_state=seed, shuffle=True)
    # 模型训练
    y_preds = []
    y_trues = []
    for fold_num, (train_idx, val_idx) in enumerate(KF.split(X, y)):
        # self._logger.info('Fold {}'.format(fold_num))
        X_train, X_eval = X[train_idx], X[val_idx]
        y_train, y_eval = y[train_idx], y[val_idx]
        X_train = torch.tensor(X_train).cuda()
        X_eval = torch.tensor(X_eval).cuda()
        y_train = torch.tensor(y_train).cuda()
        y_eval = torch.tensor(y_eval).cuda()

        model = Classifier(output_dim, num_classes=num_classes).cuda()
        opt = torch.optim.Adam(model.parameters())

        best_acc = 0.
        best_pred = 0.
        for e in range(1000):
            model.train()
            opt.zero_grad()
            ce_loss = nn.CrossEntropyLoss()(model(X_train), y_train)
            ce_loss.backward()
            opt.step()

            model.eval()
            y_pred = torch.argmax(model(X_eval), -1).detach().cpu()
            acc = accuracy_score(y_eval.cpu(), y_pred, normalize=True)
            # print('Epoch {}, acc@1 = {}'.format(e, acc))
            if acc > best_acc:
                best_acc = acc
                best_pred = y_pred
        y_preds.append(best_pred)
        y_trues.append(y_eval.cpu())

    y_preds = torch.cat(y_preds, dim=0)
    y_trues = torch.cat(y_trues, dim=0)
    macro_f1 = f1_score(y_trues, y_preds, average='macro')
    micro_f1 = f1_score(y_trues, y_preds, average='micro')
    return micro_f1, macro_f1

def evaluation_bilinear_reg(embedding, flow, kfold=5, seed=42, output_dim=128):
    """
    :param embedding: node_num*output_dim
    :param flow: node_num*node_num
    :param kfold:
    :param seed:
    :param output_dim:
    :return:
    """
    kf = KFold(n_splits=kfold, random_state=seed, shuffle=True)
    X = []
    y = []
    node_num = embedding.shape[0]
    for i in range(node_num):
        for j in range(node_num):
            if flow[i][j] > 0:
                X.append([embedding[i],embedding[j]])
                y.append(flow[i][j])
    y_preds = []
    y_trues = []
    X = np.array(X)
    y = np.array(y)
    for fold_num, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, X_eval = X[train_idx], X[val_idx]
        y_train, y_eval = y[train_idx], y[val_idx]
        X_train = torch.tensor(X_train).cuda()
        X_eval = torch.tensor(X_eval).cuda()
        y_train = torch.tensor(y_train).cuda()
        y_eval = torch.tensor(y_eval).cuda()
        model = Bilinear_Module(output_dim).cuda()
        criterion = nn.MSELoss()
        opt = torch.optim.Adam(model.parameters(),lr=0.01)
        best_mse = 100000
        best_pred = 0
        dataset = TensorDataset(X_train, y_train)

        #batch_size = 1280
        #shuffle = True

        # 创建 DataLoader
        #dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        for e in range(2000):
            #for inputs, targets in dataloader:
            model.train()
            opt.zero_grad()
            mse_loss = criterion(model(X_train).squeeze(),y_train)
            mse_loss.backward()
            opt.step()
            model.eval()
            y_val_pred  = model(X_eval).squeeze()
            val_loss = criterion(y_eval,y_val_pred)
            if val_loss<best_mse:
                best_mse = val_loss
                best_pred = y_val_pred

        y_preds.append(best_pred.detach().cpu())
        y_trues.append(y_eval.cpu())
    y_preds = torch.cat(y_preds, dim=0).cpu()
    y_trues = torch.cat(y_trues, dim=0).cpu()
    y_preds = y_preds.numpy()
    y_trues = y_trues.numpy()
    mae, rmse, mape, r2 = metrics_local(y_trues, y_preds)
    return mae,rmse,mape,r2




def evaluation_reg(X, y, kfold=5, seed=42, output_dim=128):
    kf = KFold(n_splits=kfold, random_state=seed, shuffle=True)
    y_preds = []
    y_truths = []
    for fold_num, (train_idx, test_idx) in enumerate(kf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        reg = linear_model.Ridge(alpha=1.0, random_state=seed)
        # reg = linear_model.LinearRegression()
        # reg = RandomForestRegressor()
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)
        # plt.plot(np.arange(len(y_pred)), y_pred, label='pred')
        # plt.plot(np.arange(len(y_test)), y_test, label='label')
        # plt.legend()
        # plt.title(str(fold_num))
        # plt.show()

        y_preds.append(y_pred)
        y_truths.append(y_test)

    y_preds = np.concatenate(y_preds)
    y_truths = np.concatenate(y_truths)

    mae, rmse, mape, r2 = metrics_local(y_truths, y_preds)
    return mae, rmse, mape, r2

class HHGCLEvaluator(AbstractEvaluator):

    def __init__(self, config, data_feature):
        self._logger = getLogger()
        self.representation_object = config.get('representation_object', 'region')
        self.result = {}
        self.model = config.get('model', '')
        self.dataset = config.get('dataset', '')
        self.exp_id = config.get('exp_id', None)
        self.cluster_kinds = config.get('cluster_kinds', 5)
        self.seed = config.get('seed', 0)
        self.data_path = './raw_data/' + self.dataset + '/'
        self.output_dim = config.get('output_dim', 128)
        self.roadid = config.get('roadid', None)
        self.regionid = config.get('regionid', None)
        self.region_embedding_path = './libcity/cache/{}/evaluate_cache/region_embedding_{}_{}_{}.npy'\
            .format(self.exp_id, self.model, self.dataset, self.output_dim)
        self.road_embedding_path = './libcity/cache/{}/evaluate_cache/road_embedding_{}_{}_{}.npy'\
            .format(self.exp_id, self.model, self.dataset, self.output_dim)

    def collect(self, batch):
        pass

    def evaluation_cluster(self, y_truth ,useful_index,node_emb, item_type):
        kinds = self.cluster_kinds
        self._logger.info('Start Kmeans, data.shape = {}, kinds = {}'.format(
            str(node_emb.shape), kinds))
        k_means = KMeans(n_clusters=self.cluster_kinds, random_state=self.seed)
        k_means.fit(node_emb)
        labels = k_means.labels_
        y_predict = k_means.predict(node_emb)
        y_predict_useful = y_predict[useful_index]
        nmi = normalized_mutual_info_score(y_truth, y_predict_useful)
        ars = adjusted_rand_score(y_truth, y_predict_useful)
        # SC指数
        sc = float(metrics.silhouette_score(node_emb, labels, metric='euclidean'))
        # DB指数
        db = float(metrics.davies_bouldin_score(node_emb, labels))
        # CH指数
        ch = float(metrics.calinski_harabasz_score(node_emb, labels))
        self._logger.info("Evaluate result is sc = {:6f}, db = {:6f}, ch = {:6f}, nmi = {:6f}, ars = {:6f}".format(sc, db, ch, nmi, ars))
        # result_path = './libcity/cache/{}/evaluate_cache/kmeans_evaluate_{}_{}_{}_{}_{}.json'. \
        #     format(self.exp_id, item_type, self.model, self.dataset, str(self.output_dim), str(kinds))
        # json.dump(self.result, open(result_path, 'w'), indent=4)
        # self._logger.info('Evaluate result is saved at {}'.format(result_path))

        # TSNE可视化
        plt.figure()
        x_input_tsne = normalize(node_emb, norm='l2', axis=1)  # 按行归一化
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=self.seed)  # n_components=2降维为2维并且可视化
        x_tsne = tsne.fit_transform(x_input_tsne)
        sns.scatterplot(x=x_tsne[:, 0], y=x_tsne[:, 1], hue=y_predict, palette='Set1', legend=True, linewidth=0,)
        result_path = './libcity/cache/{}/evaluate_cache/{}_kmeans_tsne_{}_{}_{}_{}_{}.png'. \
            format(self.exp_id, self.exp_id, item_type, self.model, self.dataset, self.output_dim, str(kinds))
        plt.title('{} {} {} {}'.format(item_type, self.exp_id, self.model, self.dataset))
        plt.savefig(result_path)
        # self._logger.info('Kmeans result for TSNE is saved at {}'.format(result_path))
        return sc, db, ch, nmi, ars

    def _valid_road_clf(self, road_emb):
        road_data = pd.read_csv(
            './raw_data/{}/roadmap_{}/roadmap_{}.geo'.format(self.dataset, self.dataset, self.dataset))
        road_label = road_data['highway'].values
        label = road_label
        useful_index = []
        useful_label = [2, 3, 4, 5, 6]
        num_classes = 5
        self._logger.info('Road emb shape = {}, label shape = {}'.format(road_emb.shape, label.shape))
        assert len(label) == len(road_emb)

        X = []
        y = []
        for i, label_i in enumerate(label):
            if label_i in useful_label:
                useful_index.append(i)
                X.append(road_emb[i: i + 1, :])
                y.append(label_i - 2)
        X = np.concatenate(X, axis=0)
        y = np.array(y)

        self._logger.info(
            'Selected Road emb shape = {}, label shape = {}, label min = {}, label max = {}, num_classes = {}'.format(
                X.shape, y.shape, y.min(), y.max(), num_classes))
        road_micro_f1, road_macro_f1 = evaluation_classify(X, y, kfold=5, num_classes=num_classes, seed=self.seed,
                                                           output_dim=self.output_dim)
        self._logger.info('micro F1: {:6f}, macro F1: {:6f}'.format(road_micro_f1, road_macro_f1))
        return y,useful_index,road_micro_f1, road_macro_f1

    def _valid_region_clf(self, region_emb):
        region_data = pd.read_csv(
            './raw_data/{}/regionmap_{}/regionmap_{}.geo'.format(self.dataset, self.dataset, self.dataset))
        region_label = region_data['FUNCTION'].values
        label = region_label
        useful_label = [1, 3, 4, 5, 6]
        useful_index = []
        num_classes = 5
        self._logger.info('Region emb shape = {}, label shape = {}'.format(region_emb.shape, label.shape))
        assert len(label) == len(region_emb)

        X = []
        y = []
        for i, label_i in enumerate(label):
            if label_i in useful_label:
                useful_index.append(i)
                X.append(region_emb[i: i + 1, :])
                if label_i == 1:
                    y.append(label_i - 1)
                else:
                    y.append(label_i - 2)
        X = np.concatenate(X, axis=0)
        y = np.array(y)
        self._logger.info(
            'Selected Region emb shape = {}, label shape = {}, label min = {}, label max = {}, num_classes = {}'.format(
                X.shape, y.shape, y.min(), y.max(), num_classes))
        region_micro_f1, region_macro_f1 = evaluation_classify(X, y, kfold=5, num_classes=num_classes, seed=self.seed,
                                                               output_dim=self.output_dim)
        self._logger.info('micro F1: {:6f}, macro F1: {:6f}'.format(region_micro_f1, region_macro_f1))
        return y,useful_index,region_micro_f1, region_macro_f1
    def _valid_road_flow_using_bilinear(self,road_emb):
        self._logger.warning('Evaluating Road OD-Flow Prediction Using Bilinear Module')
        od_flow = np.load('./raw_data/{}/road_od_flow_{}_11.npy'.format(
            self.dataset, self.dataset)).astype('float32')
        road_mae,road_rmse,road_mape,road_r2 = evaluation_bilinear_reg(road_emb,od_flow,kfold=5, seed=self.seed, output_dim=self.output_dim)
        self._logger.info("Result of {} bilinear estimation in {}:".format('odflow', self.dataset))
        self._logger.info(
            'MAE = {:6f}, RMSE = {:6f}, R2 = {:6f}, MAPE = {:6f}'.format(road_mae,road_rmse,road_r2,road_mape))
        return road_mae, road_rmse, road_r2, road_mape
    def _valid_region_flow_using_bilinear(self,region_emb):
        self._logger.warning('Evaluating Region OD-Flow Prediction Using Bilinear Module')
        od_flow = np.load('./raw_data/{}/region_od_flow_{}_11.npy'.format(
            self.dataset, self.dataset)).astype('float32')
        region_mae,region_rmse,region_mape,region_r2 = evaluation_bilinear_reg(region_emb,od_flow,kfold=5, seed=self.seed, output_dim=self.output_dim)
        self._logger.info("Result of {} bilinear estimation in {}:".format('odflow', self.dataset))
        self._logger.info(
            'MAE = {:6f}, RMSE = {:6f}, R2 = {:6f}, MAPE = {:6f}'.format(region_mae,region_rmse,region_r2,region_mape))
        return region_mae,region_rmse,region_r2,region_mape


    def _valid_road_flow(self, road_emb):
        self._logger.warning('Evaluating Road In-Flow Prediction')
        inflow = np.load('./raw_data/{}/road_in_flow_day_avg_{}_11.npy'.format(
            self.dataset, self.dataset, self.dataset)).astype('float32')
        in_road_mae, in_road_rmse, in_road_mape, in_road_r2 = evaluation_reg(road_emb, inflow / 24, kfold=5, seed=self.seed, output_dim=self.output_dim)
        self._logger.info("Result of {} estimation in {}:".format('inflow', self.dataset))
        self._logger.info('MAE = {:6f}, RMSE = {:6f}, R2 = {:6f}, MAPE = {:6f}'.format(in_road_mae, in_road_rmse, in_road_r2, in_road_mape))

        self._logger.warning('Evaluating Road Out-Flow Prediction')
        outflow = np.load('./raw_data/{}/road_out_flow_day_avg_{}_11.npy'.format(
            self.dataset, self.dataset, self.dataset)).astype('float32')
        out_road_mae, out_road_rmse, out_road_mape, out_road_r2 = evaluation_reg(road_emb, outflow / 24, kfold=5, seed=self.seed, output_dim=self.output_dim)
        self._logger.info("Result of {} estimation in {}:".format('outflow', self.dataset))
        self._logger.info('MAE = {:6f}, RMSE = {:6f}, R2 = {:6f}, MAPE = {:6f}'.format(out_road_mae, out_road_rmse, out_road_r2, out_road_mape))

        road_mae = (in_road_mae + out_road_mae) / 2
        road_rmse = (in_road_rmse + out_road_rmse) / 2
        road_mape = (in_road_mape + out_road_mape) / 2
        road_r2 = (in_road_r2 + out_road_r2) / 2
        self._logger.info("Result of road flow estimation in {}:".format(self.dataset))
        self._logger.info('MAE = {:6f}, RMSE = {:6f}, R2 = {:6f}, MAPE = {:6f}'.format(road_mae, road_rmse, road_r2, road_mape))
        return road_mae, road_rmse, road_r2, road_mape

    def _valid_region_flow(self, region_emb):
        self._logger.warning('Evaluating Region In-Flow Prediction')
        inflow = np.load('./raw_data/{}/region_in_flow_day_avg_{}_11.npy'.format(
            self.dataset, self.dataset, self.dataset)).astype('float32')
        in_region_mae, in_region_rmse, in_region_mape, in_region_r2 = evaluation_reg(region_emb, inflow / 24, kfold=5, seed=self.seed, output_dim=self.output_dim)
        self._logger.info("Result of {} estimation in {}:".format('inflow', self.dataset))
        self._logger.info('MAE = {:6f}, RMSE = {:6f}, R2 = {:6f}, MAPE = {:6f}'.format(in_region_mae, in_region_rmse, in_region_r2, in_region_mape))

        self._logger.warning('Evaluating Region Out-Flow Prediction')
        outflow = np.load('./raw_data/{}/region_out_flow_day_avg_{}_11.npy'.format(
            self.dataset, self.dataset, self.dataset)).astype('float32')
        out_region_mae, out_region_rmse, out_region_mape, out_region_r2 = evaluation_reg(region_emb, outflow / 24, kfold=5, seed=self.seed, output_dim=self.output_dim)
        self._logger.info("Result of {} estimation in {}:".format('outflow', self.dataset))
        self._logger.info('MAE = {:6f}, RMSE = {:6f}, R2 = {:6f}, MAPE = {:6f}'.format(out_region_mae, out_region_rmse, out_region_r2, out_region_mape))

        region_mae = (in_region_mae + out_region_mae) / 2
        region_rmse = (in_region_rmse + out_region_rmse) / 2
        region_mape = (in_region_mape + out_region_mape) / 2
        region_r2 = (in_region_r2 + out_region_r2) / 2
        self._logger.info("Result of region flow estimation in {}:".format(self.dataset))
        self._logger.info('MAE = {:6f}, RMSE = {:6f}, R2 = {:6f}, MAPE = {:6f}'.format(region_mae, region_rmse, region_r2, region_mape))
        return region_mae, region_rmse, region_mape, region_r2
    def evaluate_road_embedding(self):
        road_emb = np.load(self.road_embedding_path)
        self._logger.info('Load road emb {}, shape = {}'.format(self.road_embedding_path, road_emb.shape))
        self._logger.warning('Evaluating Road Classification')
        y_truth,useful_index,road_micro_f1, road_macro_f1 = self._valid_road_clf(road_emb)
        road_mae, road_rmse, road_r2, road_mape = self._valid_road_flow(road_emb)
        road_bilinear_mae,road_bilinear_rmse,road_bilinear_r2,road_bilinear_mape = self._valid_road_flow_using_bilinear(road_emb)
        self._logger.warning('Evaluating Road Emb by TSNE ans Kmeans')
        road_sc, road_db, road_ch, road_nmi, road_ars = self.evaluation_cluster(y_truth,useful_index,road_emb, 'Road')
        self.result['road_micro_f1'] = [road_micro_f1]
        self.result['road_macro_f1'] = [road_macro_f1]
        self.result['road_mae'] = [road_mae]
        self.result['road_rmse'] = [road_rmse]
        self.result['road_mape'] = [road_mape]
        self.result['road_r2'] = [road_r2]
        self.result['road_bilinear_mae'] = [road_bilinear_mae]
        self.result['road_bilinear_rmse'] = [road_bilinear_rmse]
        self.result['road_bilinear_mape'] = [road_bilinear_mape]
        self.result['road_bilinear_r2'] = [road_bilinear_r2]
        self.result['road_sc'] = [road_sc]
        self.result['road_db'] = [road_db]
        self.result['road_ch'] = [road_ch]
        self.result['road_nmi'] = [road_nmi]
        self.result['road_ars'] = [road_ars]
    def evaluate_region_embedding(self):
        region_emb = np.load(self.region_embedding_path)
        self._logger.info('Load region emb {}, shape = {}'.format(self.region_embedding_path, region_emb.shape))

        self._logger.warning('Evaluating Region Classification')
        y_truth,useful_index,region_micro_f1, region_macro_f1 = self._valid_region_clf(region_emb)

        region_mae, region_rmse, region_mape, region_r2 = self._valid_region_flow(region_emb)
        region_bilinear_mae, region_bilinear_rmse, region_bilinear_r2 , region_bilinear_mape = self._valid_region_flow_using_bilinear(region_emb)

        self._logger.warning('Evaluating Region Emb by TSNE ans Kmeans')
        region_sc, region_db, region_ch, region_nmi, region_ars = self.evaluation_cluster(y_truth,useful_index,region_emb, 'Region')


        self.result['region_micro_f1'] = [region_micro_f1]
        self.result['region_macro_f1'] = [region_macro_f1]
        self.result['region_mae'] = [region_mae]
        self.result['region_rmse'] = [region_rmse]
        self.result['region_mape'] = [region_mape]
        self.result['region_r2'] = [region_r2]
        self.result['region_bilinear_mae'] = [region_bilinear_mae]
        self.result['region_bilinear_rmse'] = [region_bilinear_rmse]
        self.result['region_bilinear_mape'] = [region_bilinear_mape]
        self.result['region_bilinear_r2'] = [region_bilinear_r2]
        self.result['region_sc'] = [region_sc]
        self.result['region_db'] = [region_db]
        self.result['region_ch'] = [region_ch]
        self.result['region_nmi'] = [region_nmi]
        self.result['region_ars'] = [region_ars]
    def evaluate(self):
        if self.representation_object == 'region':
            self.evaluate_region_embedding()
        else:
            self.evaluate_road_embedding()
        result_path = './libcity/cache/{}/evaluate_cache/{}_evaluate_{}_{}_{}.json'. \
            format(self.exp_id, self.exp_id, self.model, self.dataset, str(self.output_dim))
        self._logger.info(self.result)
        # json.dump(self.result, open(result_path, 'w'), indent=4)
        # self._logger.info('Evaluate result is saved at {}'.format(result_path))

        df = pd.DataFrame.from_dict(self.result, orient='columns')
        self._logger.info(df)
        result_path = './libcity/cache/{}/evaluate_cache/{}_evaluate_{}_{}_{}.csv'. \
            format(self.exp_id, self.exp_id, self.model, self.dataset, str(self.output_dim))
        df.to_csv(result_path, index=False)
        self._logger.info('Evaluate result is saved at {}'.format(result_path))
        return self.result

    def save_result(self, save_path, filename=None):
        pass

    def clear(self):
        self.result = {}

