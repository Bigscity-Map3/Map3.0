import importlib
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset
from logging import getLogger
from sklearn import manifold
from sklearn.cluster import KMeans
from sklearn import metrics
from libcity.evaluator.abstract_evaluator import AbstractEvaluator
from sklearn.metrics import accuracy_score, f1_score
import os
from sklearn import linear_model
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import manifold
from sklearn.preprocessing import normalize
from libcity.data.preprocess import cache_dir


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
        self.config = config
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

    def _valid_clf(self, emb):
        data = pd.read_csv(os.path.join('raw_data', self.dataset, self.dataset + '.geo'))
        if self.representation_object == 'region':
            label = data['region_FUNCTION'].dropna().astype(int).values
        else:
            label = data['road_highway'].dropna().astype(int).values
        num_classes = self.config.get('clf_num_classes', 5)
        tmp = []
        for i in range(label.min(), label.max() + 1):
            tmp.append((label[label == i].shape[0], i))
        assert num_classes <= len(tmp)
        tmp.sort()
        tmp = [item[1] for item in tmp]
        useful_label = tmp[::-1][:num_classes]
        relabel = {}
        for i, j in enumerate(useful_label):
            relabel[j] = i
        useful_index = []
        self._logger.info('Road emb shape = {}, label shape = {}'.format(emb.shape, label.shape))
        assert len(label) == len(emb)

        X = []
        y = []
        for i, label_i in enumerate(label):
            if label_i in useful_label:
                useful_index.append(i)
                X.append(emb[i: i + 1, :])
                y.append(relabel[label_i])
        X = np.concatenate(X, axis=0)
        y = np.array(y)

        self._logger.info(
            'Selected Road emb shape = {}, label shape = {}, label min = {}, label max = {}, num_classes = {}'.format(
                X.shape, y.shape, y.min(), y.max(), num_classes))
        road_micro_f1, road_macro_f1 = evaluation_classify(X, y, kfold=5, num_classes=num_classes, seed=self.seed,
                                                           output_dim=self.output_dim)
        self._logger.info('micro F1: {:6f}, macro F1: {:6f}'.format(road_micro_f1, road_macro_f1))
        return y,useful_index,road_micro_f1, road_macro_f1
    
    def _valid_flow_using_bilinear(self, emb):
        self._logger.warning(f'Evaluating {self.representation_object} OD-Flow Prediction Using Bilinear Module')
        od_flow = np.load(os.path.join(cache_dir, self.dataset, f'traj_{self.representation_object}_test_od.npy')).astype('float32')
        mae,rmse,mape,r2 = evaluation_bilinear_reg(emb, od_flow, kfold=5, seed=self.seed, output_dim=self.output_dim)
        self._logger.info(f"Result of odflow bilinear estimation in {self.dataset}:")
        self._logger.info('MAE = {:6f}, RMSE = {:6f}, R2 = {:6f}, MAPE = {:6f}'.format(mae,rmse,r2,mape))
        return mae, rmse, r2, mape
    
    def _valid_flow(self, emb):
        self._logger.warning(f'Evaluating {self.representation_object} In-Flow Prediction')
        inflow = np.load(os.path.join(cache_dir, self.dataset, f'traj_{self.representation_object}_test_in_avg.npy')).astype('float32')
        in_mae, in_rmse, in_mape, in_r2 = evaluation_reg(emb, inflow / 24, kfold=5, seed=self.seed, output_dim=self.output_dim)
        self._logger.info(f"Result of inflow estimation in {self.dataset}:")
        self._logger.info('MAE = {:6f}, RMSE = {:6f}, R2 = {:6f}, MAPE = {:6f}'.format(in_mae, in_rmse, in_r2, in_mape))

        self._logger.warning(f'Evaluating {self.representation_object} Out-Flow Prediction')
        outflow = np.load(os.path.join(cache_dir, self.dataset, f'traj_{self.representation_object}_test_out_avg.npy')).astype('float32')
        out_mae, out_rmse, out_mape, out_r2 = evaluation_reg(emb, outflow / 24, kfold=5, seed=self.seed, output_dim=self.output_dim)
        self._logger.info("Result of {} estimation in {}:".format('outflow', self.dataset))
        self._logger.info('MAE = {:6f}, RMSE = {:6f}, R2 = {:6f}, MAPE = {:6f}'.format(out_mae, out_rmse, out_r2, out_mape))

        mae = (in_mae + out_mae) / 2
        rmse = (in_rmse + out_rmse) / 2
        mape = (in_mape + out_mape) / 2
        r2 = (in_r2 + out_r2) / 2
        self._logger.info("Result of road flow estimation in {}:".format(self.dataset))
        self._logger.info('MAE = {:6f}, RMSE = {:6f}, R2 = {:6f}, MAPE = {:6f}'.format(mae, rmse, r2, mape))
        return mae, rmse, r2, mape
    
    def evaluate_embedding(self):
        if self.representation_object == 'road':
            embedding_path = self.road_embedding_path
        else:
            embedding_path = self.region_embedding_path
        emb = np.load(embedding_path)
        self._logger.info(f'Load {self.representation_object} emb {embedding_path}, shape = {emb.shape}')
        self._logger.warning('Evaluating Road Classification')
        y_truth,useful_index,micro_f1, macro_f1 = self._valid_clf(emb)
        mae, rmse, r2, mape = self._valid_flow(emb)
        bilinear_mae,bilinear_rmse,bilinear_r2,bilinear_mape = self._valid_flow_using_bilinear(emb)
        self._logger.warning('Evaluating Road Emb by TSNE ans Kmeans')
        sc, db, ch, nmi, ars = self.evaluation_cluster(y_truth,useful_index,emb, self.representation_object)
        self.result['micro_f1'] = [micro_f1]
        self.result['macro_f1'] = [macro_f1]
        self.result['mae'] = [mae]
        self.result['rmse'] = [rmse]
        self.result['mape'] = [mape]
        self.result['r2'] = [r2]
        self.result['bilinear_mae'] = [bilinear_mae]
        self.result['bilinear_rmse'] = [bilinear_rmse]
        self.result['bilinear_mape'] = [bilinear_mape]
        self.result['bilinear_r2'] = [bilinear_r2]
        self.result['sc'] = [sc]
        self.result['db'] = [db]
        self.result['ch'] = [ch]
        self.result['nmi'] = [nmi]
        self.result['ars'] = [ars]

    def get_downstream_model(self, model):
        try:
            return getattr(importlib.import_module('libcity.evaluator.downstream_models'), model)(self.config)
        except AttributeError:
            raise AttributeError('evaluate model is not found')

    def evaluate(self):
        self.evaluate_embedding()
        downstream_model = self.get_downstream_model('SimilaritySearchModel')
        self.result.update(downstream_model.run())
        result_path = './libcity/cache/{}/evaluate_cache/{}_evaluate_{}_{}_{}.json'. \
            format(self.exp_id, self.exp_id, self.model, self.dataset, str(self.output_dim))
        self._logger.info(self.result)
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

