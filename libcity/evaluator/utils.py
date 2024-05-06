import json
from heapq import nlargest
import pandas as pd
from libcity.model.loss import *
from tqdm import tqdm
from geopy.distance import geodesic
from datetime import datetime
import os
import pandas as pd


def output(method, value, field):
    """
    Args:
        method: 评估方法
        value: 对应评估方法的评估结果值
        field: 评估的范围, 对一条轨迹或是整个模型
    """
    if method == 'ACC':
        if field == 'model':
            print('---- 该模型在 {} 评估方法下 avg_acc={:.3f} ----'.format(method,
                                                                  value))
        else:
            print('{} avg_acc={:.3f}'.format(method, value))
    elif method in ['MSE', 'RMSE', 'MAE', 'MAPE', 'MARE', 'SMAPE']:
        if field == 'model':
            print('---- 该模型在 {} 评估方法下 avg_loss={:.3f} ----'.format(method,
                                                                   value))
        else:
            print('{} avg_loss={:.3f}'.format(method, value))
    else:
        if field == 'model':
            print('---- 该模型在 {} 评估方法下 avg_acc={:.3f} ----'.format(method,
                                                                  value))
        else:
            print('{} avg_acc={:.3f}'.format(method, value))


def transfer_data(data, model, maxk):
    """
    Here we transform specific data types to standard input type
    """
    if type(data) == str:
        data = json.loads(data)
    assert type(data) == dict, "待评估数据的类型/格式不合法"
    if model == 'DeepMove':
        user_idx = data.keys()
        for user_id in user_idx:
            trace_idx = data[user_id].keys()
            for trace_id in trace_idx:
                trace = data[user_id][trace_id]
                loc_pred = trace['loc_pred']
                new_loc_pred = []
                for t_list in loc_pred:
                    new_loc_pred.append(sort_confidence_ids(t_list, maxk))
                data[user_id][trace_id]['loc_pred'] = new_loc_pred
    return data


def sort_confidence_ids(confidence_list, threshold):
    """
    Here we convert the prediction results of the DeepMove model
    DeepMove model output: confidence of all locations
    Evaluate model input: location ids based on confidence
    :param threshold: maxK
    :param confidence_list:
    :return: ids_list
    """
    """sorted_list = sorted(confidence_list, reverse=True)
    mark_list = [0 for i in confidence_list]
    ids_list = []
    for item in sorted_list:
        for i in range(len(confidence_list)):
            if confidence_list[i] == item and mark_list[i] == 0:
                mark_list[i] = 1
                ids_list.append(i)
                break
        if len(ids_list) == threshold:
            break
    return ids_list"""
    max_score_with_id = nlargest(
        threshold, enumerate(confidence_list), lambda x: x[1])
    return list(map(lambda x: x[0], max_score_with_id))


def evaluate_model(y_pred, y_true, metrics, mode='single', path='metrics.csv'):
    """
    交通状态预测评估函数
    :param y_pred: (num_samples/batch_size, timeslots, ..., feature_dim)
    :param y_true: (num_samples/batch_size, timeslots, ..., feature_dim)
    :param metrics: 评估指标
    :param mode: 单步or多步平均
    :param path: 保存结果
    :return:
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true.shape is not equal to y_pred.shape")
    len_timeslots = y_true.shape[1]
    if isinstance(y_pred, np.ndarray):
        y_pred = torch.FloatTensor(y_pred)
    if isinstance(y_true, np.ndarray):
        y_true = torch.FloatTensor(y_true)
    assert isinstance(y_pred, torch.Tensor)
    assert isinstance(y_true, torch.Tensor)

    df = []
    for i in range(1, len_timeslots + 1):
        line = {}
        for metric in metrics:
            if mode.lower() == 'single':
                if metric == 'masked_MAE':
                    line[metric] = masked_mae_torch(y_pred[:, i - 1], y_true[:, i - 1], 0).item()
                elif metric == 'masked_MSE':
                    line[metric] = masked_mse_torch(y_pred[:, i - 1], y_true[:, i - 1], 0).item()
                elif metric == 'masked_RMSE':
                    line[metric] = masked_rmse_torch(y_pred[:, i - 1], y_true[:, i - 1], 0).item()
                elif metric == 'masked_MAPE':
                    line[metric] = masked_mape_torch(y_pred[:, i - 1], y_true[:, i - 1], 0).item()
                elif metric == 'MAE':
                    line[metric] = masked_mae_torch(y_pred[:, i - 1], y_true[:, i - 1]).item()
                elif metric == 'MSE':
                    line[metric] = masked_mse_torch(y_pred[:, i - 1], y_true[:, i - 1]).item()
                elif metric == 'RMSE':
                    line[metric] = masked_rmse_torch(y_pred[:, i - 1], y_true[:, i - 1]).item()
                elif metric == 'MAPE':
                    line[metric] = masked_mape_torch(y_pred[:, i - 1], y_true[:, i - 1]).item()
                elif metric == 'R2':
                    line[metric] = r2_score_torch(y_pred[:, i - 1], y_true[:, i - 1]).item()
                elif metric == 'EVAR':
                    line[metric] = explained_variance_score_torch(y_pred[:, i - 1], y_true[:, i - 1]).item()
                else:
                    raise ValueError('Error parameter mode={}, please set `single` or `average`.'.format(mode))
            elif mode.lower() == 'average':
                if metric == 'masked_MAE':
                    line[metric] = masked_mae_torch(y_pred[:, :i], y_true[:, :i], 0).item()
                elif metric == 'masked_MSE':
                    line[metric] = masked_mse_torch(y_pred[:, :i], y_true[:, :i], 0).item()
                elif metric == 'masked_RMSE':
                    line[metric] = masked_rmse_torch(y_pred[:, :i], y_true[:, :i], 0).item()
                elif metric == 'masked_MAPE':
                    line[metric] = masked_mape_torch(y_pred[:, :i], y_true[:, :i], 0).item()
                elif metric == 'MAE':
                    line[metric] = masked_mae_torch(y_pred[:, :i], y_true[:, :i]).item()
                elif metric == 'MSE':
                    line[metric] = masked_mse_torch(y_pred[:, :i], y_true[:, :i]).item()
                elif metric == 'RMSE':
                    line[metric] = masked_rmse_torch(y_pred[:, :i], y_true[:, :i]).item()
                elif metric == 'MAPE':
                    line[metric] = masked_mape_torch(y_pred[:, :i], y_true[:, :i]).item()
                elif metric == 'R2':
                    line[metric] = r2_score_torch(y_pred[:, :i], y_true[:, :i]).item()
                elif metric == 'EVAR':
                    line[metric] = explained_variance_score_torch(y_pred[:, :i], y_true[:, :i]).item()
                else:
                    raise ValueError('Error parameter metric={}!'.format(metric))
            else:
                raise ValueError('Error parameter evaluator_mode={}, please set `single` or `average`.'.format(mode))
        df.append(line)
    df = pd.DataFrame(df, columns=metrics)
    print(df)
    df.to_csv(path)
    return df


# utils for road representaion downstream data
def geo2distance(coordinates):
    res = 0
    for c in range(len(coordinates) - 1):
        lo1, la1 = coordinates[c]
        lo2, la2 = coordinates[c + 1]
        res += geodesic((la1, lo1), (la2, lo2)).km
    return res


def getSpeedAndTime(traj_list, geo2length, geo2speed):
    for i in range(len(traj_list) - 1):
        traj1 = traj_list[i]
        traj2 = traj_list[i + 1]
        geo_id1 = traj1['geo_id']

        t1 = datetime.strptime(traj1['time'], '%Y-%m-%d %H:%M:%S')
        t2 = datetime.strptime(traj2['time'], '%Y-%m-%d %H:%M:%S')
        length = geo2length[geo_id1]
        t_delta = float((t2 - t1).seconds)
        if t_delta > 0:
            v_temp = length * 1000 / t_delta

            avg, n = geo2speed.get(geo_id1, (0, 0))
            geo2speed[geo_id1] = ((avg * n + v_temp) / (n + 1), n + 1)

    traj_series = [x['geo_id'] for x in traj_list]
    t0 = datetime.strptime(traj_list[0]['time'], '%Y-%m-%d %H:%M:%S')
    t_ = datetime.strptime(traj_list[-1]['time'], '%Y-%m-%d %H:%M:%S')
    totaltime = (t_ - t0).seconds
    return traj_series, totaltime


def generate_road_representaion_downstream_data(dataset_name):
    save_data_path = os.path.join('libcity/cache/dataset_cache', dataset_name, "label_data")
    if not os.path.exists(save_data_path):
        os.makedirs(save_data_path)
    # read files
    traj_df_reader = pd.read_csv(os.path.join('raw_data', dataset_name, dataset_name + '.dyna'), low_memory=False, sep=',', chunksize=100)
    geo_df = pd.read_csv(os.path.join('raw_data', dataset_name, dataset_name + '.geo'), low_memory=False)
    # length.csv
    if not os.path.exists(os.path.join(save_data_path, "length.csv")):
        geo2length = {}
        for index, row in tqdm(geo_df.iterrows()):
            geo_id = row['geo_id']
            if geo_df.iloc[geo_id]['traffic_type'] == 'road':
                coordinates = eval(geo_df['coordinates'][geo_id])
                geo2length[geo_id] = geo2distance(coordinates)

        geo2lengthdf = pd.DataFrame.from_dict(geo2length, orient='index', columns=['length'])
        geo2lengthdf = geo2lengthdf.reset_index().rename(columns={'index': 'geo_id'})
        geo2lengthdf.to_csv(os.path.join(save_data_path, 'length.csv'), index=False)

    # speed.csv and time.csv for speed inference and time estimation task
    if not os.path.exists(os.path.join(save_data_path, "time.csv")) or \
        not os.path.exists(os.path.join(save_data_path, "speed.csv")):
        geo2lengthdf = pd.read_csv(os.path.join(save_data_path, 'length.csv'))
        geo2length = dict(zip(geo2lengthdf['geo_id'], geo2lengthdf['length']))
        geo2speed = {}
        traj_id_temp = 0
        traj_list = []
        trajAndtime = []
        for chunk in tqdm(traj_df_reader):
            for index, row in chunk.iterrows():
                if row['geo_id'] not in geo2length.keys():
                    continue
                if type(row['total_traj_id']) == str:
                    traj_id = eval(row['total_traj_id'])
                else:
                    traj_id = row['total_traj_id']
                if traj_id != traj_id_temp and traj_list != []:
                    trajs, time = getSpeedAndTime(traj_list, geo2length, geo2speed)
                    trajAndtime.append([trajs, time])
                    traj_id_temp = traj_id
                    traj_list = [row]
                else:
                    traj_list.append(row)

        geo2speeddf = pd.DataFrame.from_dict(geo2speed, orient='index', columns=['speed', 'freq'])
        geo2speeddf = geo2speeddf.reset_index().rename(columns={'index': 'index'})
        geo2speeddf.to_csv(os.path.join(save_data_path, 'speed.csv'), index=False)
        geo2timedf = pd.DataFrame(data=trajAndtime, columns=['trajs', 'time'])
        geo2timedf.to_csv(os.path.join(save_data_path, 'time.csv'), index=True)
