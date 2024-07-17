import itertools
import os
import torch
import pandas as pd
import numpy as np
import torch.utils.data as data_utils

from veccity.data.dataset.dataset_subclass.utils import construct_spatial_matrix_accordingDistance,tid_list_48
from veccity.data.dataset.dataset_subclass.cacsr_sample import cacsr_sample

class CacsrData:
    def __init__(self,config,data_feature):
        self.distance_theta = data_feature.get('distance_theta',1)
        self.gaussian_beta = data_feature.get('gaussian_beta',10)
        self.max_his_period_days = data_feature.get('max_his',120)
        self.max_merge_seconds_limit = data_feature.get('max_merge_seconds_limit',10800)
        self.max_delta_mins = data_feature.get('max_delta_mins',1440)
        self.min_session_mins = data_feature.get('min_session_mins',1440)
        self.latN = data_feature.get('latN',50)
        self.lngN = data_feature.get('lngN',50)
        dirname = os.path.join(os.getcwd(), 'veccity','cache', 'dataset_cache', config['dataset'],'cacsr')
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        train_save_filename = os.path.join(os.getcwd(), 'veccity','cache', 'dataset_cache', config['dataset'],'cacsr','train.npz')
        ss_save_filename = os.path.join(os.getcwd(), 'veccity','cache', 'dataset_cache', config['dataset'],'cacsr','ss.npz')
        venue_cnt = data_feature['num_loc']
        venue_lat = [coord[1] for coord in data_feature['coor_mat']]
        venue_lng = [coord[2] for coord in data_feature['coor_mat']]
        venue_category = data_feature['coor_df']['category'].values
        category_cnt = len(set(venue_category))
        venueId_lidx = {venue: idx for idx, venue in enumerate(data_feature['coor_df']['geo_id'])}
        if os.path.exists(ss_save_filename):
            ss = np.load(ss_save_filename)
            SS_distance = ss['SS_distance']
            SS_proximity = ss['SS_proximity']
            SS_gaussian_distance = ss['SS_gaussian_distance']
        else:
            print('Constructing spatial matrix...')
            SS_distance, SS_proximity, SS_gaussian_distance = construct_spatial_matrix_accordingDistance(self.distance_theta, venue_cnt, venue_lng, venue_lat, gaussian_beta=self.gaussian_beta)
            
            np.savez_compressed(ss_save_filename, SS_distance=SS_distance, SS_proximity=SS_proximity, SS_gaussian_distance=SS_gaussian_distance)
        if os.path.exists(train_save_filename):
            print('Loading data from cache...')
            
        else :
            max_lat = max(venue_lat)
            min_lat = min(venue_lat)
            max_lng = max(venue_lng)
            min_lng = min(venue_lng)

            lats = []
            lngs = []
            for i in range(venue_cnt):
                lats.append(venue_lat[i])
                lngs.append(venue_lng[i])

            venue_latidx = {}
            venue_lngidx = {}
            for i in range(venue_cnt):
                eps = 1e-7
                latidx = int((venue_lat[i]-min_lat)*self.latN/(max_lat - min_lat + eps)) 
                lngidx = int((venue_lng[i]-min_lng)*self.lngN/(max_lng - min_lng + eps))
                venue_latidx[i]= latidx if latidx < self.latN else (self.latN-1)
                venue_lngidx[i]= lngidx if lngidx < self.lngN else (self.lngN-1)

            feature_category = []
            feature_lat = []
            feature_lng = []
            feature_lat_ori = []
            feature_lng_ori = []
            for i in range(venue_cnt):
                feature_category.append(venue_category[i])
                feature_lat.append(venue_latidx[i])
                feature_lng.append(venue_lngidx[i])
                feature_lat_ori.append(venue_lat[i])
                feature_lng_ori.append(venue_lng[i])

            sample_constructor = cacsr_sample(venueId_lidx, SS_distance=SS_distance, SS_gaussian_distance=SS_gaussian_distance, max_his_period_days=self.max_his_period_days, 
                                              max_merge_seconds_limit=self.max_merge_seconds_limit, max_delta_mins=self.max_delta_mins, min_session_mins=self.min_session_mins)
            checkins_filter = pd.DataFrame(data_feature['df'], copy=True)
            checkins_filter['local time'] = pd.to_datetime(checkins_filter["datetime"])
            
            checkins_filter['timestamp'] = checkins_filter['local time'].apply(lambda x: x.timestamp()) 
            checkins_filter['local weekday'] = checkins_filter['local time'].apply(lambda x: x.weekday())
            checkins_filter['local hour'] = checkins_filter['local time'].apply(lambda x: x.hour)
            checkins_filter['local minute'] = checkins_filter['local time'].apply(lambda x: x.minute)
            checkins_filter['lid'] = checkins_filter['loc_index'].apply(lambda x: venueId_lidx[x]) 
            checkins_filter.rename(columns={'user_index': 'userId'}, inplace=True)
            #print('after filtering, %d check-ins points are left.' % len(checkins_filter), flush=True)
            #print("checkins_filter's columns: ", checkins_filter.columns, checkins_filter.dtypes)           
            userId_checkins = checkins_filter.groupby('userId')

            all_drops = []
            all_drops_ratio = []
            for userId, checkins in userId_checkins:
                #print('userId:', userId, flush=True)
                uid = sample_constructor.user_cnt
                #print('uid:', uid, flush=True)
                checkins = checkins.sort_values(by=['timestamp'])  
                checkins = checkins.reset_index(drop=True)  

                tmp_len = len(checkins)
                checkins, drops = sample_constructor.deal_cluster_sequence_for_each_user(checkins) 
                all_drops.append(drops) 
                all_drops_ratio.append(drops / tmp_len) 

                total = len(checkins)
                user_lidFreq = (checkins[:].groupby(['lid']).count()).iloc[:, [0]]/total
                lid_visitFreq = venue_cnt * [0]
                for index, row in user_lidFreq.iterrows(): 
                    lid_visitFreq[index] = row['userId']

                flag = sample_constructor.construct_sample_seq2seq(checkins, uid)
                if flag:
                    sample_constructor.userId2uid[userId] = uid
                    sample_constructor.user_cnt += 1
                    sample_constructor.user_lidfreq.append(lid_visitFreq)
            np.savez_compressed(train_save_filename,
                                trainX_target_lengths=sample_constructor.trainX_target_lengths,
                                trainX_arrival_times=sample_constructor.trainX_arrival_times,
                                trainX_duration2first=sample_constructor.trainX_duration2first,
                                trainX_session_arrival_times=sample_constructor.trainX_session_arrival_times,
                                trainX_local_weekdays=sample_constructor.trainX_local_weekdays,
                                trainX_session_local_weekdays=sample_constructor.trainX_session_local_weekdays,
                                trainX_local_hours=sample_constructor.trainX_local_hours,
                                trainX_session_local_hours=sample_constructor.trainX_session_local_hours,
                                trainX_local_mins=sample_constructor.trainX_local_mins,
                                trainX_session_local_mins=sample_constructor.trainX_session_local_mins,
                                trainX_delta_times=sample_constructor.trainX_delta_times,
                                trainX_session_delta_times=sample_constructor.trainX_session_delta_times,
                                trainX_locations=sample_constructor.trainX_locations,
                                trainX_session_locations=sample_constructor.trainX_session_locations,
                                trainX_last_distances=sample_constructor.trainX_last_distances,
                                trainX_users=sample_constructor.trainX_users, trainX_lengths=sample_constructor.trainX_lengths,
                                trainX_session_lengths=sample_constructor.trainX_session_lengths,
                                trainX_session_num=sample_constructor.trainX_session_num,
                                trainY_arrival_times=sample_constructor.trainY_arrival_times,
                                trainY_delta_times=sample_constructor.trainY_delta_times,
                                trainY_locations=sample_constructor.trainY_locations,
                                user_lidfreq=sample_constructor.user_lidfreq,
                                us=sample_constructor.us, vs=sample_constructor.vs,

                                feature_category=feature_category, feature_lat=feature_lat, feature_lng=feature_lng,
                                feature_lat_ori=feature_lat_ori, feature_lng_ori=feature_lng_ori,

                                latN=self.latN, lngN=self.lngN, category_cnt=category_cnt,

                                user_cnt=sample_constructor.user_cnt, venue_cnt=sample_constructor.venue_cnt,

                                SS_distance=sample_constructor.SS_distance, SS_guassian_distance=sample_constructor.SS_gaussian_distance)

            
        loader = np.load(train_save_filename,allow_pickle=True)
        user_cnt = loader['user_cnt']
        venue_cnt = loader['venue_cnt']
        feature_category = loader['feature_category']
        feature_lat = loader['feature_lat']  # index
        feature_lng = loader['feature_lng']  # index

        # put spatial point features into tensor
        self.feature_category = torch.LongTensor(feature_category)
        self.feature_lat = torch.LongTensor(feature_lat)
        self.feature_lng = torch.LongTensor(feature_lng)

        self.latN, self.lngN = loader['latN'], loader['lngN']
        self.category_cnt = loader['category_cnt']

        # ----- load train / val / test to get dataset -----
        self.data_train = self.load_data_from_dataset('train', loader, user_cnt, venue_cnt)
        self.collate = collate_session_based
   
    def load_data_from_dataset(self,set_name, loader, user_cnt, venue_cnt) :
        X_target_lengths = loader[f'{set_name}X_target_lengths']
        X_arrival_times = loader[f'{set_name}X_arrival_times']
        X_users = loader[f'{set_name}X_users']
        X_locations = loader[f'{set_name}X_locations']
        Y_location = loader[f'{set_name}Y_locations']

        X_all_loc = []
        X_all_tim = []
        X_lengths = []

        for i in range(len(X_arrival_times)):
            tim = X_arrival_times[i]
            loc = X_locations[i]

            len_ = len(tim)
            for j in range(len_):
                tim[j] = tid_list_48(tim[j]) 

            X_all_loc.append(loc)
            X_all_tim.append(tim)
            X_lengths.append(len_)

        #print("X_all_loc: ", len(X_all_loc), X_all_loc[0])
        #print("X_all_tim: ", len(X_all_tim), X_all_tim[0])
        #print("X_target_lengths: ", len(X_target_lengths), X_target_lengths[0])
        #print("X_lengths: ", len(X_lengths), X_lengths[0])
        #print("X_users:", len(X_users), X_users)
        #print("Y_location:", len(Y_location), Y_location[0])

        dataset = SessionBasedSequenceDataset(user_cnt, venue_cnt, X_users, X_all_loc,
                                              X_all_tim, Y_location, X_target_lengths, X_lengths, None)
        #print(f'samples cnt of data_{set_name}:', dataset.real_length())

        return dataset

class SessionBasedSequenceDataset(data_utils.Dataset):
    """Dataset class containing variable length sequences.
    """

    def __init__(self, user_cnt, venue_cnt, X_users, X_all_loc,
                 X_all_tim, Y_location, target_lengths, X_lengths, X_all_text):
        # torch.set_default_tensor_type(torch.cuda.FloatTensor)
        self.user_cnt = user_cnt
        self.venue_cnt = venue_cnt
        self.X_users = X_users
        self.X_all_loc = X_all_loc
        self.X_all_tim = X_all_tim
        self.target_lengths = target_lengths
        self.X_lengths = X_lengths
        self.Y_location = Y_location
        self.X_all_text = X_all_text
        self.validate_data()

    @property
    def num_series(self):
        return len(self.Y_location)

    def real_length(self):  
        res = 0
        n = len(self.Y_location)
        for i in range(n):
            res += len(self.Y_location[i])
        return res

    def validate_data(self):
        if len(self.X_all_loc) != len(self.Y_location) or len(self.X_all_tim) != len(self.Y_location):
            raise ValueError("Length of X_all_loc, X_all_tim, Y_location should match")

    def __getitem__(self, key):
        '''
        the outputs are feed into collate()
        :param key:
        :return:
        '''
        return self.X_all_loc[key], self.X_all_tim[key], None, self.Y_location[key], self.target_lengths[key], \
               self.X_lengths[key], self.X_users[key]

    def __len__(self):
        return self.num_series

    def __repr__(self):  
        pass

def collate_session_based(batch,device):
    '''
    get the output of dataset.__getitem__, and perform padding
    :param batch:
    :return:
    '''

    batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)  

    X_all_loc = [item[0] for item in batch]
    X_all_tim = [item[1] for item in batch]
    X_all_text = [item[2] for item in batch]
    Y_location = [lid for item in batch for lid in item[3]] 
    target_lengths = [item[4] for item in batch]
    X_lengths = [item[5] for item in batch]
    X_users = [item[6] for item in batch]

    padded_X_all_loc = pad_session_data_one(X_all_loc)
    padded_X_all_tim = pad_session_data_one(X_all_tim)
    padded_X_all_loc = torch.tensor(padded_X_all_loc).long().to(device)
    padded_X_all_tim = torch.tensor(padded_X_all_tim).long().to(device)

    return session_Batch(padded_X_all_loc, padded_X_all_tim, X_all_text, Y_location, target_lengths, X_lengths, X_users,
                         device)


class session_Batch():
    def __init__(self, padded_X_all_loc, padded_X_all_tim, X_all_text, Y_location, target_lengths, X_lengths, X_users,
                 device):
        self.X_all_loc = torch.LongTensor(padded_X_all_loc).to(device)  # (batch, max_all_length)
        self.X_all_tim = torch.LongTensor(padded_X_all_tim).to(device)  # (batch, max_all_length)
        self.X_all_text = X_all_text 
        self.Y_location = torch.Tensor(Y_location).long().to(device)  # (Batch,) 
        self.target_lengths = target_lengths 
        self.X_lengths = X_lengths 
        self.X_users = torch.Tensor(X_users).long().to(device)
        
        
def pad_session_data_one(data):
    fillvalue = 0
    # zip_longest
    data = list(zip(*itertools.zip_longest(*data, fillvalue=fillvalue)))
    res = []
    res.extend([list(data[i]) for i in range(len(data))])

    return res