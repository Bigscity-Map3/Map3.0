import numpy as np
import os
import pandas as pd
from libcity.data.dataset.poi_representation_dataset import PoiRepresentationDataset


class CTLEDataset(PoiRepresentationDataset):
    def __init__(self, config):
        self.config = config
        self.dataset = self.config.get('dataset', '')
        self.data_path = './raw_data/' + self.dataset + '/'
        self.geo_file = self.config.get('geo_file', self.dataset)
        self.rel_file = self.config.get('rel_file', self.dataset)
        self.dyna_file = self.config.get('dyna_file', self.dataset)
        if not os.path.exists('./libcity/cache/GEOTEASER_{}'.format(self.dataset)):
            os.mkdir('./libcity/cache/GEOTEASER_{}'.format(self.dataset))
        super().__init__(config)


        self.users = self.embed_train_entitys
        self.distance_threshold = self.config.get("distance_threshold", 0.2)
        self.sample = self.config.get("sample", 1e-3)

        self.coordinates  = self.traj_poi[['poi_id','lat','lng']].drop_duplicates('poi_id').to_numpy()


    def get_data_feature(self):
        self.user_num = self.traj_poi['entity_id'].value_counts().count()
        self.poi_num = self.traj_poi['poi_id'].value_counts().count()
        self.user_ids, self.src_tokens, self.src_weekdays, \
        self.src_ts, self.src_lens = zip(*self.gen_sequence())
        return { "num_loc": self.poi_num,
                "user_num": self.user_num,
                 "user_ids" : self.user_ids,
                 "src_tokens":self.src_tokens,
                "src_weekdays":self.src_weekdays,
                "src_ts":self.src_ts,
                 "src_lens":self.src_lens,
                 "max_seq_len":self.max_seq_len
                }
