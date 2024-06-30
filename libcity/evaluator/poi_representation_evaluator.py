import csv
import numpy as np
from logging import getLogger
from libcity.evaluator.abstract_evaluator import AbstractEvaluator
from libcity.evaluator.downstream_models.loc_pred_model import *
from sklearn.model_selection import  StratifiedKFold
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import adjusted_rand_score
from libcity.model.poi_representation.static import StaticEmbed, DownstreamEmbed
from sklearn.metrics import normalized_mutual_info_score



class POIRepresentationEvaluator(AbstractEvaluator):

    def __init__(self, config, data_feature):
        self._logger = getLogger()
        self.config = config
        self.data_feature = data_feature
        self.model_name = self.config.get('downstream_model', 'gru')
        self.device = self.config.get('device')
        self.result = {}
        self.model = config.get('model')
        self.dataset = config.get('dataset')
        self.exp_id = config.get('exp_id')
        self.embed_size = config.get('embed_size')

    def collect(self, batch):
        pass

    def evaluate_loc_pre(self):
        pre_model_seq2seq = self.config.get('pre_model_seq2seq', True)
        st_aux_embed_size = self.config.get('st_aux_embed_size', 16)
        st_num_slots = self.config.get('st_num_slots', 10)
        embed_layer = self.data_feature.get('embed_layer')
        num_loc = self.data_feature.get('num_loc')
        embed_size = self.config.get('embed_size', 128)
        hidden_size = embed_size * 4
        pre_len = self.config.get('pre_len', 3)
        task_epoch = self.config.get('task_epoch', 5)
        train_set = self.data_feature.get('train_set')
        test_set = self.data_feature.get('test_set')
        downstream_batch_size = self.data_feature.get('downstream_batch_size', 32)

        
        pre_model = TrajectoryPredictor(embed_layer, num_slots=st_num_slots, aux_embed_size=st_aux_embed_size,
                                           time_thres=10800, dist_thres=0.1,
                                           input_size=embed_size, lstm_hidden_size=hidden_size,
                                           fc_hidden_size=hidden_size, output_size=num_loc, num_layers=2,
                                           seq2seq=pre_model_seq2seq)
        
            
        
        self.result['loc_pre_acc1'], self.result['loc_pre_acc5'], self.result['loc_pre_f1_micro'], self.result['loc_pre_f1_macro'] =\
        loc_prediction(train_set, test_set, num_loc, pre_model, pre_len=pre_len,
                       num_epoch=task_epoch, batch_size=downstream_batch_size, device=self.device)
    
    def evaluate_traj_clf(self):
        embed_layer = self.data_feature.get('embed_layer')
        num_loc = self.data_feature.get('num_loc')
        num_user = self.data_feature.get('num_user')
        embed_size = self.config.get('embed_size', 128)
        hidden_size = embed_size * 4
        task_epoch = self.config.get('task_epoch', 5)
        train_set = self.data_feature.get('train_set')
        test_set = self.data_feature.get('test_set')

        downstream_batch_size = self.data_feature.get('downstream_batch_size', 32)

        clf_model=LstmUserPredictor(embed_layer,embed_size,hidden_size,hidden_size,num_user,num_layers=2,device=self.device)
        
        self.result['traj_clf_acc'], self.result['traj_clf_pre'], self.result['traj_clf_recall'], self.result['traj_clf_f1_micro'], self.result['traj_clf_f1_macro'] =\
        traj_user_classification(train_set, test_set, num_user, num_loc, clf_model,
                       num_epoch=task_epoch, batch_size=downstream_batch_size, device=self.device)
        
    
    def evaluate_loc_clf(self):
        logger=getLogger()
        logger.info('Start training downstream model [loc_clf]...')
        embed_layer = self.data_feature.get('embed_layer')

        if not self.config.get('is_static', True):
            embed_layer=embed_layer.static_embed()
            embed_layer=StaticEmbed(embed_layer)

        num_loc = self.data_feature.get('num_loc')
        embed_size = self.config.get('embed_size', 128)
        task_epoch = self.config.get('task_epoch', 5)
        category = self.data_feature.get('coor_df')
        device=self.config.get('device','cuda:0')
        downstream_batch_size = self.data_feature.get('downstream_batch_size', 32)
        # 随机划分数据集
        assert num_loc == len(category)
        inputs=category.geo_id.to_numpy()
        labels=category.category.to_numpy()
        indices = list(range(num_loc))
        
        np.random.shuffle(indices)
        inputs=inputs[indices]
        labels=labels[indices]
        # 记录num category
        num_class = labels.max()+1
        # 写mlp
        hidden_size = 1024
        clf_model = nn.Sequential(
            nn.Linear(embed_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_class)
        ).to(device)

        embed_layer=embed_layer.to(device)
        
        # optimizer & loss
        optimizer = torch.optim.Adam(clf_model.parameters(), lr=1e-4)
        loss_func = nn.CrossEntropyLoss()
        # kflod test
        skf=StratifiedKFold(n_splits=5)
        score_log = []

        for i,(train_ind,valid_ind) in enumerate(skf.split(inputs,labels)):
            
            for epoch in range(task_epoch):
                for _, batch in enumerate(next_batch(train_ind, downstream_batch_size)):
                    bacth_input = torch.tensor(inputs[batch],dtype=torch.long,device=device)
                    batch_label = torch.tensor(labels[batch],dtype=torch.long,device=device)
                    bacth_input=embed_layer(bacth_input)
                    out=clf_model(bacth_input)
                    loss=loss_func(out,batch_label)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
            pres_raw=[]
            test_labels=[]
            for _, batch in enumerate(next_batch(valid_ind, downstream_batch_size)):
                bacth_input = torch.tensor(inputs[batch],dtype=torch.long,device=device)
                batch_label = torch.tensor(labels[batch],dtype=torch.long,device=device)
                bacth_input=embed_layer(bacth_input)
                out=clf_model(bacth_input)

                pres_raw.append(out.detach().cpu().numpy())
                test_labels.append(batch_label.detach().cpu().numpy())

            pres_raw, test_labels = np.concatenate(pres_raw), np.concatenate(test_labels)
            pres = pres_raw.argmax(-1)

            pre = precision_score(test_labels, pres, average='macro', zero_division=0.0)
            acc, recall = accuracy_score(test_labels, pres), recall_score(test_labels, pres, average='macro', zero_division=0.0)
            f1_micro, f1_macro = f1_score(test_labels, pres, average='micro'), f1_score(test_labels, pres, average='macro')
            score_log.append([acc, pre, recall, f1_micro, f1_macro])
        
        best_acc, best_pre, best_recall, best_f1_micro, best_f1_macro = np.mean(score_log, axis=0)
        logger.info('Acc %.6f, Pre %.6f, Recall %.6f, F1-micro %.6f, F1-macro %.6f' % (
            best_acc, best_pre, best_recall, best_f1_micro, best_f1_macro))
        self.result['loc_clf_acc'] = best_acc
        self.result['loc_clf_pre'] = best_pre
        self.result['loc_clf_recall'] = best_recall
        self.result['loc_clf_f1_micro'] = best_f1_micro
        self.result['loc_clf_f1_macro'] = best_f1_macro
        # 记录结果

    def evaluate_loc_cluster(self):
        embed_layer = self.data_feature.get('embed_layer')

        if not self.config.get('is_static', True):
            embed_layer=embed_layer.static_embed()
            embed_layer=StaticEmbed(embed_layer)
        
        category = self.data_feature.get('coor_df')

        inputs=category.geo_id.to_numpy()
        labels=category.category.to_numpy()
        num_class = labels.max()+1

        node_embedding=embed_layer(torch.tensor(inputs)).cpu()
        self.evaluation_cluster(labels,node_embedding,num_class)
        

    def evaluation_cluster(self, y_truth, node_emb, kinds):
        self._logger.info('Start Kmeans, data.shape = {}, kinds = {}'.format(
            str(node_emb.shape), kinds))
        k_means = KMeans(n_clusters=kinds, random_state=2024)
        k_means.fit(node_emb)
        labels = k_means.labels_
        y_predict = k_means.predict(node_emb)
        y_predict_useful = y_predict
        nmi = normalized_mutual_info_score(y_truth, y_predict_useful)
        ars = adjusted_rand_score(y_truth, y_predict_useful)
        # SC指数
        sc = float(metrics.silhouette_score(node_emb, labels, metric='euclidean'))
        # DB指数
        db = float(metrics.davies_bouldin_score(node_emb, labels))
        # CH指数
        ch = float(metrics.calinski_harabasz_score(node_emb, labels))
        self._logger.info("Evaluate result [loc_cluaster] is sc = {:6f}, db = {:6f}, ch = {:6f}, nmi = {:6f}, ars = {:6f}".format(sc, db, ch, nmi, ars))
        self.result['sc'] = sc
        self.result['db'] = db
        self.result['ch'] = ch
        self.result['nmi'] = nmi
        self.result['ars'] = ars
        return sc, db, ch, nmi, ars

    def evaluate(self):
        self._logger.info('Start evaluating ...')
        self.evaluate_loc_pre()
        self.evaluate_traj_clf()
        poi_type_name = self.config.get('poi_type_name', None)
        if poi_type_name is not None:
            self.evaluate_loc_clf()
            self.evaluate_loc_cluster()
        result_path = './libcity/cache/{}/evaluate_cache/{}_evaluate_{}_{}_{}.json'. \
            format(self.exp_id, self.exp_id, self.model, self.dataset, str(self.embed_size))
        self._logger.info(self.result)

        def dict_to_csv(dictionary, filename):
            with open(filename, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=dictionary.keys())
                writer.writeheader()
                writer.writerow(dictionary)
        result_path = './libcity/cache/{}/evaluate_cache/{}_evaluate_{}_{}_{}.csv'. \
            format(self.exp_id, self.exp_id, self.model, self.dataset, str(self.embed_size))
        dict_to_csv(self.result, result_path)
        self._logger.info('Evaluate result is saved at {}'.format(result_path))

    def save_result(self, save_path, filename=None):
        pass

    def clear(self):
        pass
