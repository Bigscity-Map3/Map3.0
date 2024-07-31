import os
import math
import pandas as pd
import numpy as np
from logging import getLogger
import faiss
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import copy
from veccity.data.dataset.dataset_subclass.sts_dataset import STSDataset

from veccity.evaluator.downstream_models.abstract_model import AbstractModel
from veccity.data.preprocess import preprocess_detour

class STSModel(nn.Module):
    def __init__(self, embedding,device,dropout_prob=0.2):
        super().__init__()
        self.traj_encoder = embedding
        self.dropout = nn.Dropout(p=dropout_prob)
        self.criterion = torch.nn.CrossEntropyLoss(reduction='mean')
        self.device=device
        self.temperature=0.05
        
    def forward(self, batch):
        out=self.dropout(self.traj_encoder.encode_sequence(batch)) # bxd
        return out

    def calculate_loss(self,batch):
        batch_view2=copy.copy(batch)
        out_view1=self.forward(batch)
        out_view2=self.forward(batch_view2)
        out_view1 = F.normalize(out_view1, dim=-1)
        out_view2 = F.normalize(out_view2, dim=-1)

        similarity_matrix = F.cosine_similarity(out_view1.unsqueeze(1), out_view2.unsqueeze(0), dim=-1)
        similarity_matrix /= self.temperature
        labels = torch.arange(similarity_matrix.shape[0]).long().to(self.device)
        loss_res = self.criterion(similarity_matrix, labels)
        return loss_res


class STSExecutor(AbstractModel):
    def __init__(self, config):
        preprocess_detour(config)
        self._logger = getLogger()
        self._logger.warning('Evaluating Trajectory Similarity Search')
        self.dataset=STSDataset(config)
        self.train_dataloader,self.eval_dataloader,self.test_dataloader = self.dataset.get_data()
        self.device=config.get('device')        
        self.epochs=0#config.get('task_epoch',10)
        self.learning_rate=config.get('learning_rate',1e-4)
        self.weight_decay=config.get('weight_decay',1e-3)
    
    def run(self,model,**kwargs):
        self._logger.info("-------- STS START --------")
        self.train(model,**kwargs)
        self.evaluation()
        return self.result

    def train(self,model,**kwargs):
        """
        返回评估结果
        """
        self.model = STSModel(embedding=model, device=self.device)
        self.train_dataloader=self.train_dataloader
        self.model.to(self.device)
        optimizer = Adam(lr=self.learning_rate, params=self.model.parameters(), weight_decay=self.weight_decay)
        best_loss=-1
        best_model=None
        best_epoch=0
        for epoch in range(self.epochs):
            total_loss = 0.0
            for step,batch in enumerate(self.train_dataloader):
                loss=self.model.calculate_loss(batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            total_loss=total_loss/len(self.train_dataloader)
            
            valid_loss=0.0
            with torch.no_grad():
                for batch in self.eval_dataloader:
                    loss=self.model.calculate_loss(batch)
                    valid_loss += loss.item()
            valid_loss=valid_loss/len(self.eval_dataloader)

            if best_loss==-1 or valid_loss<best_loss:
                best_model=copy.deepcopy(self.model)
                best_loss=valid_loss
                best_epoch=epoch
            self._logger.info("epoch {} complete! training loss {:.2f}, valid loss {:2f}, best_epoch {}, best_loss {:2f}".format(epoch, total_loss, valid_loss,best_epoch,best_loss))
        if best_model:
            self.model=best_model

        self.evaluation()
        return self.result

    def evaluation(self):
        ori_dataloader=self.dataset.ori_dataloader
        qry_dataloader=self.dataset.qry_dataloader
        num_queries=len(qry_dataloader)

        self.model.eval()
        x = []

        for batch in ori_dataloader:
            seq_rep = self.model.traj_encoder.encode_sequence(batch)
            if isinstance(seq_rep, tuple):
                seq_rep = seq_rep[0]
            x.append(seq_rep.detach().cpu())
        x = torch.cat(x, dim=0).numpy()

        q = []
        for batch in qry_dataloader:
            seq_rep = self.model.traj_encoder.encode_sequence(batch)
            if isinstance(seq_rep, tuple):
                seq_rep = seq_rep[0]
            q.append(seq_rep.detach().cpu())
        q = torch.cat(q, dim=0).numpy()

        y=np.arange(x.shape[0])
        d = x.shape[1]                                           # 向量维度
        nb = x.shape[0]
        index_type = 'Flat'                              # index 类型
        metric_type = faiss.METRIC
        index = faiss.IndexFlatL2(d)
        index.add(x)
        D, I = index.search(q, 10000)
        self.result = {}
        top = [1, 3, 5, 10, 20]
        for k in top:
            hit = 0
            rank_sum = 0
            no_hit = 0
            for i, r in enumerate(I):
                if y[i] in r:
                    rank_sum += np.where(r == y[i])[0][0]
                    if y[i] in r[:k]:
                        hit += 1
                else:
                    no_hit += 1
        
            self.result['Mean Rank'] = rank_sum / num_queries + 1.0
            self.result['No Hit'] = no_hit 
            self.result['HR@' + str(k)] =  hit / (num_queries - no_hit)
            self._logger.info(f'HR@{k}: {hit / (num_queries - no_hit)}')
        self._logger.info('Mean Rank: {}, No Hit: {}'.format(self.result['Mean Rank'], self.result['No Hit']))

    def save_result(self, save_path, filename=None):
        pass

    def clear(self):
        pass