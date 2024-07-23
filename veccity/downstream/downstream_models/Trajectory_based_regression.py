from abc import ABC

import numpy as np
import torch
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, label_ranking_average_precision_score
from sklearn.utils import shuffle
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from logging import getLogger
from veccity.upstream.point.utils import next_batch, create_src_trg, weight_init, top_n_accuracy
from torch.nn.utils.rnn import pad_sequence,pad_packed_sequence,pack_padded_sequence
from veccity.downstream.utils import accuracy
import copy
import torch.nn.functional as F

class Trajectory_based_regression(nn.Module, ABC):
    def __init__(self, embed_size, hidden_size, encoder_model, device):
        super().__init__()
        self.encoder_model = encoder_model
        self.hidden_size=hidden_size
        self.embed_size=embed_size
        self.device=device

        self.fc = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                        nn.ReLU(), nn.Linear(hidden_size, 1))
    
        self.apply(weight_init)

    def forward(self, seq,valid_len,**kwargs):
        traj_embed = self.encoder_model.encode(seq, valid_len, **kwargs)
        pred = self.fc(traj_embed)
        return pred
    
class TbRExecutor:
    def __init__(self,encoder_model,embed_size,hidden_size,learner='adam',lr=1e-3,max_epoch=100,device='cpu') -> None:

        self.embed_size=embed_size
        self.hidden_size=hidden_size
        self.device = device
        self.encoder_model =encoder_model
        self.learner = learner
        self.lr=lr
        self.max_epoch = max_epoch
        self.model = Trajectory_based_regression(self.embed_size,self.hidden_size,self.encoder_model,self.device)
        self._logger = getLogger()

        self.optimizer = self._build_optimizer()
    
    def _build_optimizer(self):
        """
        根据全局参数`learner`选择optimizer
        """
        self._logger.info('You select `{}` optimizer.'.format(self.learner.lower()))
        if self.learner.lower() == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        elif self.learner.lower() == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        elif self.learner.lower() == 'adagrad':
            optimizer = torch.optim.Adagrad(self.model.parameters(), lr=self.lr)
        elif self.learner.lower() == 'rmsprop':
            optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.lr)
        elif self.learner.lower() == 'sparse_adam':
            optimizer = torch.optim.SparseAdam(self.model.parameters(), lr=self.lr)
        elif self.learner.lower() == 'asgd':
            optimizer = torch.optim.ASGD(self.model.parameters(), lr=self.lr)
        else:
            self._logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer

    def metrics_fn(self,y_pred,y_true,metric):
        if metric=='rmse':
            return np.sqrt(np.mean((y_pred-y_true)**2))
        elif metric=='mae':
            return np.mean(np.abs(y_pred-y_true))
        elif metric=='r2':
            return 1-np.sum((y_pred-y_true)**2)/np.sum((y_true-np.mean(y_true))**2)
        elif metric=='mape':
            return np.mean(np.abs((y_pred-y_true)/y_true))
        
    @torch.no_grad() 
    def valid(self,valid_data):
        self.model.eval()
        labels=[]
        preds=[]
        for batch in valid_data:
            seq, valid_len, label = batch
            seq, valid_len, label = seq.to(self.device), valid_len.to(self.device), label.to(self.device)
            pred = self.model(seq, valid_len)
            labels.append(label)
            preds.append(pred)

        return self.metrics_fn(preds,labels,"mae")
    
    def train(self,train_data,valid_data):
        self.model.train()
            
        for epoch in range(self.max_epoch):
            train_loss = 0
            train_acc = 0
            best_acc = 0
            best_model=None
            patience=10
            self.model.train()
            for batch in train_data:
                seq, valid_len, labels = batch
                seq, valid_len, labels = seq.to(self.device), valid_len.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                pred = self.model(seq, valid_len)
                loss = F.cross_entropy(pred, labels)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                train_acc += self.metrics(pred.argmax(1), labels, 'acc@1')
                    
            train_loss = train_loss / len(train_data)
            train_acc = train_acc / len(train_data)
            valid_acc = self.valid(valid_data)
            print('Epoch: {}, train_loss: {:.4f}, train_acc: {:.4f}, valid_mae: {:.4f}'.format(epoch, train_loss, train_acc, valid_acc))
            if valid_acc>best_acc:
                best_acc=valid_acc
                best_model=copy.deepcopy(self.model)
                patience=10
            else:
                patience-=1
                if patience==0:
                    break
        
        self.model=best_model


    def evaluate(self,train_data,valid_data,test_data):
        
        self.train(train_data,valid_data)
            
        self.model.eval()
        labels=[]
        preds=[]
        for batch in test_data:
            seq, valid_len, label = batch
            seq, valid_len, label = seq.to(self.device), valid_len.to(self.device), label.to(self.device)
            pred = self.model(seq, valid_len)
            labels.append(label)
            preds.append(pred)
        
        res={}
        for metric in self.metrics:
            res[metric]=self.metrics_fn(preds,labels,metric)

        return res