from torch import nn
from abc import ABC
import torch
from veccity.upstream.point.utils import weight_init
from torch.nn import functional as F
from logging import getLogger
import numpy as np
import copy
from sklearn.utils import shuffle
from veccity.upstream.point.utils import next_batch, create_src_trg, weight_init, top_n_accuracy


class TrajectoryPredictor(nn.Module, ABC):
    def __init__(self, num_slots, aux_embed_size, input_size, hidden_size, output_size, encoder_model,device):
        super().__init__()

        self.time_embed = nn.Embedding(num_slots + 1, aux_embed_size)
        self.dist_embed = nn.Embedding(num_slots + 1, aux_embed_size)
        
        self.out_linear = nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                        nn.ReLU(), nn.Linear(hidden_size, output_size))
        self.sos = nn.Parameter(torch.zeros(input_size + 2 * aux_embed_size).float(), requires_grad=True)
        self.aux_sos = nn.Parameter(torch.zeros(aux_embed_size * 2).float(), requires_grad=True)
        self.apply(weight_init)

        self.encoder_model = encoder_model
        

    def forward(self, full_seq, valid_len, **kwargs):
        batch_size = full_seq.size(0)

        time_delta = kwargs['time_delta'][:, 1:]
        dist = kwargs['dist'][:, 1:]

        time_slot_i = torch.floor(torch.clamp(time_delta, 0, self.time_thres) / self.time_thres * self.num_slots).long()
        dist_slot_i = torch.floor(
            torch.clamp(dist, 0, self.dist_thres) / self.dist_thres * self.num_slots).long()  # (batch, seq_len-1)
        aux_input = torch.cat([self.aux_sos.reshape(1, 1, -1).repeat(batch_size, 1, 1),
                               torch.cat([self.time_embed(time_slot_i),
                                          self.dist_embed(dist_slot_i)], dim=-1)],
                              dim=1)  # (batch, seq_len, aux_embed_size*2)

        full_embed = self.encoder_model.encode(full_seq,valid_len,**kwargs)  # (batch_size, seq_len, input_size)

        out = torch.cat([full_embed, aux_input], dim=-1)  # (batch_size, seq_len, input_size + aux_embed_size * 2)
        out = self.out_linear(out)
        return out

class TbSExecutor:
    def __init__(self,encoder_model,embed_size,hidden_size,output_size,learner='adam',lr=1e-3,max_epoch=100,device='cpu') -> None:

        self.embed_size=embed_size
        self.hidden_size=hidden_size
        self.device = device
        self.encoder_model =encoder_model
        self.learner = learner
        self.output_size=output_size
        self.lr=lr
        self.max_epoch = max_epoch
        self.model = TrajectoryPredictor(self.embed_size,self.hidden_size,self.output_size,self.encoder_model,self.device)
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
        for i, batch in enumerate(next_batch(valid_data, self.batch_size)):
            batch=self.collection(batch)
            pred = self.model(batch)
            label = batch['labels']
            labels.append(label)
            preds.append(pred)

        return self.metrics_fn(preds,labels,"acc@1")
    
    def train(self,train_data,valid_data):
        self.model.train()
            
        for epoch in range(self.max_epoch):
            train_loss = 0
            train_acc = 0
            best_acc = 0
            best_model=None
            patience=10
            self.model.train()
            for i, batch in enumerate(next_batch(shuffle(train_data), self.batch_size)):
                batch=self.collection(batch)
                self.optimizer.zero_grad()
                pred = self.model(batch)
                labels=batch['labels']
                loss = F.cross_entropy(pred, labels)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                train_acc += self.metrics(pred.argmax(1), labels, 'acc@1')
                    
            train_loss = train_loss / len(train_data)
            train_acc = train_acc / len(train_data)
            valid_acc = self.valid(valid_data)
            print('Epoch: {}, train_loss: {:.4f}, train_acc: {:.4f}, valid_acc: {:.4f}'.format(epoch, train_loss, train_acc, valid_acc))
            if valid_acc>best_acc:
                best_acc=valid_acc
                best_model=copy.deepcopy(self.model)
                patience=10
            else:
                patience-=1
                if patience==0:
                    break
        
        self.model=best_model

    def collection(self,batch):
        #user_index, full_seq, weekday, timestamp, length, time_delta, dist, lat, lng
        # 需要seq，weekday，timestamp，length，tdel，sdel
        user_index, full_seq, weekday, timestamp, length, time_delta, dist, lat, lng = zip(*batch)
        
        src_seq, trg_seq = create_src_trg(full_seq, 1, fill_value=self.output_size)

        src_t,_=create_src_trg(timestamp,1,0)
        src_time_delta,_ = create_src_trg(time_delta,1, 0)
        src_dist,_ = create_src_trg(dist,1, 0)
        src_lat,_ = create_src_trg(lat,1, 0)
        src_lng,_ = create_src_trg(lng,1, 0)

        src_week = create_src_trg(weekday,1,0)
        src_hour = (src_t % (24 * 60 * 60) / 60 / 60)
        src_duration = ((src_t[:, 1:] - src_t[:, :-1]) % (24 * 60 * 60) / 60 / 60)
        src_duration = torch.clamp(src_duration, 0, 23)
        res=torch.zeros([src_duration.size(0),1],dtype=torch.long)
        src_duration = torch.hstack([res,src_duration])

        src_seq, trg_seq,src_week,src_hour = (torch.from_numpy(item).long().to(self.device) for item in [src_seq, trg_seq,src_week,src_hour])
        src_t,src_time_delta,src_dist,src_lat, src_lng = (torch.from_numpy(item).float().to(self.device) for item in [src_t,src_time_delta,src_dist,src_lat, src_lng])

        batch={
            'seq':src_seq,
            'target':trg_seq,
            'time_delta':src_time_delta,
            'dist':src_dist,
            'time_slot':src_week,
            'time_hour':src_hour,
            'time_duration':src_duration,
            'lat':src_lat,
            'lng':src_lng
        }
        
        return batch

    def evaluate(self,train_data,valid_data,test_data):
        """
            train_data,valid_data,test_data是dataloader
        """
        
        self.train(train_data,valid_data)
            
        self.model.eval()
        labels=[]
        preds=[]
        for i, batch in enumerate(next_batch(test_data, self.batch_size)):
            batch=self.collection(batch)
            pred = self.model(batch)
            label=batch['labels']
            labels.append(label)
            preds.append(pred)
        
        res={}
        for metric in self.metrics:
            res[metric]=self.metrics_fn(preds,labels,metric)

        return res
    