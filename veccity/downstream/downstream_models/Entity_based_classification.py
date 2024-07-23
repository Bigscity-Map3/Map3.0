import torch
from torch import nn
from logging import getLogger
from sklearn.model_selection import  StratifiedKFold
import torch.utils
import torch.utils.data
import copy
import numpy as np


class Entity_based_classification(nn.Module):
    def __init__(self,embed_size,hidden_size,encoder_model,num_class,device):
        super(Entity_based_classification,self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_class = num_class
        self.device = device

        self.model = nn.Sequential(
            nn.Linear(self.embed_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.num_class)
        ).to(self.device)

        self.loss_fn = nn.CrossEntropyLoss()

        self.encoder_model = encoder_model
    
    def forward(self,nodes):
        return self.model(self.encoder_model.encode(nodes))
    
    def calculate_loss(self,nodes,labels):
        return self.loss_fn(self.forward(nodes),labels)
    
    @torch.no_grad()
    def evaluate(self,nodes,labels):
        return torch.argmax(self.forward(nodes),dim=1), labels

class EbCExecutor:
    def __init__(self,encoder_model,embed_size,hidden_size,num_class,learner='adam',lr=1e-3,max_epoch=100,device='cpu') -> None:

        self.embed_size=embed_size
        self.hidden_size=hidden_size
        self.num_class = num_class
        self.device = device
        self.encoder_model =encoder_model
        self.learner = learner
        self.lr=lr
        self.max_epoch = max_epoch
        self.model = Entity_based_classification(self.embed_size,self.hidden_size,self.encoder_model,self.num_class,self.device)
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

    def metrics_fn(self, preds, labels, metric_name):
        if metric_name=='acc@1':
            return (preds == labels).sum() / len(labels)
        elif metric_name=='acc@3':
            return (preds[:3] == labels).sum() / len(labels)
        elif metric_name=='acc@5':
            return (preds[:5] == labels).sum() / len(labels)
        elif metric_name=='F1-macro':
            return (preds == labels).sum() / len(labels)
        elif metric_name=='F1-micro':
            return (preds == labels).sum() / len(labels)
        

    def evaluate(self,nodes,labels):
        # kfold test
        skf=StratifiedKFold(n_splits=5)
        res_preds=[]
        res_labels=[]
        indices=len(nodes)
        np.random.shuffle(indices)
        nodes=nodes[indices]
        labels=labels[indices]
        for i,(train_ind,valid_ind) in enumerate(skf.split(nodes,labels)):
            
            input=torch.tensor(nodes[train_ind],dtype=torch.long,device=self.device)
            label=torch.tensor(labels[train_ind],dtype=torch.long,device=self.device)
            data=torch.utils.data.TensorDataset(input,label)
            dataloder=torch.utils.data.DataLoader(data,batch_size=1024)


            best_loss=99999
            best_model=None
            patience=10
            for epoch in range(self.max_epoch):
                losses=[]
                for _,batch in enumerate(dataloder):
                    bacth_input = batch[0]
                    batch_label = batch[1]
                    self.optimizer.zero_grad()
                    loss=self.model.calculate_loss(bacth_input,batch_label)
                    loss.backward()
                    self.optimizer.step()
                    losses.append(loss.item())
                
                if sum(losses)/len(losses)<best_loss:
                    best_loss=sum(losses)/len(losses)
                    best_model=copy.deepcopy(self.model)
                    patience=10
                else:
                    patience-=1
                    if patience==0:
                        break

            self.model=best_model
            
            eval_inputs=torch.tensor(nodes[valid_ind],dtype=torch.long,device=self.device)
            eval_labels=torch.tensor(labels[valid_ind],dtype=torch.long,device=self.device)
            eval_data=torch.utils.data.TensorDataset(eval_inputs,eval_labels)
            eval_dataloder=torch.utils.data.DataLoader(eval_data,batch_size=1024)

            for _,batch in enumerate(eval_dataloder):
                bacth_input = batch[0]
                batch_label = batch[1]
                pred, label=self.model.evaluate(bacth_input,batch_label)
                res_preds.append(pred)
                res_labels.append(label)
        
        res={}
        for metric in self.metrics:
            res[metric]=self.metrics_fn(res_preds,res_labels,metric)

        return res
        


    
