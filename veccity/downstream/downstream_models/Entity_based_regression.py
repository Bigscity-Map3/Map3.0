import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import KFold


class EbRExecutor:
    def __init__(self,alpha,encoder_model, metrics=['rmse']):
        super(EbRExecutor,self).__init__()

        self.model = linear_model.Ridge(alpha=alpha)

        self.encoder_model = encoder_model
        self.metrics=metrics

    
    def _test_one_fit(self,X_train,y_train,X_test):
        
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        return y_pred
    
    def metrics_fn(self,y_pred,y_true,metric):
        if metric=='rmse':
            return np.sqrt(np.mean((y_pred-y_true)**2))
        elif metric=='mae':
            return np.mean(np.abs(y_pred-y_true))
        elif metric=='r2':
            return 1-np.sum((y_pred-y_true)**2)/np.sum((y_true-np.mean(y_true))**2)
        elif metric=='mape':
            return np.mean(np.abs((y_pred-y_true)/y_true))
    
    def evaluate(self,nodes,Y):
        kf = KFold(n_splits=5)
        X=self.encoder_model.encode(nodes)
        y_preds = []
        y_truths = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = Y[train_index], Y[test_index]
            y_pred = self._test_one_fit(X_train, y_train, X_test)
            y_preds.append(y_pred)
            y_truths.append(y_test)
        
        res={}
        for metric in self.metrics:
            res[metric]=self.metrics_fn(y_preds,y_truths,metric)
        
        return res
    
    
    


    
