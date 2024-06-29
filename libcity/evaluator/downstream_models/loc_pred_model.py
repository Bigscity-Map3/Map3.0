from abc import ABC

import numpy as np
import torch
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score, label_ranking_average_precision_score
from sklearn.utils import shuffle
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from logging import getLogger
from libcity.model.poi_representation.utils import next_batch, create_src_trg, weight_init, top_n_accuracy
from torch.nn.utils.rnn import pad_sequence,pad_packed_sequence,pack_padded_sequence
from libcity.evaluator.utils import accuracy
import torch.nn.functional as F


class LstmUserPredictor(nn.Module, ABC):
    def __init__(self, embed_layer, input_size, rnn_hidden_size, fc_hidden_size, output_size, num_layers,device):
        super().__init__()
        self.embed_layer = embed_layer
        self.add_module('embed_layer', self.embed_layer)
        self.hidden_size=rnn_hidden_size
        self.num_layers=num_layers
        self.device=device

        self.encoder = nn.LSTM(input_size, rnn_hidden_size, num_layers, dropout=0.1 if num_layers>1 else 0.0, batch_first=True)

        self.fc = nn.Sequential(nn.Tanh(), nn.Linear(rnn_hidden_size, fc_hidden_size),
                                        nn.LeakyReLU(), nn.Linear(fc_hidden_size, output_size))
    
        self.apply(weight_init)

    def forward(self, seq,valid_len,**kwargs):
        
        full_embed = self.embed_layer.encode(seq, **kwargs)
        pack_x = pack_padded_sequence(full_embed, lengths=valid_len,batch_first=True,enforce_sorted=False)

        h0 = torch.zeros(self.num_layers, full_embed.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, full_embed.size(0), self.hidden_size).to(self.device)

        out, _ = self.encoder(pack_x, (h0, c0))
        out, out_len = pad_packed_sequence(out, batch_first=True)

        out = torch.stack([out[i,ind-1,:] for i,ind in enumerate(valid_len)])
        
        pred = self.fc(out)
        return pred

def traj_user_classification(train_set, test_set, num_user, num_loc, clf_model, num_epoch, batch_size, device):
    def _create_src_trg(origin, fill_value):
            src, trg = create_src_trg(origin, 0, fill_value)
            return torch.from_numpy(src).float().to(device)

    logger = getLogger()
    logger.info('Start training downstream model [user_clf]...')
    clf_model = clf_model.to(device)
    optimizer = torch.optim.Adam(clf_model.parameters(), lr=1e-4)
    loss_func = nn.CrossEntropyLoss()

    def one_step(batch):
        user_index, full_seq, weekday, timestamp, length, time_delta, dist, lat, lng = zip(*batch)
        src_t = _create_src_trg(timestamp, 0)
        src_time_delta = _create_src_trg(time_delta, 0)
        src_dist = _create_src_trg(dist, 0)
        src_lat = _create_src_trg(lat, 0)
        src_lng = _create_src_trg(lng, 0)

        full_seq=[torch.tensor(seq,dtype=torch.long,device=device) for seq in full_seq]
        inputs = pad_sequence(full_seq,batch_first=True, padding_value=num_loc)
        targets = torch.tensor(user_index).long().to(device)
        length = list(length)
        # timestamp = torch.tensor(timestamp).long().to(device)

        out = clf_model(inputs,length,user_index=user_index, timestamp=src_t,
                        time_delta=src_time_delta, dist=src_dist, lat=src_lat, lng=src_lng)
        return out, targets

    score_log = []
    test_point = max(1, int(len(train_set) / batch_size / 2))
    logger.info('Test set size: {}'.format(len(test_set)))

    for epoch in range(num_epoch):
        for i, batch in enumerate(next_batch(shuffle(train_set), batch_size)):
            out, label = one_step(batch)
            loss = loss_func(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (i + 1) % test_point == 0:
                pres_raw, labels = [], []
                for test_batch in next_batch(test_set, batch_size * 4):
                    test_out, test_label = one_step(test_batch)
                    pres_raw.append(test_out.detach().cpu().numpy())
                    labels.append(test_label.detach().cpu().numpy())
                pres_raw, labels = np.concatenate(pres_raw), np.concatenate(labels)
                pres = pres_raw.argmax(-1)

                pre = precision_score(labels, pres, average='macro', zero_division=0.0)
                acc, recall = accuracy_score(labels, pres), recall_score(labels, pres, average='macro', zero_division=0.0)
                f1_micro, f1_macro = f1_score(labels, pres, average='micro'), f1_score(labels, pres, average='macro')
                score_log.append([acc, pre, recall, f1_micro, f1_macro])
                best_acc, best_pre, best_recall, best_f1_micro, best_f1_macro = np.max(score_log, axis=0)
                
        logger.info('epoch {} complete!'.format(epoch))
        logger.info('Acc %.6f, Pre %.6f, Recall %.6f, F1-micro %.6f, F1-macro %.6f' % (
                    best_acc, best_pre, best_recall, best_f1_micro, best_f1_macro))

    best_acc, best_pre, best_recall, best_f1_micro, best_f1_macro = np.max(score_log, axis=0)
    logger.info('Finished Evaluation.')
    logger.info(
        'Acc %.6f, Pre %.6f, Recall %.6f, F1-micro %.6f, F1-macro %.6f' % (
            best_acc, best_pre, best_recall, best_f1_micro, best_f1_macro))
    return best_acc, best_pre, best_recall, best_f1_micro, best_f1_macro

def seq2seq_forward(encoder, lstm_input, valid_len, pre_len):
    his_len = valid_len - pre_len
    src_padded_embed = pack_padded_sequence(lstm_input, his_len, batch_first=True, enforce_sorted=False)
    out, hc = encoder(src_padded_embed)
    out,out_len=pad_packed_sequence(out,batch_first=True)
    return out,out_len


class TrajectoryPredictor(nn.Module, ABC):
    def __init__(self, embed_layer, num_slots, aux_embed_size, time_thres, dist_thres,
                 input_size, lstm_hidden_size, fc_hidden_size, output_size, num_layers, seq2seq=True):
        super().__init__()
        self.__dict__.update(locals())

        self.time_embed = nn.Embedding(num_slots + 1, aux_embed_size)
        self.dist_embed = nn.Embedding(num_slots + 1, aux_embed_size)

        self.encoder = nn.LSTM(input_size + 2 * aux_embed_size, lstm_hidden_size, num_layers, dropout=0.3,
                               batch_first=True)
        self.ln = nn.LayerNorm(lstm_hidden_size)
        self.out_linear = nn.Sequential(nn.Tanh(), nn.Linear(lstm_hidden_size, fc_hidden_size),
                                        nn.Tanh(), nn.Linear(fc_hidden_size, output_size))
        self.sos = nn.Parameter(torch.zeros(input_size + 2 * aux_embed_size).float(), requires_grad=True)
        self.aux_sos = nn.Parameter(torch.zeros(aux_embed_size * 2).float(), requires_grad=True)
        self.apply(weight_init)

        self.embed_layer = embed_layer
        try:
            self.add_module('embed_layer', self.embed_layer)
        except Exception:
            pass
        

    def forward(self, full_seq, valid_len, pre_len, **kwargs):
        batch_size = full_seq.size(0)
        # his_len = valid_len - pre_len

        time_delta = kwargs['time_delta'][:, 1:]
        dist = kwargs['dist'][:, 1:]

        time_slot_i = torch.floor(torch.clamp(time_delta, 0, self.time_thres) / self.time_thres * self.num_slots).long()
        dist_slot_i = torch.floor(
            torch.clamp(dist, 0, self.dist_thres) / self.dist_thres * self.num_slots).long()  # (batch, seq_len-1)
        aux_input = torch.cat([self.aux_sos.reshape(1, 1, -1).repeat(batch_size, 1, 1),
                               torch.cat([self.time_embed(time_slot_i),
                                          self.dist_embed(dist_slot_i)], dim=-1)],
                              dim=1)  # (batch, seq_len, aux_embed_size*2)

        full_embed = self.embed_layer.encode(full_seq,**kwargs)  # (batch_size, seq_len, input_size)
    

        lstm_input = torch.cat([full_embed, aux_input],
                               dim=-1)  # (batch_size, seq_len, input_size + aux_embed_size * 2)

        if self.seq2seq:
            lstm_out_pre,out_len = seq2seq_forward(self.encoder, lstm_input, valid_len, pre_len)
        else:
            lstm_out_pre = rnn_forward(self.encoder, self.sos, lstm_input, valid_len, pre_len)

        lstm_out_pre=self.ln(lstm_out_pre)
        out = self.out_linear(lstm_out_pre)
        return out


def loc_prediction(train_set, test_set, num_loc, pre_model, pre_len, num_epoch, batch_size,device):
    def one_step(batch):
        def _create_src_trg(origin, fill_value):
            src, trg = create_src_trg(origin, pre_len, fill_value)
            return torch.from_numpy(src).float().to(device)


        user_index, full_seq, weekday, timestamp, length, time_delta, dist, lat, lng = zip(*batch)
        index_matrix=torch.zeros([len(length),max(length)-1],dtype=torch.bool)
        for i in range(len(length)):
            index_matrix[i][:length[i]-1]=~index_matrix[i][:length[i]-1]
        index_matrix=index_matrix.to(device)
        user_index, length = (torch.tensor(item).long().to(device) for item in (user_index, length))

        src_seq, trg_seq = create_src_trg(full_seq, pre_len, fill_value=num_loc)

        src_seq, trg_seq = (torch.from_numpy(item).long().to(device) for item in [src_seq, trg_seq])

        src_t = _create_src_trg(timestamp, 0)
         
        src_time_delta = _create_src_trg(time_delta, 0)
        src_dist = _create_src_trg(dist, 0)
        src_lat = _create_src_trg(lat, 0)
        src_lng = _create_src_trg(lng, 0)

        src_week = _create_src_trg(weekday,0).long()
        src_hour = (src_t % (24 * 60 * 60) / 60 / 60).long()
        src_duration = ((src_t[:, 1:] - src_t[:, :-1]) % (24 * 60 * 60) / 60 / 60).long()
        src_duration = torch.clamp(src_duration, 0, 23)
        res=torch.zeros([src_duration.size(0),1],dtype=torch.long).to(device)
        src_duration = torch.hstack([res,src_duration])

        out = pre_model(src_seq, length, pre_len, user_index=user_index, timestamp=src_t,
                        time_delta=src_time_delta, dist=src_dist, lat=src_lat, lng=src_lng,
                        week=src_week,hour=src_hour,duration=src_duration)

        # out = out.reshape(-1, pre_model.output_size)
        out = out[index_matrix]

        label = trg_seq[index_matrix]
        return out, label


    logger = getLogger()
    logger.info('Start training downstream model [next_loc]...')
    pre_model = pre_model.to(device)
    
    optimizer = torch.optim.Adam(pre_model.parameters(), lr=1e-3)

    loss_func = nn.CrossEntropyLoss()

    score_log = []
    test_point = max(1, int(len(train_set) / batch_size / 2))
    logger.info('Test set size: {}'.format(len(test_set)))
    for epoch in range(num_epoch):
        losses=[]
        for i, batch in enumerate(next_batch(shuffle(train_set), batch_size)):
            out, label = one_step(batch)
            loss = loss_func(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            losses.append(loss.item())
            if (i + 1) % test_point == 0:

                pres_raw, labels = [], []
                for test_batch in next_batch(test_set, batch_size * 4):
                    test_out, test_label = one_step(test_batch)
                    pres_raw.append(test_out.detach().cpu())
                    labels.append(test_label.detach().cpu())
                pres_raw, labels = torch.vstack(pres_raw), torch.hstack(labels)
                pres = pres_raw.argmax(-1)

                # pre = precision_score(labels, pres, average='macro', zero_division=0.0)
                # acc, recall = accuracy_score(labels, pres), recall_score(labels, pres, average='macro', zero_division=0.0)
                acc1,acc5=accuracy(pres_raw,labels,topk=(1,5)) 
                # mrr=label_ranking_average_precision_score(F.one_hot(labels,num_classes=pres_raw.shape[-1]),pres_raw)
                f1_micro, f1_macro = f1_score(labels.numpy(), pres.numpy(), average='micro'), f1_score(labels.numpy(), pres.numpy(), average='macro')
                score_log.append([acc1, acc5, f1_micro, f1_macro])
                logger.info('Acc@1 %.6f, Acc@5 %.6f, F1-micro %.6f, F1-macro %.6f' % (
                acc1, acc5, f1_micro, f1_macro))
                best_acc1, best_acc5, best_f1_micro, best_f1_macro = np.max(score_log, axis=0)
                
        logger.info('epoch {} complete! avg loss:{}'.format(epoch,np.mean(losses)))
        # logger.info('Best Acc %.6f, Pre %.6f, Recall %.6f, F1-micro %.6f, F1-macro %.6f' % (
        #         best_acc, best_pre, best_recall, best_f1_micro, best_f1_macro))

    best_acc1, best_acc5, best_f1_micro, best_f1_macro = np.max(score_log, axis=0)
    logger.info('Finished Evaluation.')
    logger.info(
        'Acc1 %.6f, Acc5 %.6f, F1-micro %.6f, F1-macro %.6f' % (
            best_acc1, best_acc5, best_f1_micro, best_f1_macro))
    return best_acc1, best_acc5, best_f1_micro, best_f1_macro
