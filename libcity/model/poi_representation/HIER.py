from collections import Counter
from torch.nn import init
import math
from itertools import zip_longest

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from sklearn.utils import shuffle
from libcity.model.abstract_traffic_tradition_model import AbstractTraditionModel


def next_batch(data, batch_size):
    data_length = len(data)
    num_batches = math.ceil(data_length / batch_size)
    for batch_index in range(num_batches):
        start_index = batch_index * batch_size
        end_index = min((batch_index + 1) * batch_size, data_length)
        yield data[start_index:end_index]

class HIEREmbedding(nn.Module):
    def __init__(self, token_embed_size, num_vocab, week_embed_size, hour_embed_size, duration_embed_size):
        super().__init__()
        self.num_vocab = num_vocab
        self.token_embed_size = token_embed_size
        self.embed_size = token_embed_size + week_embed_size + hour_embed_size + duration_embed_size

        self.token_embed = nn.Embedding(num_vocab, token_embed_size)
        self.token_embed.weight.data.uniform_(-0.5/token_embed_size, 0.5/token_embed_size)
        self.week_embed = nn.Embedding(7, week_embed_size)
        self.hour_embed = nn.Embedding(24, hour_embed_size)
        self.duration_embed = nn.Embedding(24, duration_embed_size)

        self.dropout = nn.Dropout(0.1)

    def forward(self, token, week, hour, duration):
        token = self.token_embed(token)
        week = self.week_embed(week)
        hour = self.hour_embed(hour)
        duration = self.duration_embed(duration)

        return self.dropout(torch.cat([token, week, hour, duration], dim=-1))

class HIERModel(nn.Module):
    def __init__(self, embed: HIEREmbedding, hidden_size, num_layers, share=True, dropout=0.1):
        super().__init__()
        self.embed = embed
        self.add_module('embed', self.embed)
        self.encoder = nn.LSTM(self.embed.embed_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        if share:
            self.out_linear = nn.Sequential(nn.Linear(hidden_size, self.embed.token_embed_size), nn.LeakyReLU())
        else:
            self.out_linear = nn.Sequential(nn.Linear(hidden_size, self.embed.token_embed_size),
                                            nn.LeakyReLU(),
                                            nn.Linear(self.embed.token_embed_size, self.embed.num_vocab))
        self.share = share

    def forward(self, token, week, hour, duration, valid_len, **kwargs):
        """
        :param token: sequences of tokens, shape (batch, seq_len)
        :param week: sequences of week indices, shape (batch, seq_len)
        :param hour: sequences of visit time slot indices, shape (batch, seq_len)
        :param duration: sequences of duration slot indices, shape (batch, seq_len)
        :return: the output prediction of next vocab, shape (batch, seq_len, num_vocab)
        """
        embed = self.embed(token, week, hour, duration)  # (batch, seq_len, embed_size)
        packed_embed = pack_padded_sequence(embed, valid_len, batch_first=True, enforce_sorted=False)
        encoder_out, hc = self.encoder(packed_embed)  # (batch, seq_len, hidden_size)
        out = self.out_linear(encoder_out.data)  # (batch, seq_len, token_embed_size)

        if self.share:
            out = torch.matmul(out, self.embed.token_embed.weight.transpose(0, 1))  # (total_valid_len, num_vocab)
        return out

    def static_embed(self):
        return self.embed.token_embed.weight[:self.embed.num_vocab].detach().cpu().numpy()

class HIER(AbstractTraditionModel):
    def __init__(self, config, data_feature):
        self.config = config
        super().__init__(config, data_feature)
        self.num_loc = self.data_feature.get("num_loc")

        self.hier_num_layers = self.config.get('hier_num_layer', 3)
        self.hier_week_embed_size = self.config.get('hier_week_embed_size', 4)
        self.hier_hour_embed_size = self.config.get('hier_hour_embed_size', 4)
        self.hier_duration_embed_size = self.config.get('hier_duration_embed_size', 4)
        self.hier_share = self.config.get('teaser_week_embed_size', False)

        self.user_ids = self.data_feature.get("user_ids")
        self.src_tokens=self.data_feature.get("src_tokens")
        self.src_weekdays=self.data_feature.get("src_weekdays")
        self.src_ts = self.data_feature.get("src_ts")
        self.src_lens = self.data_feature.get("src_lens")

        self.embed_size = self.config.get("embed_size", 128)
        self.hidden_size = 4 * self.embed_size
        self.num_epochs = self.config.get('num_epoch', 5)
        self.batch_size = self.config.get('batch_size', 64)
        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

        self.init_lr = 1e-3

    def run(self):
        hier_embedding = HIEREmbedding(self.embed_size, self.num_loc,
                                   self.hier_week_embed_size, self.hier_hour_embed_size, self.hier_duration_embed_size)
        model = HIERModel(hier_embedding, self.hidden_size, self.hier_num_layers, share=self.hier_share)
        model = model.to(self.device)

        user_ids, src_tokens, src_weekdays, src_ts, src_lens = \
            self.user_ids, self.src_tokens, self.src_weekdays, self.src_ts, self.src_lens

        loss_func = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters())
        for epoch in range(self.num_epochs):
            for batch in next_batch(shuffle(list(zip(src_tokens, src_weekdays, src_ts, src_lens))), batch_size=self.batch_size):
                src_token, src_weekday, src_t, src_len = zip(*batch)
                src_token, src_weekday = [
                    torch.from_numpy(np.transpose(np.array(list(zip_longest(*item, fillvalue=0))))).long().to(self.device)
                    for item in (src_token, src_weekday)]
                src_t = torch.from_numpy(np.transpose(np.array(list(zip_longest(*src_t, fillvalue=0))))).float().to(self.device)
                src_len = torch.tensor(src_len).int()

                src_hour = (src_t % (24 * 60 * 60) / 60 / 60).long()
                src_duration = ((src_t[:, 1:] - src_t[:, :-1]) % (24 * 60 * 60) / 60 / 60).long()
                src_duration = torch.clamp(src_duration, 0, 23)

                hier_out = model(token=src_token[:, :-1], week=src_weekday[:, :-1], hour=src_hour[:, :-1],
                                      duration=src_duration, valid_len=src_len.to('cpu') - 1)  # (batch, seq_len, num_vocab)
                trg_token = pack_padded_sequence(src_token[:, 1:], src_len - 1, batch_first=True, enforce_sorted=False).data
                loss = loss_func(hier_out, trg_token)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        return model.static_embed()




