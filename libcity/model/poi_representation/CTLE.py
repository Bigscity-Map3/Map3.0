from collections import Counter
from torch.nn import init
import math
from itertools import zip_longest

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.utils import shuffle
from libcity.model.abstract_traffic_tradition_model import AbstractTraditionModel
from libcity.model.utils import next_batch


class CTLE(AbstractTraditionModel):
    def __init__(self, config, data_feature):
        self.config = config
        super().__init__(config, data_feature)
        self.model = config.get('model', '')
        self.dataset = config.get('dataset', '')
        self.exp_id = config.get('exp_id', None)
        self.output_dim = config.get('output_dim',128)
        self.embedding_path = './libcity/cache/{}/evaluate_cache/embedding_{}_{}_{}.npy' \
            .format(self.exp_id, self.model, self.dataset, self.output_dim)

        self.num_loc = self.data_feature.get("num_loc")
        self.user_num = self.data_feature.get("user_num")

        self.max_seq_len = self.data_feature.get('max_seq_len')
        self.user_ids = self.data_feature.get("user_ids")
        self.src_tokens = self.data_feature.get("src_tokens")
        self.src_weekdays = self.data_feature.get("src_weekdays")
        self.src_ts = self.data_feature.get("src_ts")
        self.src_lens = self.data_feature.get("src_lens")

        self.encoding_type = self.config.get('encoding_type', 'positional')
        self.ctle_num_layers = self.config.get('ctle_num_layers', 4)
        self.ctle_num_heads = self.config.get('ctle_num_heads', 8)
        self.ctle_mask_prop = self.config.get('ctle_mask_prop', 0.2)
        self.ctle_detach = self.config.get('ctle_detach', False)
        self.ctle_objective = self.config.get('ctle_objective', "mlm")
        self.init_param = self.config.get('init_param',False)


        self.embed_size = self.config.get("embed_size", 128)
        self.hidden_size = 4 * self.embed_size
        self.num_epochs = self.config.get('num_epoch', 5)
        self.batch_size = self.config.get('batch_size', 64)
        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')





    def run(self):
        encoding_layer = PositionalEncoding(self.embed_size, self.max_seq_len)
        if self.encoding_type == 'temporal':
            encoding_layer = TemporalEncoding(self.embed_size)

        obj_models = [MaskedLM(self.embed_size, self.num_loc)]
        if self.ctle_objective == "mh":
            obj_models.append(MaskedHour(self.embed_size))
        obj_models = nn.ModuleList(obj_models)
        ctle_embedding = CTLEEmbedding(encoding_layer, self.embed_size, self.num_loc)
        ctle_model = CTLEModel(ctle_embedding, self.hidden_size, num_layers=self.ctle_num_layers,
                               num_heads=self.ctle_num_heads,
                               init_param=self.init_param, detach=self.ctle_detach)

        model = ctle_model.to(self.device)
        init_lr = 1e-3
        optimizer = torch.optim.SGD(model.parameters(), lr=init_lr)

        ctle_model = ctle_model.to(self.device)
        obj_models = obj_models.to(self.device)

        user_ids, src_tokens, src_weekdays, src_ts, src_lens = \
            self.user_ids, self.src_tokens, self.src_weekdays, self.src_ts, self.src_lens

        optimizer = torch.optim.Adam(list(ctle_model.parameters()) + list(obj_models.parameters()), lr=1e-4)
        for epoch in range(self.num_epochs):
            for batch in next_batch(shuffle(list(zip(src_tokens, src_weekdays, src_ts, src_lens))),
                                    batch_size=self.batch_size):
                # Value filled with num_loc stands for masked tokens that shouldn't be considered.
                src_batch, _, src_t_batch, src_len_batch = zip(*batch)
                src_batch = np.transpose(np.array(list(zip_longest(*src_batch, fillvalue=ctle_model.num_vocab))))
                src_t_batch = np.transpose(np.array(list(zip_longest(*src_t_batch, fillvalue=0))))

                src_batch = torch.tensor(src_batch).long().to(self.device)
                src_t_batch = torch.tensor(src_t_batch).float().to(self.device)
                hour_batch = (src_t_batch % (24 * 60 * 60) / 60 / 60).long()

                batch_len, src_len = src_batch.size(0), src_batch.size(1)
                src_valid_len = torch.tensor(src_len_batch).long().to(self.device)

                mask_index = gen_random_mask(src_valid_len, src_len, mask_prop=self.ctle_mask_prop)

                src_batch = src_batch.reshape(-1)
                hour_batch = hour_batch.reshape(-1)
                origin_tokens = src_batch[mask_index]  # (num_masked)
                origin_hour = hour_batch[mask_index]

                # Value filled with num_loc+1 stands for special token <mask>.
                masked_tokens = src_batch.index_fill(0, mask_index, ctle_model.embed.num_vocab + 1).reshape(batch_len,
                                                                                                            -1)  # (batch_size, src_len)

                ctle_out = ctle_model(masked_tokens, timestamp=src_t_batch)  # (batch_size, src_len, embed_size)
                masked_out = ctle_out.reshape(-1, ctle_model.embed_size)[mask_index]  # (num_masked, embed_size)
                loss = 0.
                for obj_model in obj_models:
                    loss += obj_model(masked_out, origin_tokens=origin_tokens, origin_hour=origin_hour)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        np.save(self.embedding_path, ctle_model.static_embed())
        return ctle_model.static_embed()


def gen_random_mask(src_valid_lens, src_len, mask_prop):
    """
    @param src_valid_lens: valid length of sequence, shape (batch_size)
    """
    # all_index = np.arange((batch_size * src_len)).reshape(batch_size, src_len)
    # all_index = shuffle_along_axis(all_index, axis=1)
    # mask_count = math.ceil(mask_prop * src_len)
    # masked_index = all_index[:, :mask_count].reshape(-1)
    # return masked_index
    index_list = []
    for batch, l in enumerate(src_valid_lens):
        mask_count = torch.ceil(mask_prop * l).int()
        masked_index = torch.randperm(l)[:mask_count]
        masked_index += src_len * batch
        index_list.append(masked_index)
    return torch.cat(index_list).long().to(src_valid_lens.device)


def gen_casual_mask(seq_len, include_self=True):
    """
    Generate a casual mask which prevents i-th output element from
    depending on any input elements from "the future".
    Note that for PyTorch Transformer model, sequence mask should be
    filled with -inf for the masked positions, and 0.0 else.

    :param seq_len: length of sequence.
    :return: a casual mask, shape (seq_len, seq_len)
    """
    if include_self:
        mask = 1 - torch.triu(torch.ones(seq_len, seq_len)).transpose(0, 1)
    else:
        mask = 1 - torch.tril(torch.ones(seq_len, seq_len)).transpose(0, 1)
    return mask.bool()


class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len):
        super().__init__()
        pe = torch.zeros(max_len, embed_size).float()
        pe.requires_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, embed_size, 2).float() * -(math.log(10000.0) / embed_size)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, **kwargs):
        return self.pe[:, :x.size(1)]


class TemporalEncoding(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.omega = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, embed_size))).float(), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(embed_size).float(), requires_grad=True)
        self.div_term = math.sqrt(1. / embed_size)

    def forward(self, x, **kwargs):
        timestamp = kwargs['timestamp']  # (batch, seq_len)
        time_encode = timestamp.unsqueeze(-1) * self.omega.reshape(1, 1, -1) + self.bias.reshape(1, 1, -1)
        time_encode = torch.cos(time_encode)
        return self.div_term * time_encode


class CTLEEmbedding(nn.Module):
    def __init__(self, encoding_layer, embed_size, num_vocab):
        super().__init__()
        self.embed_size = embed_size
        self.num_vocab = num_vocab
        self.encoding_layer = encoding_layer
        self.add_module('encoding', self.encoding_layer)

        self.token_embed = nn.Embedding(num_vocab+2, embed_size, padding_idx=num_vocab)
        # self.token_embed.weight.data.uniform_(-0.5/embed_size, 0.5/embed_size)

    def forward(self, x, **kwargs):
        token_embed = self.token_embed(x)
        pos_embed = self.encoding_layer(x, **kwargs)
        return token_embed + pos_embed


class CTLEModel(nn.Module):
    def __init__(self, embed, hidden_size, num_layers, num_heads, init_param=False, detach=True):
        super().__init__()
        self.embed_size = embed.embed_size
        self.num_vocab = embed.num_vocab

        self.embed = embed
        self.add_module('embed', embed)

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embed_size, nhead=num_heads,
                                                   dim_feedforward=hidden_size, dropout=0.1)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers,
                                             norm=nn.LayerNorm(self.embed_size, eps=1e-6))
        self.detach = detach
        if init_param:
            self.apply(weight_init)

    def forward(self, x, **kwargs):
        """
        @param x: sequence of tokens, shape (batch, seq_len).
        """
        seq_len = x.size(1)
        downstream = kwargs.get('downstream', False)

        src_key_padding_mask = (x == self.num_vocab)
        token_embed = self.embed(x, **kwargs)  # (batch_size, seq_len, embed_size)
        if downstream:
            pre_len = kwargs['pre_len']
            src_mask = torch.ones(seq_len, seq_len).bool()
            src_mask[:, :-pre_len] = False
            src_mask[-pre_len:, -pre_len:] = gen_casual_mask(pre_len)
            src_mask = torch.zeros(seq_len, seq_len).masked_fill(src_mask, float('-inf'))
            src_mask = src_mask.to(x.device)
        else:
            src_mask = None

        encoder_out = self.encoder(token_embed.transpose(0, 1), mask=src_mask,
                                   src_key_padding_mask=src_key_padding_mask).transpose(0, 1)  # (batch_size, src_len, embed_size)
        if self.detach and downstream:
            encoder_out = encoder_out.detach()
        return encoder_out

    def static_embed(self):
        return self.embed.token_embed.weight[:self.num_vocab].detach().cpu().numpy()


class MaskedLM(nn.Module):
    def __init__(self, input_size, vocab_size):
        super().__init__()
        self.linear = nn.Linear(input_size, vocab_size)
        self.dropout = nn.Dropout(0.1)
        self.loss_func = nn.CrossEntropyLoss()

        self.vocab_size = vocab_size

    def forward(self, x, **kwargs):
        """
        :param x: input sequence (batch, seq_len, embed_size).
        :param origin_tokens: original tokens, shape (batch, seq_len)
        :return: the loss value of MLM objective.
        """
        origin_tokens = kwargs['origin_tokens']
        origin_tokens = origin_tokens.reshape(-1)
        lm_pre = self.linear(self.dropout(x))  # (batch, seq_len, vocab_size)
        lm_pre = lm_pre.reshape(-1, self.vocab_size)  # (batch * seq_len, vocab_size)
        return self.loss_func(lm_pre, origin_tokens)


class MaskedHour(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size, 24)
        self.dropout = nn.Dropout(0.1)
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, x, **kwargs):
        """
        @param x: input sequence (batch, seq_len, embed_size)
        @param original_hour: original hour indices, shape (batch, seq_len)
        @returns: the loss value of MH objective.
        """
        origin_hour = kwargs['origin_hour']
        origin_hour = origin_hour.reshape(-1)
        hour_pre = self.linear(self.dropout(x))
        hour_pre = hour_pre.reshape(-1, 24)
        return self.loss_func(hour_pre, origin_hour)


class MaskedWeekday(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.linear = nn.Linear(input_size, 7)
        self.dropout = nn.Dropout(0.1)
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, x, **kwargs):
        """
        @param x: input sequence (batch, seq_len, embed_size)
        @param original_hour: original hour indices, shape (batch, seq_len)
        @returns: the loss value of MH objective.
        """
        origin_weekday = kwargs['origin_weekday']
        origin_weekday = origin_weekday.reshape(-1)
        weekday_pre = self.linear(self.dropout(x))
        weekday_pre = weekday_pre.reshape(-1, 7)
        return self.loss_func(weekday_pre, origin_weekday)


def weight_init(m):
    """
    Usage:
        model = Model()
        model.apply(weight_init)
    """
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.Embedding):
        embed_size = m.weight.size(-1)
        if embed_size > 0:
            init_range = 0.5/m.weight.size(-1)
            init.uniform_(m.weight.data, -init_range, init_range)