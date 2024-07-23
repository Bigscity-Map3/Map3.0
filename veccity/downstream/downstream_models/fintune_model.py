import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence

class Static_model(nn.Module):
    def __init__(self, embeddings, input_size, hidden_size):
        super(Static_model, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True) 
        embed_size=len(embeddings)+1
        embeddings=torch.cat((embeddings,torch.zeros(1,embed_size))) # add unk
        self.embedding = nn.Embedding(embed_size+1, hidden_size)
        self.embedding.weight.data.copy_(embeddings)
        self.embedding.weight.requires_grad = False
    
    def encode_token(self, x):
        return self.embedding(x)
    
    def encode_seq(self,seq,valid_len):
        seq=self.encode_token(seq)
        packed = nn.utils.rnn.pack_padded_sequence(seq, valid_len, batch_first=True, enforce_sorted=False)
        output, _ = self.lstm(packed)
        output, _ = pad_packed_sequence(output, batch_first=True)
        return output

class Sequence_model(nn.Module):
    def __init__(self, embedding_model, ft=True):
        super(Sequence_model, self).__init__()
        self.embedding = embedding_model
        self.embedding.add_unk()
        self.ft=ft
    
    def encode_token(self, x):
        if self.ft:
            return self.embedding.encode_token(x)
        else:
            with torch.no_grad():
                return self.embedding.encode_token(x)
    
    def encode_seq(self,seq,valid_len,**kwargs):
        if self.ft:
            return self.embedding.encode_seq(seq,valid_len,**kwargs)
        else:
            with torch.no_grad():
                return self.embedding.encode_seq(seq,valid_len,**kwargs)




