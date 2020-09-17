import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import reset_params

class LinearLayer(nn.Module):
    ''' Linear '''
    def __init__(self, opt):
        super(LinearLayer, self).__init__()
        OD = opt.hidden_dim * 2 if opt.bidirectional else opt.hidden_dim
        self.dropout = nn.Dropout(opt.dropout)
        self.output_linear = nn.Linear(OD, 3)
        reset_params(self.output_linear, opt.initializer)
    
    def forward(self, inputs, rnn_input, rnn_output):
        output_vector = self.output_linear(self.dropout(rnn_output))
        return output_vector
    

class MeanPooling(nn.Module):
    ''' Mean Pooling '''
    def __init__(self, opt):
        super(MeanPooling, self).__init__()
        OD = opt.hidden_dim * 2 if opt.bidirectional else opt.hidden_dim
        self.dropout = nn.Dropout(opt.dropout)
        self.output_linear = nn.Linear(OD, 3)
        reset_params(self.output_linear, opt.initializer)
    
    def forward(self, inputs, rnn_input, rnn_output):
        text = inputs[0]
        text_len = torch.sum(text!=0, dim=-1)
        output_vector = rnn_output.sum(1).div(text_len.float().unsqueeze(-1))
        output_vector = self.output_linear(self.dropout(output_vector))
        return output_vector
    

class Attention(nn.Module):
    ''' Standard attention mechanism with bilinear kernel '''
    def __init__(self, opt):
        super(Attention, self).__init__()
        WD = opt.word_dim
        OD = opt.hidden_dim * 2 if opt.bidirectional else opt.hidden_dim
        self.dropout = nn.Dropout(opt.dropout)
        self.projection = nn.Linear(OD, WD)
        self.output_linear = nn.Linear(OD, 3)
        reset_params(self.projection, opt.initializer)
        reset_params(self.output_linear, opt.initializer)
        
    def forward(self, inputs, rnn_input, rnn_output):
        text, aspect, aspect_mask = inputs[0], inputs[1], inputs[2]
        BS, SL, OD = rnn_output.shape
        text_mask = torch.unsqueeze(text!=0, dim=-1)
        aspect_len = torch.sum(aspect!=0, dim=-1)
        aspect_mask = aspect_mask.unsqueeze(-1)
        aspect_feature = torch.where(aspect_mask!=0, rnn_input, torch.zeros_like(rnn_input))
        aspect_pool = aspect_feature.sum(1).div(aspect_len.float().unsqueeze(-1))
        attention_weight = torch.tanh(torch.bmm(self.projection(rnn_output), aspect_pool.unsqueeze(-1)))
        attention_weight = torch.where(text_mask!=0, attention_weight, torch.zeros_like(attention_weight)-1e10)
        attention_weight = F.softmax(attention_weight, dim=1).transpose(1, 2)
        output_vector = torch.bmm(attention_weight, rnn_output).squeeze(1)
        output_vector = self.output_linear(self.dropout(output_vector))
        return output_vector
    

class DynamicLSTM(nn.Module):
    ''' LSTM which can hold variable length sequence, use like TensorFlow's RNN(input, length...). '''
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=True, dropout=0,
                 bidirectional=False, only_use_last_hidden_state=False, rnn_type='LSTM'):
        super(DynamicLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.only_use_last_hidden_state = only_use_last_hidden_state
        self.rnn_type = rnn_type
        
        if self.rnn_type == 'LSTM':
            self.RNN = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                               bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        elif self.rnn_type == 'GRU':
            self.RNN = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                              bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        elif self.rnn_type == 'RNN':
            self.RNN = nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                              bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
    
    def forward(self, x, x_len):
        '''
        sequence -> sort -> pad and pack -> process using RNN -> unpack -> unsort
        '''
        total_length = x.size(1)
        ''' sort '''
        x_sort_idx = torch.sort(x_len, descending=True)[1].long()
        x_unsort_idx = torch.sort(x_sort_idx)[1].long()
        x_len = x_len[x_sort_idx]
        x = x[x_sort_idx]
        ''' pack '''
        x_emb_p = torch.nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=self.batch_first)
        ''' process '''
        if self.rnn_type == 'LSTM':
            out_pack, (ht, ct) = self.RNN(x_emb_p, None)
        else:
            out_pack, ht = self.RNN(x_emb_p, None)
            ct = None
        ''' unsort '''
        ht = ht[:, x_unsort_idx]
        if self.only_use_last_hidden_state:
            return ht
        else:
            out, _ = torch.nn.utils.rnn.pad_packed_sequence(out_pack, batch_first=self.batch_first, total_length=total_length)
            if self.batch_first:
                out = out[x_unsort_idx]
            else:
                out = out[:, x_unsort_idx]
            if self.rnn_type == 'LSTM':
                ct = ct[:, x_unsort_idx]
            return out, (ht, ct)
    
