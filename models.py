import math
import torch
import torch.nn as nn
from utils import reset_params
from layers import DynamicLSTM, Attention, MeanPooling, LinearLayer

class TransDelta(nn.Module):
    ''' Bi-GRU network with soft transferring '''
    def __init__(self, embedding_matrix, opt):
        super(TransDelta, self).__init__()
        
        WD = opt.word_dim
        HD = opt.hidden_dim
        
        self.word_embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.shared_params = nn.ParameterDict()
        self.shared_modules = nn.ModuleDict({
            'rnn': DynamicLSTM(WD, HD, num_layers=opt.layer_num, batch_first=True, bidirectional=opt.bidirectional, rnn_type=opt.rnn_type)
        })
        reset_params(self.shared_modules['rnn'], opt.initializer)
        
        for name, param in self.shared_modules.named_parameters():
            name = name.split('.')[-1]
            self.shared_params[f"common_{name}"] = nn.Parameter(param.data, requires_grad=True)
            self.shared_params[f"main_{name}"] = nn.Parameter(torch.zeros_like(param), requires_grad=True)
            self.shared_params[f"aux_{name}"] = nn.Parameter(torch.zeros_like(param), requires_grad=True)
        
        output_layers = {
            'attention': {'asc': Attention, 'dsc': MeanPooling, 'ae': LinearLayer},
            'mean_pooling': {'asc': MeanPooling, 'dsc': MeanPooling, 'ae': LinearLayer}
        }
        self.output = nn.ModuleDict({side: output_layers[opt.output_layer][opt.tasks[side]](opt) for side in ['main', 'aux']})
        self.threshold = nn.Parameter(torch.tensor(-math.log(1/opt.beta-1)), requires_grad=True)
        self.dropout = nn.Dropout(opt.dropout)
        self.zeta = 0.0 if opt.hard_M else opt.zeta
        self.model_hard_transfer = opt.hard_M
    
    def forward(self, inputs, side):
        text = inputs[0]
        text_len = torch.sum(text!=0, dim=-1)
        rnn_input = self.word_embedding(text)
        rnn_output, _ = self.shared_modules['rnn'](self.dropout(rnn_input), text_len)
        return self.output[side](inputs, rnn_input, rnn_output)
    
    def update_params(self, side):
        self.shared_modules.zero_grad()
        for name, param in self.shared_modules.named_parameters():
            name = name.split('.')[-1]
            if self.model_hard_transfer:
                new_param = self.shared_params[f"common_{name}"]
            else:
                new_param = self.shared_params[f"common_{name}"] + self.shared_params[f"{side}_{name}"]
            setattr(param, 'data', new_param.data)
    
    def compute_final_loss(self, side):
        final_loss, norm_loss = 0, 0
        for name, param in self.shared_modules.named_parameters():
            name = name.split('.')[-1]
            if self.model_hard_transfer:
                temp = self.shared_params[f"common_{name}"]
            else:
                temp = self.shared_params[f"common_{name}"] + self.shared_params[f"{side}_{name}"]
            final_loss += torch.sum(temp * param.grad)
            norm_loss += torch.sum(torch.pow(self.shared_params[f"{side}_{name}"], 2))
        return final_loss + self.zeta * norm_loss
    
