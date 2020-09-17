import math
import torch
import torch.nn as nn

def _reset(param, initializer):
    if param.requires_grad:
        if len(param.shape) > 1:
            initializer(param)
        else:
            stdv = 1. / math.sqrt(param.shape[0])
            torch.nn.init.uniform_(param, a=-stdv, b=stdv)

def reset_params(params, initializer, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if isinstance(params, nn.Module): # Layers
        for p in params.parameters():
            _reset(p, initializer)
    elif isinstance(params, nn.Parameter): # Weights or bias
        _reset(params, initializer)
    else:
        raise Exception
