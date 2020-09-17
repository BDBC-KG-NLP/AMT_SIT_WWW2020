import torch
import torch.nn as nn
import torch.nn.functional as F

class CE_Loss(nn.Module):
    
    def __init__(self, opt):
        super(CE_Loss, self).__init__()
        self.celoss = nn.CrossEntropyLoss(reduction='none')
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.opt = opt
    
    def forward(self, inputs, outputs, targets, task, model=None, gamma=None, pseudo=False, inference=False):
        if task == 'ae':
            text = inputs[0]
            text_mask = (text!=0)
            text_len = torch.sum(text!=0, dim=-1)
        if pseudo:
            if task == 'ae':
                pseudo_targets = torch.argmax(outputs, dim=-1)
                probs = F.softmax(outputs, dim=-1)
                maxp = torch.max(probs, dim=-1).values
                maxp = torch.where(text_mask!=0, maxp, torch.zeros_like(maxp))
                maxp = torch.sum(maxp, dim=-1).div(text_len.float())
                beta = torch.sigmoid(model.threshold) if self.opt.learned_beta else self.opt.beta
                if self.opt.hard_D:
                    weight = torch.where(maxp > beta, torch.ones_like(maxp), torch.zeros_like(maxp))
                else:
                    weight = torch.sigmoid(self.opt.alpha * (maxp - beta))
                pseudo_logits = self.logsoftmax(outputs)
                pseudo_targets = F.one_hot(pseudo_targets, num_classes=3).float()
                log_likelihood = torch.sum(pseudo_targets * pseudo_logits, dim=-1)
                log_likelihood = torch.where(text_mask!=0, log_likelihood, torch.zeros_like(log_likelihood))
                log_likelihood = torch.sum(log_likelihood, dim=-1).div(text_len.float())
                loss = -torch.mean(weight * log_likelihood)
            else:
                pseudo_targets = torch.argmax(outputs, dim=-1)
                probs = F.softmax(outputs, dim=-1)
                maxp = torch.max(probs, dim=-1).values
                beta = torch.sigmoid(model.threshold) if self.opt.learned_beta else self.opt.beta
                if self.opt.hard_D:
                    weight = torch.where(maxp > beta, torch.ones_like(maxp), torch.zeros_like(maxp))
                else:
                    weight = torch.sigmoid(self.opt.alpha * (maxp - beta))
                loss = gamma * torch.sum(weight * self.celoss(outputs, pseudo_targets)).div(weight.sum()+1e-6)
        else:
            if task == 'ae':
                logits = self.logsoftmax(outputs)
                targets = F.one_hot(targets, num_classes=3).float()
                log_likelihood = torch.sum(targets * logits, dim=-1)
                log_likelihood = torch.where(text_mask!=0, log_likelihood, torch.zeros_like(log_likelihood))
                log_likelihood = torch.sum(log_likelihood, dim=-1).div(text_len.float())
                loss = -torch.mean(log_likelihood)
            else:
                loss = torch.mean(self.celoss(outputs, targets))
        if pseudo and inference:
            return loss, weight
        else:
            return loss
    
