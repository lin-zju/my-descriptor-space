import torch
from torch import optim

def make_optimizer(cfg, model: torch.nn.Module):
    lr = cfg.TRAIN.BASE_LR
    weight_decay = cfg.TRAIN.WEIGHT_DECAY
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    return optimizer

def make_scheduler(cfg, optimizer):
    return None
