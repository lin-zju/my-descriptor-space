import sys
import torch
import os
import argparse

if not '.' in sys.path:
    sys.path.insert(0, '.')

from lib.config.parse import parse
from lib.data import make_dataloader
from lib.modeling import make_model
from lib.solver import make_optimizer, make_scheduler
from lib.engine.trainer import train
from lib.engine.evaluator import evaluate
from lib.utils.checkpoint import Checkpointer
from lib.config import cfg


def train_net(cfg):
    """
    General training procedure
    """
    
    # model
    device = torch.device(cfg.MODEL.DEVICE)
    model = make_model(cfg)
    model = model.to(device)
    if cfg.MODEL.PARALLEL:
        model = torch.nn.DataParallel(model)
    
    optimizer = make_optimizer(cfg, model)
    scheduler = make_scheduler(cfg, optimizer)
    dataloader = make_dataloader(cfg, mode='train')
    
    # checkpointer
    arguments = {'epoch': 0, 'iteration': 0}
    save_dir = os.path.join(cfg.MODEL_DIR, cfg.EXP.NAME)
    checkpointer = Checkpointer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        other=arguments,
        max_checkpoints=cfg.TRAIN.NUM_CHECKPOINTS,
        save_dir=cfg.MODEL_DIR
    )
    if cfg.TRAIN.RESUME:
        checkpointer.load()
    
    # training parameters
    params = {
        'max_epochs': cfg.TRAIN.MAX_EPOCHS,
        'checkpoint_period': cfg.TRAIN.CHECKPOINT_PERIOD,
        'print_every': cfg.TRAIN.PRINT_EVERY
    }
    
    train(
        model,
        dataloader,
        optimizer,
        scheduler,
        checkpointer,
        device,
        arguments,
        params,
    )
    
    return model

def main():
    parse(cfg)
    model = train_net(cfg)
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    

