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
from lib.utils.tensorboard import TensorBoard
from lib.utils.vis_logger import make_getter
from lib.config import cfg
from lib.data import make_evaulator


def train_net(cfg):
    """
    General training procedure
    """
    
    # model
    device = torch.device(cfg.MODEL.DEVICE)
    device_ids = cfg.MODEL.DEVICE_IDS
    if not device_ids: device_ids = None # use all devices
    model = make_model(cfg)
    model = model.to(device)
    if cfg.MODEL.PARALLEL:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    
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
        save_dir=save_dir
    )
    if cfg.TRAIN.RESUME:
        checkpointer.load()
        
    # tensorboard and visualization
    tensorboard = None
    getter = None
    logdir = os.path.join(cfg.TENSORBOARD.LOG_DIR, cfg.EXP.NAME)
    if cfg.TENSORBOARD.IS_ON:
        tensorboard = TensorBoard(
            logdir=logdir,
            scalars=cfg.TENSORBOARD.TARGETS.SCALAR,
            images=cfg.TENSORBOARD.TARGETS.IMAGE,
            resume=cfg.TRAIN.RESUME
        )
        getter = make_getter(cfg)
    
    # validation
    dataloader_val, evaluator = None, None
    if cfg.VAL.IS_ON:
        dataloader_val = make_dataloader(cfg, 'val')
        evaluator = make_evaulator(cfg, 'val')
    # training parameters
    params = {
        'max_epochs': cfg.TRAIN.MAX_EPOCHS,
        'checkpoint_period': cfg.TRAIN.CHECKPOINT_PERIOD,
        'print_every': cfg.TRAIN.PRINT_EVERY,
        'val_every': cfg.TRAIN.VAL_EVERY
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
        tensorboard,
        getter,
        
        dataloader_val=dataloader_val,
        evaluator=evaluator
    )
    
    return model

def main():
    parse(cfg)
    model = train_net(cfg)
    
if __name__ == '__main__':
    main()
