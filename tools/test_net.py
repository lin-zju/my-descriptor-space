import sys
import torch
import os
import argparse

if not '.' in sys.path:
    sys.path.insert(0, '.')

from lib.config.parse import parse
from lib.data import make_dataloader, make_evaulator
from lib.modeling import make_model
from lib.engine.evaluator import evaluate
from lib.utils.checkpoint import Checkpointer
from lib.config import cfg


def test_net(cfg):
    """
    General test procedure
    """
    
    # model
    device = torch.device(cfg.MODEL.DEVICE)
    model = make_model(cfg)
    if cfg.MODEL.PARALLEL:
        model = torch.nn.DataParallel(model)
    # model = model.to(device)
    
    dataloader = make_dataloader(cfg, mode='test')
    evaluator = make_evaulator(cfg)
    
    # checkpointer
    save_dir = os.path.join(cfg.MODEL_DIR, cfg.EXP.NAME)
    checkpointer = Checkpointer(
        model=model,
        optimizer=None,
        scheduler=None,
        other={},
        max_checkpoints=cfg.TRAIN.NUM_CHECKPOINTS,
        save_dir=save_dir
    )
    checkpointer.load()
    
    print()
    print('-' * 80)
    print('Testing model "{}" on "{}"...'.format(cfg.MODEL.NAME, cfg.DATASET.TEST))
    evaluate(
        model,
        device,
        dataloader,
        evaluator
    )
    print('-' * 80)
    
    return model


def main():
    parse(cfg)
    model = test_net(cfg)


if __name__ == '__main__':
    main()









