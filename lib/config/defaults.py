"""
    Default setting for the network,
    Could be overwrite with training configs
"""

import os
from yacs.config import CfgNode as CN

_C = CN()

# -----------------------------------------------------------------------------
# EXP
# -----------------------------------------------------------------------------
_C.EXP = CN()
_C.EXP.NAME = 'test'

# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# this specifies the model to use
_C.MODEL.NAME = "MSNetV0"
# _C.MODEL.TEST = False
_C.MODEL.DEVICE = "cpu"
_C.MODEL.PARALLEL = False

# -----------------------------------------------------------------------------
# PATH SETTING
# -----------------------------------------------------------------------------
_C.PATH = CN()
_C.PATH.CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
_C.PATH.LIB_DIR = os.path.dirname(_C.PATH.CONFIG_DIR)
_C.PATH.ROOT_DIR = os.path.dirname(_C.PATH.LIB_DIR)
_C.PATH.DATA_DIR = os.path.join(_C.PATH.ROOT_DIR, 'data')

# -----------------------------------------------------------------------------
# DATASET
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.TRAIN = 'COCO.train'
_C.DATASET.VAL = 'COCO.val'
_C.DATASET.TEST = 'COCO.test'

# -----------------------------------------------------------------------------
# DATASET specific
# -----------------------------------------------------------------------------
_C.DATASET.COCO = CN()
_C.DATASET.COCO.WIDTH = 320
_C.DATASET.COCO.HEIGHT = 240
# number of keypoints to sample
_C.DATASET.COCO.KPS = 3000

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 4

# ---------------------------------------------------------------------------- #
# Specific train options
# ---------------------------------------------------------------------------- #
_C.TRAIN = CN()

# if true, the model, optimizer, schduler will be loaded
_C.TRAIN.RESUME = True

# number of epochs
_C.TRAIN.MAX_EPOCHS = 30

# batch size
_C.TRAIN.BATCH_SIZE = 128

# use Adam as default
_C.TRAIN.BASE_LR = 0.001
_C.TRAIN.WEIGHT_DECAY = 0.0005

_C.TRAIN.CHECKPOINT_PERIOD = 2500
_C.TRAIN.NUM_CHECKPOINTS = 10
_C.TRAIN.PRINT_EVERY = 20

# ---------------------------------------------------------------------------- #
# Validation settings
# ---------------------------------------------------------------------------- #
_C.VAL = CN()

# validation on?
_C.VAL.IS_ON = False
_C.VAL.BATCH_SIZE = 1

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
_C.TEST.BATCH_SIZE = 1

# ---------------------------------------------------------------------------- #
# Tensorboard
# ---------------------------------------------------------------------------- #
_C.TENSORBOARD = CN()
_C.TENSORBOARD.IS_ON = True
_C.TENSORBOARD.TARGETS = CN()
_C.TENSORBOARD.TARGETS.SCALAR = ["loss"]
_C.TENSORBOARD.TARGETS.IMAGE = []
_C.TENSORBOARD.LOG_DIR = os.path.join(_C.PATH.ROOT_DIR, "logs")


# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #

# default model saving directory
_C.MODEL_DIR = os.path.join(_C.PATH.DATA_DIR, "model")

# ---------------------------------------------------------------------------- #
# Path setups
# ---------------------------------------------------------------------------- #
import sys
import os

if _C.PATH.ROOT_DIR not in sys.path:
    sys.path.append(_C.PATH.ROOT_DIR)

if not os.path.exists(_C.MODEL_DIR):
    os.makedirs(_C.MODEL_DIR)
