from __future__ import absolute_import, division, print_function 
from yacs.config import CfgNode as CN

"""
Config defaults and can be extended by yaml config files
"""

_C = CN()


_C.SEED = 42
_C.OUTPUT_DIR = 'output'

# Dataset related params
_C.DATASET = CN()
_C.DATASET.DATA_DIR = ''
_C.DATASET.TRAIN_SIZE = 0.8
_C.DATASET.HEIGHT = 288
_C.DATASET.WIDTH = 384

# Model related params
_C.MODEL = CN()
_C.MODEL.IN_CHANNELS = 3
_C.MODEL.OUT_CHANNELS = 1
_C.MODEL.N_BLOCKS = 4
_C.MODEL.START_FILTERS = 8
_C.MODEL.ACTIVATION = 'relu'
_C.MODEL.NORMALIZATION = 'batch'
_C.MODEL.CONV_MODE = 'same'
_C.MODEL.DIM = 2

# Training related params
_C.TRAIN = CN()
_C.TRAIN.OPTIMIZER = 'adam'
_C.TRAIN.LR = 0.001
_C.TRAIN.CRITERION = 'dice'
_C.TRAIN.PATIENCE = 10
_C.TRAIN.DECAY_FACTOR = 0.1
_C.TRAIN.BATCH_SIZE = 8
_C.TRAIN.NUM_EPOCHS = 100
_C.TRAIN.START_EPOCH = 0

# Cudnn related params 
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True




def update_config(cfg, args_cfg):

    cfg.defrost()
    cfg.merge_from_file(args_cfg)
    cfg.freeze()


if __name__ == '__main__':
    import sys

    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)
