import torch
from lib.config.paths import PATHS
# from .datasets.df import OpenWorldNoDef

from .datasets.coco import COCO
from .datasets.hpatches import Hpatches, HpatchesViewpoint, HpatchesIllum
from .evaluators.pck import DescPckEvaluator


class DatasetFactory:
    
    def __init__(self, cfg):
        self.cfg = cfg
        
    def get(self, name: str):
        """
        """
        factory = None
        args = {}
        if 'COCO' in name:
            # COCO.train, COCO.val
            # name: COCO, mode: 'train', 'val'
            name, _, mode = name.partition('.')
            factory = COCO
            args = {
                'root': PATHS[name]['root'],
                'mode': mode,
                'num_kps': self.cfg.DATASET.COCO.KPS,
                'size': (self.cfg.DATASET.COCO.HEIGHT, self.cfg.DATASET.COCO.WIDTH)
            }
        elif 'HPATCHES' in name:
            name, _, mode = name.partition('.')
            factory = {
                'HPATCHES': Hpatches,
                'HPATCHES_VIEWPOINT': HpatchesViewpoint,
                'HPATCHES_ILLUM': HpatchesIllum
            }[name]
            args = {
                'root': PATHS['HPATCHES']['root'],
                'size': self.cfg.DATASET.HPATCHES.SIZE,
                'num_kps': self.cfg.DATASET.HPATCHES.KPS,
                'mode': mode
            }
            

        return factory, args

class EvaluatorFactory:
    
    def __init__(self, cfg):
        self.cfg = cfg
    
    def get(self, name: str):
        """
        """
        factory = None
        args = {}
        
        if name == 'DESC_PCK':
            factory = DescPckEvaluator
            args = {
                'threshold': self.cfg.TEST.PCK_THRESHOLD
            }
        
        
        return factory, args
    
