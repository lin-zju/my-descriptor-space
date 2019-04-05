import torch
from lib.config.paths import PATHS
from lib.modeling import matcher



class ModelFactory:
    
    def __init__(self, cfg):
        self.cfg = cfg
    
    def get(self, name: str):
        """
        """
        func = None
        args = {}
        if 'MSNet' in name:
            # MSNetVx, where x is a number
            func = getattr(matcher, name)
            args = {}
        elif 'SuperPoint' in name:
            func = getattr(matcher, name)
            args = {}
        elif 'Daisy' in name:
            func = getattr(matcher, name)
            args = {}
        
        return func, args

