from .factory import ModelFactory

def make_model(cfg):
    factory = ModelFactory(cfg)
    func, args = factory.get(cfg.MODEL.NAME)
    
    return func(args)
    
