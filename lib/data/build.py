import torch
from torch.utils.data import DataLoader
from .factory import DatasetFactory
from .transforms import transforms as T

def make_dataloader(cfg, mode):
    # the number of GPUs, or one CPU
    num_gpus = int(torch.cuda.device_count()) \
        if torch.cuda.is_available() else 1
    
    # test or train mode setting
    if mode == 'train':
        batch_size = cfg.TRAIN.BATCH_SIZE
        shuffle = True
    elif mode == 'val':
        batch_size = cfg.VAL.BATCH_SIZE
        shuffle = False
    elif mode == 'test':
        batch_size = cfg.TEST.BATCH_SIZE
        shuffle = False
        
    assert batch_size % num_gpus == 0,\
        'Batch size ({}) must be divisble by the number of GPUs ({})'.format(
            batch_size, num_gpus)
        
    # build dataset
    dataset = make_dataset(cfg, mode)
    num_workers = cfg.DATALOADER.NUM_WORKERS
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return dataloader
    
def make_dataset(cfg, mode):
    datasets = DatasetFactory(cfg)
    name = {
        'train': cfg.DATASET.TRAIN,
        'val': cfg.DATASET.VAL,
        'test': cfg.DATASET.TEST,
    }[mode]
    factory, args = datasets.get(name)
    # make transform
    transforms = make_transforms(cfg, mode == 'train')
    args['transforms'] = transforms
    return factory(**args)

def make_transforms(cfg, is_train):
    if is_train is True:
        transform = T.Compose(
            [
                # T.JpegCompress(),
                # T.GaussianBlur(),
                # T.AddNoise(),
                # T.Jitter(),
                # T.ToTensor(),
                # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    else:
        transform = T.Compose(
            [
                # T.ToTensor(),
                # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    return transform
