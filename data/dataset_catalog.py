from .input_ksdd import KSDDDataset
from .input_dagm import DagmDataset
from .input_steel import SteelDataset
from .input_ksdd2 import KSDD2Dataset
from config import Config
from torch.utils.data import DataLoader
from typing import Optional
import torch
from .input_potSeg import CustomPotSeg

def get_dataset(kind: str, cfg: Config, args = None):
    
    if cfg.DATASET == 'CustomPotSeg':
        #print(f'calling {__file__}, {sys._getframe().f_lineno}')
        '''
        label_dict = {'lasi_heavy': 11, 'lasi_medium':12, 'lasi_slight':13,
        'gengshang_heavy':21, 'gengshang_medium':22, 'gengshang_slight':23,  
        'gengshi_heavy':31, 'gengshi_medium':32, 'gengshi_slight':33,
        'shayan_heavy':41, 'shayan_medium':42, 'shayan_medium':43,
        'huahen_heavy':51, 'huahen_medium':52, 'huahen_medium':53,
        'zhoubian_heavy':61, 'zhoubian_medium':62, 'zhoubian_medium':63,
        'bowen_heavy':71, 'bowen_medium':72, 'bowen_medium':73,
        'youwu_heavy':81, 'youwu_medium':82, 'youwu_medium':83,
        }
        '''    
        batch_size = cfg.BATCH_SIZE if kind == "TRAIN" else 1
        split = kind.lower()
        data_set = CustomPotSeg(cfg, args, split=split)
        # if args.pot_train_mode == 1: #不区分类别
        #     num_class = 2
        # if args.pot_train_mode == 2: #不区分类别,只处理前三类
        #     num_class = 2
        # else:
        #     num_class = args.n_classes

        # train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(data_set, num_replicas=args.world_size, rank=args.rank, shuffle = True)
        else:
            sampler = None
        if split == 'train':
            data_loader = DataLoader(data_set, batch_size= batch_size, shuffle=(sampler is None), drop_last=True, sampler=sampler)
        else:
            data_loader = DataLoader(data_set, batch_size= batch_size, shuffle=(sampler is None), drop_last=False, sampler=sampler)
        
        return data_loader
    
    else:
        if kind == "VAL" and not cfg.VALIDATE:
            return None
        if kind == "VAL" and cfg.VALIDATE_ON_TEST:
            kind = "TEST"
        if cfg.DATASET == "KSDD":
            ds = KSDDDataset(kind, cfg)
        elif cfg.DATASET == "DAGM":
            ds = DagmDataset(kind, cfg)
        elif cfg.DATASET == "STEEL":
            ds = SteelDataset(kind, cfg)
        elif cfg.DATASET == "KSDD2":
            ds = KSDD2Dataset(kind, cfg)
        else:
            raise Exception(f"Unknown dataset {cfg.DATASET}")

        # shuffle = kind == "TRAIN"
        shuffle = True
        batch_size = cfg.BATCH_SIZE if kind == "TRAIN" else 1
        num_workers = 0
        drop_last = kind == "TRAIN"
        pin_memory = False

        return DataLoader(dataset=ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last, pin_memory=pin_memory)
