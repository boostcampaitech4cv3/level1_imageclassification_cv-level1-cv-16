import random
import os
import torch
import numpy as np

def load_cfg():
    CFG = {
        'IMG_SIZE':224,
        'EPOCHS':10,
        'LEARNING_RATE':3e-4,
        'BATCH_SIZE':4,
        'SEED':41
    }

    def seed_everything(seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

    seed_everything(CFG['SEED']) # Seed 고정

    return CFG