import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from .augmentation import undercrop

import json
cfg = json.load(open("cfg.json", "r"))
class Train_Transform:
    age = A.Compose([
                A.ToGray(p=1),
                A.CenterCrop(cfg["IMG_SIZE"], cfg["IMG_SIZE"]),
                A.Sharpen(alpha=0.7, p=1),
                A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
    
    gender = A.Compose([
                A.ToGray(p=1),
                A.Lambda(image=undercrop),
                A.Sharpen(alpha=(0.5, 1), p=1),
                A.HorizontalFlip(p=0.3),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                ToTensorV2()
            ])
    
    mask = A.Compose([
                A.ToGray(p=1),
                A.RandomResizedCrop(cfg["IMG_SIZE"], cfg["IMG_SIZE"], scale=(0.8, 1.0)),
                A.HorizontalFlip(p=0.3),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                ToTensorV2()
            ])
       
class Val_Transform:
    age = A.Compose([
                A.ToGray(p=1),
                A.CenterCrop(cfg["IMG_SIZE"], cfg["IMG_SIZE"]),
                A.Sharpen(alpha=0.7, p=1),
                A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
    
    gender = A.Compose([
                A.ToGray(p=1),
                A.Lambda(image=undercrop),
                A.Sharpen(alpha=0.7, p=1),
                A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
    
    mask = A.Compose([
                A.ToGray(p=1),
                A.Resize(cfg["IMG_SIZE"], cfg["IMG_SIZE"]),
                A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ToTensorV2()
            ])