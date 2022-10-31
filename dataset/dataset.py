import pandas as pd
import torch
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from glob import glob
from PIL import Image
import cv2
from torch.utils.data import Dataset
import os
import random
import json


class TestDataset(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.img_dir = "/opt/ml/input/data"
        self.test_dir = f"{self.img_dir}/eval"
        
        self.test_df = pd.read_csv(f"{self.test_dir}/info.csv")
        self.img_paths = self.get_data(self.test_df)
        
        self.transforms = self.transformation()
        
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        img_path = f"{self.test_dir}/images/{self.img_paths[index]}"
            
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transforms is not None:
            image = self.transforms(image=image)['image']
            
        return image
        
    def get_data(self, df):
        return df["ImageID"].values
    
    def transformation(self):
        return A.Compose([
                A.ToGray(p=1),
                A.Resize(int(self.cfg["IMG_SIZE"]), int(self.cfg["IMG_SIZE"])),
                A.Sharpen(alpha=0.7, p=1),
                A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
        
    def submit(self, preds):
        self.test_df['ans'] = preds
        self.test_df.to_csv('./output.csv', index=False)
    
        
class CustomDataset(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        
        self.img_dir = "/opt/ml/input/data"
        self.train_dir = f"{self.img_dir}/train"
        
        train_df = pd.read_csv(f"{self.train_dir}/train.csv")
        self.dataset = self._preprocess(train_df)
        
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        img_path = self._get_method(index)
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transforms is not None:
            image = self.transforms(image=image)['image']

        if self.labels is not None:
            label = self.labels[index]
            return image, label
        else:
            return image
    
    def _get_method(self, index):
        img_path = self.img_paths[index]
        
        return img_path
    
    
    def _preprocess(self, df):
        def get_mask_label(x):
            if x.startswith("incorrect"):
                return 1
            elif x.startswith("normal"):
                return 2
            else:
                return 0
        
        df["path"] = df["path"].apply(lambda x: f"{self.train_dir}/images/{x}/")
        df["file"] = df.apply(lambda x: list(map(os.path.basename, glob(f"{x['path']}/*"))), axis=1)
        bins = [0,  29, 59, 60] ## 이상 미만 vs 초과 이하 구분 필요
        bins_label = [0, 1, 2]
        df["Age"] = pd.cut(df["age"], bins, labels=bins_label)
        df["Gender"] = df["gender"].replace({'male':0,'female':1})
        df["Mask"] = df["file"].apply(lambda x: [get_mask_label(i) for i in x])
        
        base_dataset = pd.DataFrame({"path": df["path"], "Age": df["Age"], "Gender": df["Gender"], "file": df["file"], "Mask": df["Mask"]})
        
        return base_dataset
    
    def get_data(self, df, infer=False):
        if infer:
            return df['path'].values
        return df['path'].values, df['label'].values
    
    def transformation(self, val):
        if not val:
            return A.Compose([
                A.ToGray(p=1),
                A.RandomResizedCrop(self.cfg["IMG_SIZE"], self.cfg["IMG_SIZE"], scale=(0.5, 1.0)),
                A.Sharpen(alpha=(0.5, 1), p=1),
                A.HorizontalFlip(p=0.3),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                ToTensorV2()
            ])

        else:
            return A.Compose([
                A.ToGray(p=1),
                A.Resize(int(self.cfg["IMG_SIZE"]), int(self.cfg["IMG_SIZE"])),
                A.Sharpen(alpha=0.7, p=1),
                A.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
    

class Age_Dataset(CustomDataset):
    def __init__(self, cfg, val=False):
        super().__init__(cfg)
        self.num_classes = 3
        
        self.df = self.preprocess(val)
        self.img_paths, self.labels = self.get_data(self.df)
    
    def _get_method(self, index):
        img_path = self.img_paths[index]
        imgs = glob(f"{img_path}/*")
        img_path = random.choice(imgs)
        
        return img_path
        
    def preprocess(self, val):
        dataset = self.dataset.drop(["file", "Mask", "Gender"], axis=1)
        dataset = dataset.rename(columns={"Age": "label"})
        train_df, val_df = train_test_split(dataset, test_size=0.2, random_state=41)
        
#         train_df = self.undersampling(dataset)
#         print(train_df.groupby('label', as_index=False).count())
        
        train_df = train_df.sort_index()
        val_df = val_df.sort_index()

        self.transforms = self.transformation(val)
        
        if not val:
            return train_df
        else:
            return val_df
    
    def undersampling(self, df):
        nMax = 192 #change to 2500
        res = df.groupby('label', as_index=False).apply(lambda x: x.sample(n=min(nMax, len(x))))
        res = pd.DataFrame({"path": res["path"].values, "label": res["label"].values})
        res = res.sample(frac=1)
        
        return res

    
    
class Gender_Dataset(CustomDataset):
    def __init__(self, cfg, val=False):
        super().__init__(cfg)
        self.num_classes = 2
        
        self.df = self.preprocess(val)
        self.img_paths, self.labels = self.get_data(self.df)
        
    def _get_method(self, index):
        img_path = self.img_paths[index]
        imgs = glob(f"{img_path}/*")
        img_path = random.choice(imgs)
        
        return img_path
        
    def preprocess(self, val):
        dataset = self.dataset.drop(["file", "Mask", "Age"], axis=1)
        dataset = dataset.rename(columns={"Gender": "label"})
        train_df, val_df = train_test_split(dataset, test_size=0.2, random_state=41)
              
        
        train_df = train_df.sort_index()
        val_df = val_df.sort_index()
        
        self.transforms = self.transformation(val)
        
        if not val:
            return train_df
        else:
            return val_df
    
        
    
class Mask_Dataset(CustomDataset):
    def __init__(self, cfg, val=False):
        super().__init__(cfg)
        self.num_classes = 3
        
        self.df = self.preprocess(val)
        
        
        self.img_paths, self.labels = self.get_data(self.df)
                
    def preprocess(self, val):
        dataset = self.dataset.drop(["Age", "Gender"], axis=1)
        path = dataset.apply(lambda x: [os.path.join(x["path"], i) for i in x["file"]], axis = 1)
        path = pd.DataFrame(path.tolist()).add_prefix('path').stack().tolist()
        
        label = dataset["Mask"]
        label = pd.DataFrame(label.tolist()).add_prefix('label').stack().tolist()
        
        dataset = pd.DataFrame({"path": path, "label": label})
        
        train_df, val_df = train_test_split(dataset, test_size=0.2, random_state=41)
        
        train_df = train_df.sort_index()
        val_df = val_df.sort_index()
    
        
        self.transforms = self.transformation(val)
        
        if not val:
            return train_df
        else:
            return val_df
    
        
    
if __name__ == "__main__":
    import json
    cfg = json.load(open("../cfg.json", "r"))
    dataset = Age_Dataset(cfg)