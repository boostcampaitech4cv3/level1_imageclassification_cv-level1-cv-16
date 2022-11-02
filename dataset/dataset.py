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
import numpy as np

from .augmentation import undercrop

cfg = json.load(open("cfg.json", "r"))

class HumanInfo:
    def __init__(self, test=False):
        self.img_dir = "/opt/ml/input/data"
        
        if not test:
            self.train_dir = f"{self.img_dir}/train"
    
            df = pd.read_csv(f"{self.train_dir}/train.csv")
            age_dataset, gender_dataset, mask_dataset = self._preprocess(df)

            self.train_age, self.val_age = train_test_split(age_dataset, test_size=0.2, random_state=41)
            self.train_gender, self.val_gender = train_test_split(gender_dataset, test_size=0.2, random_state=41)
            self.train_mask, self.val_mask = train_test_split(mask_dataset, test_size=0.2, random_state=41)
        
        if test:
            self.test_dir = f"{self.img_dir}/eval"
            self.test = pd.read_csv(f"{self.test_dir}/info.csv")

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
        bins = [0, 28, 58, 60] ## 이상 미만 vs 초과 이하 구분 필요
        bins_label = [0, 1, 2]
        df["Age"] = pd.cut(df["age"], bins, labels=bins_label)
        df["Gender"] = df["gender"].replace({'male':0,'female':1})
        df["Mask"] = df["file"].apply(lambda x: [get_mask_label(i) for i in x])
        
        age_dataset = pd.DataFrame({"path": df["path"], "Age": df["Age"]})
        
        gender_dataset = pd.DataFrame({"path": df["path"],"Gender": df["Gender"]})
        
        mask_dataset = pd.DataFrame({"path": df["path"], "file": df["file"], "Mask": df["Mask"]})
        mask_dataset = mask_dataset.apply(pd.Series.explode).reset_index()    ## spread file and label => 2700 * 7 = 18900
        mask_dataset['path'] = mask_dataset.apply(lambda x: f"{x['path']}/{x['file']}", axis = 1)
        
        return age_dataset, gender_dataset, mask_dataset
    
        
class CustomDataset(Dataset):
    def __init__(self, df):
        self.img_paths, self.labels = self.get_data(df)
        
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
    
        
class Age_Dataset(CustomDataset):
    def __init__(self, df, transforms):
        super().__init__(df)
        self.num_classes = 3
        self.transforms = transforms
    
    def _get_method(self, index):
        img_path = self.img_paths[index]
        imgs = glob(f"{img_path}/*")
        img_path = random.choice(imgs)
        
        return img_path

    def get_data(self, df, infer=False):
        if infer:
            return df['path'].values
        return df['path'].values, df['Age'].values
    
    def undersampling(self, df):
        nMax = 250 #change to 2500
        res = df.groupby('label', as_index=False).apply(lambda x: x.sample(n=min(nMax, len(x))))
        res = pd.DataFrame({"path": res["path"].values, "Age": res["Age"].values})
        res = res.sample(frac=1)
        
        return res
    
    
class Gender_Dataset(CustomDataset):
    def __init__(self, df, transforms):
        super().__init__(df)
        self.num_classes = 2
        self.transforms = transforms
    
    def _get_method(self, index):
        img_path = self.img_paths[index]
        imgs = glob(f"{img_path}/*")
        img_path = random.choice(imgs)
        
        return img_path        
        
    def get_data(self, df, infer=False):
        if infer:
            return df['path'].values
        return df['path'].values, df['Gender'].values
        

class Mask_Dataset(CustomDataset):
    def __init__(self, df, transforms):
        super().__init__(df)
        self.num_classes = 3
        self.transforms = transforms
    
    def _get_method(self, index):
        img_path = self.img_paths[index]
        return img_path
    
    def get_data(self, df, infer=False):
        if infer:
            return df['path'].values
        return df['path'].values, df['Mask'].values
    
    
class TestDataset(Dataset):
    def __init__(self, df, transforms):
        self.img_dir = "/opt/ml/input/data"
        self.test_dir = f"{self.img_dir}/eval"
        
        self.img_paths = self.get_data(df)
        self.age_transforms, self.gender_transforms, self.mask_transforms = transforms
        
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        img_path = f"{self.test_dir}/images/{self.img_paths[index]}"
            
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        
        age_image = self.age_transforms(image=image)['image']
        gender_image = self.gender_transforms(image=image)['image']
        mask_image = self.mask_transforms(image=image)['image']

        return age_image, gender_image, mask_image
        
    def get_data(self, df):
        return df["ImageID"].values

        
    def submit(self, df, preds):
        df['ans'] = preds
        df.to_csv('./output.csv', index=False)


class Viz_Dataset:
    def __init__(self, df, transforms):
        self.num_classes = 18
        self.train_dir = "/opt/ml/input/data/train"
        df = self.preprocess(df)
        self.img_paths, self.labels = self.get_data(df)
        self.age_transforms, self.gender_transforms, self.mask_transforms = transforms
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        img_path = self._get_method(index)
        image = cv2.imread(img_path)
        
        
        age_image = self.age_transforms(image=image)['image']
        gender_image = self.gender_transforms(image=image)['image']
        mask_image = self.mask_transforms(image=image)['image']

        label = self.labels[index]

        return age_image, gender_image, mask_image, label, img_path
        
    def _get_method(self, index):
        img_path = self.img_paths[index]
        return img_path
    
    def get_data(self, df, infer=False):
        if infer:
            return df['path'].values
        return df['path'].values, list(zip(df['Age'].values, df['Gender'].values, df['Mask'].values))
    
    def preprocess(self, df):
        def get_mask_label(x):
            if x.startswith("incorrect"):
                return 1
            elif x.startswith("normal"):
                return 2
            else:
                return 0
        
        df["path"] = df["path"].apply(lambda x: f"{self.train_dir}/images/{x}/")
        df["file"] = df.apply(lambda x: list(map(os.path.basename, glob(f"{x['path']}/*"))), axis=1)
        bins = [0, 28, 58, 60] ## 이상 미만 vs 초과 이하 구분 필요
        bins_label = [0, 1, 2]
        df["Age"] = pd.cut(df["age"], bins, labels=bins_label)
        df["Gender"] = df["gender"].replace({'male':0,'female':1})
        df["Mask"] = df["file"].apply(lambda x: [get_mask_label(i) for i in x])
                
        df = df.apply(pd.Series.explode).reset_index()    ## spread file and label => 2700 * 7 = 18900
        df['path'] = df.apply(lambda x: f"{x['path']}/{x['file']}", axis = 1)
        
        return df
        

    
if __name__ == "__main__":
    import json
    cfg = json.load(open("../cfg.json", "r"))
    dataset = Viz_Dataset(cfg)
    print(dataset.dataset[50])