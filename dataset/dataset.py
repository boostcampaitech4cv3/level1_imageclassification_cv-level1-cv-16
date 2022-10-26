import pandas as pd
import torch
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from torchvision import transforms
from glob import glob
import os

class Dataset:
    def __init__(self):
        self.img_dir = "/opt/ml/input/data"
        self.train_dir = f"{self.img_dir}/train"
        self.test_dir = f"{self.img_dir}/eval"
        
        df = pd.read_csv(f"{self.train_dir}/train.csv")
        self.dataset = self._preprocess(df)
# #         train_df, val_df, _, _ = train_test_split(df, df['artist'].values, test_size=0.2, random_state=CFG["SEED"])

#         train_df = train_df.sort_values(by=['id'])
#         val_df = val_df.sort_values(by=['id'])
#         test_df = pd.read_csv(f"{train_dir}/test.csv")
        
#         self.train_img_paths, self.train_labels = self.get_data(train_df)
#         self.val_img_paths, self.val_labels = self.get_data(val_df)
#         self.test_img_paths = self.get_data(test_df, infer=True)
        
#         self.train_transform, self.val_transform, self.test_transforms = self.transformation()
    
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
        bins = range(0, 61, 30) ## 이상 미만 vs 초과 이하 구분 필요
        bins_label = [0, 1, 2]
        df["Age"] = pd.cut(df["age"], bins, labels=bins_label[:-1])
        df["Gender"] = df["gender"].replace({'male':0,'female':1})
        df["Mask"] = df["file"].apply(lambda x: [get_mask_label(i) for i in x])
        
        base_dataset = pd.DataFrame({"path": df["path"], "Age": df["Age"], "Gender": df["Gender"], "file": df["file"], "Mask": df["Mask"]})
        
        return base_dataset
        
        
    def transformation(self):
        train_transforms = transforms.Compose([
            transforms.Resize((int(CFG["IMG_SIZE"] * 2), int(CFG["IMG_SIZE"] * 2))),
            transforms.RandomCrop((int(CFG["IMG_SIZE"]), int(CFG["IMG_SIZE"]))),
            transforms.RandomHorizontalFlip(0.3),
            transforms.RandomInvert(0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        val_transforms = transforms.Compose([
            transforms.Resize((int(CFG["IMG_SIZE"]*2), int(CFG["IMG_SIZE"]*2))),
            transforms.RandomCrop((int(CFG["IMG_SIZE"]), int(CFG["IMG_SIZE"]))),
            transforms.RandomHorizontalFlip(0.3),
            transforms.RandomInvert(0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        test_transforms = transforms.Compose([
            transforms.Resize((int(CFG["IMG_SIZE"]), int(CFG["IMG_SIZE"]))),
            transforms.RandomHorizontalFlip(0.3),
            transforms.RandomInvert(0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        
        return train_transforms, val_transforms, test_transforms
    
class AG_Dataset(Dataset):
    def __init__(self):
        super().__init__()
        self.dataset = self.preprocess()
        
        train_df, val_df = train_test_split(self.dataset, test_size=0.2, random_state=41)
        
        train_df = train_df.sort_index()
        val_df = val_df.sort_index()
        
        self.train_img_paths, self.train_age_labels, self.train_gender_labels = self.get_data(train_df)
        self.val_img_paths, self.val_age_labels, self.val_gender_labels = self.get_data(val_df)
#         self.test_img_paths = self.get_data(test_df, infer=True)

        
    def preprocess(self):
        dataset = self.dataset.drop(["file", "Mask"], axis=1)
        return dataset
    
    def get_data(self, df, infer=False):
        if infer:
            return df['path'].values
        return df['path'].values, df['Age'].values, df['Gender'].values
        
    
class Mask_Dataset(Dataset):
    def __init__(self):
        super().__init__()
        self.dataset = self.preprocess()
        
        train_df, val_df = train_test_split(self.dataset, test_size=0.2, random_state=41)
        
        train_df = train_df.sort_index()
        val_df = val_df.sort_index()
        
        
        self.train_img_paths, self.train_mask_labels = self.get_data(train_df)
        self.val_img_paths, self.val_mask_labels = self.get_data(val_df)
        self.test_img_paths = self.get_data(test_df, infer=True)

        
    def preprocess(self):
        dataset = self.dataset.drop(["Age", "Gender"], axis=1)
        path = dataset.apply(lambda x: [os.path.join(x["path"], i) for i in x["file"]], axis = 1)
        path = pd.DataFrame(path.tolist()).add_prefix('path').stack().tolist()
        
        label = dataset["Mask"]
        label = pd.DataFrame(label.tolist()).add_prefix('label').stack().tolist()
        
        dataset = pd.DataFrame({"path": path, "label": label})
        
        return dataset
        
    def get_data(self, df, infer=False):
        if infer:
            return df['path'].values
        return df['path'].values, df['label'].values
        
    
if __name__ == "__main__":
    dataset = Mask_Dataset()