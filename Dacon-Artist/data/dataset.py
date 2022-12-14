import pandas as pd
import torch
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from torchvision import transforms

class Artists:
    def __init__(self, CFG):
        df = pd.read_csv('train.csv')

        self.train_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((int(CFG["IMG_SIZE"] * 2), int(CFG["IMG_SIZE"] * 2))),
            transforms.ConvertImageDtype(torch.uint8),      # uint 타입을 필요로 하는 layer에 대한 에러 방지
            transforms.RandomCrop((int(CFG["IMG_SIZE"]), int(CFG["IMG_SIZE"]))),
            # transforms.RandomApply([
            #     transforms.RandomRotation((-30, 30)),
            #     transforms.RandomPerspective(distortion_scale=0.3, p=0.8),
            # ]),
            transforms.ConvertImageDtype(torch.float32),    # ufloat타입(normalize 등..)을 필요로 하는 layer에 대한 에러 방지
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transforms.RandomHorizontalFlip(0.3),
            transforms.RandomInvert(0.2),
        ])

        self.val_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((int(CFG["IMG_SIZE"]*2), int(CFG["IMG_SIZE"]*2))),
            transforms.RandomCrop((int(CFG["IMG_SIZE"]), int(CFG["IMG_SIZE"]))),
            ###
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transforms.RandomHorizontalFlip(0.3),
            transforms.RandomInvert(0.2),
        ])


        self.test_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((int(CFG["IMG_SIZE"]), int(CFG["IMG_SIZE"]))),
            # transforms.RandomChoice([
            #     transforms.CenterCrop((int(CFG["IMG_SIZE"]), int(CFG["IMG_SIZE"]))),
            #     transforms.RandomCrop((int(CFG["IMG_SIZE"]), int(CFG["IMG_SIZE"]))),
            # ]),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transforms.RandomHorizontalFlip(0.3),
            transforms.RandomInvert(0.2),
        ])


        self.le = preprocessing.LabelEncoder()
        df['artist'] = self.le.fit_transform(df['artist'].values)

        train_df, val_df, _, _ = train_test_split(df, df['artist'].values, test_size=0.2, random_state=CFG["SEED"])

        train_df = train_df.sort_values(by=['id'])
        val_df = val_df.sort_values(by=['id'])
        test_df = pd.read_csv('test.csv')

        self.train_img_paths, self.train_labels = self.get_data(train_df)
        self.val_img_paths, self.val_labels = self.get_data(val_df)
        self.test_img_paths = self.get_data(test_df, infer=True)

    def get_data(self, df, infer=False):
        if infer:
            return df['img_path'].values
        return df['img_path'].values, df['artist'].values