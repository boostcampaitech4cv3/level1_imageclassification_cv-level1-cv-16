import random
import pandas as pd
import numpy as np
import os
import cv2

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm.auto import tqdm
# from tqdm.notebook import tqdm
import timm
from torchvision import models

# import albumentations as A
# from albumentations.pytorch.transforms import ToTensorV2

from torchvision import transforms

import torchvision.models as models

from sklearn.metrics import f1_score

import warnings
warnings.filterwarnings(action='ignore')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

CFG = {
    'IMG_SIZE':224,
    'EPOCHS':10,
    'LEARNING_RATE':3e-4,
    'BATCH_SIZE':2,
    'SEED':41
}

df = pd.read_csv('./train.csv')

le = preprocessing.LabelEncoder()
df['artist'] = le.fit_transform(df['artist'].values)

train_df, val_df, _, _ = train_test_split(df, df['artist'].values, test_size=0.2, random_state=CFG['SEED'])

train_df = train_df.sort_values(by=['id'])
val_df = val_df.sort_values(by=['id'])

def get_data(df, infer=False):
    if infer:
        return df['img_path'].values
    return df['img_path'].values, df['artist'].values

train_img_paths, train_labels = get_data(train_df)
val_img_paths, val_labels = get_data(val_df)

class CustomDataset(Dataset):
    def __init__(self, img_paths, labels, transforms=None):
        self.img_paths = img_paths
        self.labels = labels
        self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transforms is not None:
            image = self.transforms(image)
            image = torch.tensor(image)

        if self.labels is not None:
            label = self.labels[index]
            return image, label
        else:
            return image

    def __len__(self):
        return len(self.img_paths)

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((int(CFG["IMG_SIZE"]*1.5), int(CFG["IMG_SIZE"]*1.5))),
    transforms.RandomChoice([
        transforms.Resize((CFG["IMG_SIZE"], CFG["IMG_SIZE"])),
        transforms.CenterCrop((CFG["IMG_SIZE"], CFG["IMG_SIZE"])),
        transforms.RandomCrop((CFG["IMG_SIZE"], CFG["IMG_SIZE"])),
    ]),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    transforms.RandomApply([
        transforms.RandomRotation((-30, 30)),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.8),
    ]),
    transforms.RandomHorizontalFlip(0.3),
    transforms.RandomInvert(0.2),
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    transforms.RandomApply([
        transforms.RandomRotation((-30, 30)),
        transforms.RandomPerspective(distortion_scale=0.3, p=0.8),
    ]),
    transforms.Resize((CFG["IMG_SIZE"], CFG["IMG_SIZE"])),
    transforms.RandomHorizontalFlip(0.3),
    transforms.RandomInvert(0.2),
])


train_dataset = CustomDataset(train_img_paths, train_labels, train_transform)
train_loader = DataLoader(train_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=True, num_workers=0)

val_dataset = CustomDataset(val_img_paths, val_labels, test_transform)
val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

def train(model, optimizer, train_loader, test_loader, scheduler, device):
    model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)

    best_score = 0
    best_model = None

    for epoch in range(1, CFG["EPOCHS"] + 1):
        model.train()
        train_loss = []
        for img, label in tqdm(iter(train_loader)):
            img, label = img.float().to(device), label.type(torch.LongTensor).to(device)

            optimizer.zero_grad()

            model_pred = model(img)

            loss = criterion(model_pred, label)

            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        tr_loss = np.mean(train_loss)

        val_loss, val_score = validation(model, criterion, test_loader, device)

        print(
            f'Epoch [{epoch}], Train Loss : [{tr_loss:.5f}] Val Loss : [{val_loss:.5f}] Val F1 Score : [{val_score:.5f}]')

        if scheduler is not None:
            scheduler.step()

        if best_score < val_score:
            best_model = model
            best_score = val_score

    return best_model


class BaseModel(nn.Module):
    def __init__(self, num_classes=len(le.classes_)):
        super(BaseModel, self).__init__()
        #         self.backbone = models.efficientnet_b0(pretrained=True)
        self.backbone = timm.create_model('swin_large_patch4_window7_224', pretrained=True,
                                          num_classes=num_classes).cuda()

    #         self.classifier1 = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        #         x = self.classifier(x)
        return x


def competition_metric(true, pred):
    return f1_score(true, pred, average="macro")


def validation(model, criterion, test_loader, device):
    model.eval()

    model_preds = []
    true_labels = []

    val_loss = []

    with torch.no_grad():
        for img, label in tqdm(iter(test_loader)):
            img, label = img.float().to(device), label.to(device)

            model_pred = model(img)

            loss = criterion(model_pred, label)

            val_loss.append(loss.item())

            model_preds += model_pred.argmax(1).detach().cpu().numpy().tolist()
            true_labels += label.detach().cpu().numpy().tolist()

    val_f1 = competition_metric(true_labels, model_preds)
    return np.mean(val_loss), val_f1

import pprint
pprint.pprint(timm.models.list_models())

model = BaseModel()
model.eval()
optimizer = torch.optim.Adam(params = model.parameters(), lr = CFG["LEARNING_RATE"])
scheduler = None

infer_model = train(model, optimizer, train_loader, val_loader, scheduler, device)