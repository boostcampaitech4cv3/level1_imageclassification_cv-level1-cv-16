

import numpy as np
import os
import cv2

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm.auto import tqdm

from torchvision import transforms

from dataloader.dataloaders import *
from model.models import *
from predict import *
from cfg import *
from data.dataset import *

import warnings
warnings.filterwarnings(action='ignore')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

CFG = load_cfg()

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
    transforms.Resize((CFG["IMG_SIZE"], CFG["IMG_SIZE"])),
    transforms.RandomHorizontalFlip(0.3),
    transforms.RandomInvert(0.2),
])

artists = Artists(seed=CFG["SEED"])

train_dataset = CustomDataset(artists.train_img_paths, artists.train_labels, train_transform)
train_loader = CustomDataLoader(train_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=True, num_workers=0)

val_dataset = CustomDataset(artists.val_img_paths, artists.val_labels, test_transform)
val_loader = CustomDataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

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
            model.save("model/model.pt")

    return best_model




import pprint
pprint.pprint(timm.models.list_models())

model = BaseModel(num_classes=len(artists.le.classes_))
model.eval()
optimizer = torch.optim.Adam(params = model.parameters(), lr = CFG["LEARNING_RATE"])
scheduler = None

infer_model = train(model, optimizer, train_loader, val_loader, scheduler, device)