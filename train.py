import json
from dataset.dataset import *
from dataset.transformation import *
from model.models import *
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
import os

import random
import torch.backends.cudnn as cudnn
from ema_pytorch import EMA
from optim.sam import SAM

from loss.focal import Focal_Loss

import warnings
warnings.filterwarnings(action='ignore')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

cfg = json.load(open("cfg.json", "r"))

torch.manual_seed(cfg["SEED"])
torch.cuda.manual_seed(cfg["SEED"])
torch.cuda.manual_seed_all(cfg["SEED"])
np.random.seed(cfg["SEED"])
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(cfg["SEED"])


def competition_metric(true, pred):
    return f1_score(true, pred, average="macro")

def validation(model, criterion, test_loader, device):
    model.eval()

    model_preds = []
    true_labels = []

    val_loss = []

    with torch.no_grad():
        for img, label in tqdm(iter(test_loader)):
            img, label = img.float().to(device), label.type(torch.LongTensor).to(device)

            model_pred = model(img)

            loss = criterion(model_pred, label)

            val_loss.append(loss.item())

            model_preds += model_pred.argmax(1).detach().cpu().numpy().tolist()
            true_labels += label.detach().cpu().numpy().tolist()

    val_f1 = competition_metric(true_labels, model_preds)
    return np.mean(val_loss), val_f1

def train(mtype, model, ema, optimizer, criterion, train_loader, test_loader, scheduler, device):
    model.to(device)


    best_score = 0
    best_model = None

    for epoch in range(1, cfg[mtype]["EPOCHS"] + 1):
        model.train()
        train_loss = []
        for img, label in tqdm(iter(train_loader)):
            img, label = img.float().to(device), label.type(torch.LongTensor).to(device)

            optimizer.zero_grad()

            model_pred = model(img)
 
            loss = criterion(model_pred, label)  # use this loss for any training statistics
            loss.backward()
            optimizer.first_step(zero_grad=True)

            # second forward-backward pass
            criterion(model(img), label).backward()  # make sure to do a full forward pass
            optimizer.second_step(zero_grad=True)
            
            if ema is not None:
                pass

#             loss = criterion(model_pred, label)
#             loss.backward()
#             optimizer.step(closure)
#             ema.update()

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
            torch.save(best_model, f"exp/{last}/{mtype}/best.pt")

    return best_model


def exp_generator():
    exp = os.listdir("exp")
    exp = [x for x in exp if not x.startswith(".")]
    
    if not exp:
        os.mkdir("exp/0")
        os.mkdir("exp/0/age")
        os.mkdir("exp/0/gender")
        os.mkdir("exp/0/mask")
        last = 0
    else:
        last = list(map(int, exp))
        last.sort()
        last = last[-1]
        last+= 1
        
        os.mkdir(f"exp/{last}")
        os.mkdir(f"exp/{last}/age")
        os.mkdir(f"exp/{last}/gender")
        os.mkdir(f"exp/{last}/mask")
        
    return last


# import pprint
# pprint.pprint(timm.models.list_models())

dataset = HumanInfo()
train_transform = Train_Transform()
val_transform = Val_Transform()


train_age_dataset = Age_Dataset(dataset.train_age, train_transform.age)
train_age_loader = DataLoader(train_age_dataset, batch_size = cfg['age']['BATCH_SIZE'], shuffle=True, num_workers=0)
val_age_dataset = Age_Dataset(dataset.val_age, val_transform.age)
val_age_loader = DataLoader(val_age_dataset, batch_size=cfg['age']['BATCH_SIZE'], shuffle=False, num_workers=0)

train_gender_dataset = Gender_Dataset(dataset.train_gender, train_transform.gender)
train_gender_loader = DataLoader(train_gender_dataset, batch_size = cfg['gender']['BATCH_SIZE'], shuffle=True, num_workers=0)
val_gender_dataset = Gender_Dataset(dataset.val_gender, val_transform.gender)
val_gender_loader = DataLoader(val_gender_dataset, batch_size=cfg['gender']['BATCH_SIZE'], shuffle=False, num_workers=0)

train_mask_dataset = Mask_Dataset(dataset.train_mask, train_transform.mask)
train_mask_loader = DataLoader(train_mask_dataset, batch_size = cfg['mask']['BATCH_SIZE'], shuffle=True, num_workers=0)
val_mask_dataset = Mask_Dataset(dataset.val_mask, val_transform.mask)
val_mask_loader = DataLoader(val_mask_dataset, batch_size=cfg['mask']['BATCH_SIZE'], shuffle=False, num_workers=0)

last = exp_generator()
# last = 12

scheduler = None


print(">> Age Clasification -----------------------")
model = Age_Model(num_classes=train_age_dataset.num_classes)
# ema = EMA(model, beta = 0.9999, update_after_step = 100, update_every = 10)
ema = None
criterion = Focal_Loss(gamma = 2).to(device)
base_optimizer = torch.optim.SGD
optimizer = SAM(model.parameters(), base_optimizer, lr=cfg["LEARNING_RATE"], momentum=0.9, nesterov = True)
age_model = train("age", model, ema, optimizer, criterion, train_age_loader, val_age_loader, scheduler, device)
print("--------------------------------------------")
torch.cuda.empty_cache()

print(">> Gender Clasification -----------------------")
model = Gender_Model(num_classes=train_gender_dataset.num_classes)
# ema = EMA(model, beta = 0.9999, update_after_step = 100, update_every = 10)
ema = None
criterion = Focal_Loss(gamma = 2).to(device)
base_optimizer = torch.optim.SGD
optimizer = SAM(model.parameters(), base_optimizer, lr=cfg["LEARNING_RATE"], momentum=0.9, nesterov = True)
gender_model = train("gender", model, ema, optimizer, criterion, train_gender_loader, val_gender_loader, scheduler, device)
print("--------------------------------------------")
torch.cuda.empty_cache()


print(">> Mask Clasification -----------------------")
model = Mask_Model(num_classes=train_mask_dataset.num_classes)
# ema = EMA(model, beta = 0.9999, update_after_step = 100, update_every = 10)
ema = None
criterion = Focal_Loss(gamma = 2).to(device)
base_optimizer = torch.optim.SGD
optimizer = SAM(model.parameters(), base_optimizer, lr=cfg["LEARNING_RATE"], momentum=0.9, nesterov = True)
mask_model = train("mask", model, ema, optimizer, criterion, train_mask_loader, val_mask_loader, scheduler, device)
print("--------------------------------------------")
torch.cuda.empty_cache()