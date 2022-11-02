from dataset.dataset import Viz_Dataset
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from dataset.dataset import *
from dataset.transformation import *

from model.models import *
import json
import cv2
import numpy as np


def inference(model, custom_loader, device):
    model.to(device)
    model.eval()

    age_preds = []
    gender_preds = []
    mask_preds = []

    with torch.no_grad():
        for age_img, gender_img, mask_img, label, path in tqdm(iter(custom_loader)):
            age_img = age_img.float().to(device)
            gender_img = gender_img.float().to(device)
            mask_img = mask_img.float().to(device)
            
            age, gender, mask = label
            age = age.long().to(device)
            gender = gender.long().to(device)
            mask = mask.long().to(device)
            
            age_pred, gender_pred, mask_pred = model(age_img, gender_img, mask_img)
            
            age_pred = age_pred.argmax(1).detach().cpu().numpy().tolist()
            gender_pred = gender_pred.argmax(1).detach().cpu().numpy().tolist()
            mask_pred = mask_pred.argmax(1).detach().cpu().numpy().tolist()
            for i, (a, ap, g, gp, m, mp, path) in enumerate(zip(age, age_pred, gender, gender_pred, mask, mask_pred, path)):
                if a!=ap:
                    tmp = cv2.imread(path)
                    cv2.imwrite(f"Error/age/{path.split('/')[-3]}-{ap}.jpg", tmp)
                    
                if g!=gp:
                    tmp = cv2.imread(path)
                    cv2.imwrite(f"Error/gender/{path.split('/')[-3]}-{gp}.jpg", tmp)
                    
                if m!=mp:
                    tmp = cv2.imread(path)
                    path = "_".join(path.split('/')[-3:])
                    path = path.rstrip(".jpg")
                    cv2.imwrite(f"Error/mask/{path}-{mp}.jpg", tmp)
            



def calc_ans(age, gender, mask):
    return mask*6 + gender*3 + age

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings(action='ignore')
    
    cfg = json.load(open("cfg.json", "r"))
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    dataset = HumanInfo()
    val_transform = Val_Transform()
    
    df = pd.read_csv("/opt/ml/input/data/train/train.csv")
    
    viz_dataset = Viz_Dataset(df, [val_transform.age, val_transform.gender, val_transform.mask])
    viz_loader = DataLoader(viz_dataset, batch_size=cfg['test']['BATCH_SIZE'], shuffle=False, num_workers=0)
    
    model = Ensemble(2) ## exp dir
    inference(model, viz_loader, device)
    
#     test_dataset.submit(preds)
