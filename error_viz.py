from dataset.dataset import Viz_Dataset
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

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
        for img, label, path in tqdm(iter(custom_loader)):
            img = img.float().to(device)
            age, gender, mask = label
            age = age.long().to(device)
            gender = gender.long().to(device)
            mask = mask.long().to(device)
            
            age_pred, gender_pred, mask_pred = model(img)
            
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
                    path = "_".joinpath.split('/')[-3:]
                    path = path.rstrip(".jpg")
                    cv2.imwrite(f"Error/mask/{path}-{mp}.jpg", tmp)
            

#     model_preds = list(map(lambda preds: calc_ans(*preds), list(zip(age_preds, gender_preds, mask_preds))))
    return model_preds

def calc_ans(age, gender, mask):
    return mask*6 + gender*3 + age

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings(action='ignore')
    
    cfg = json.load(open("cfg.json", "r"))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = Ensemble(20) ## exp dir
    
    custom_dataset = Viz_Dataset(cfg)
    custom_loader = DataLoader(custom_dataset, batch_size=cfg['BATCH_SIZE'], shuffle=False, num_workers=0)
    
    preds = inference(model, custom_loader, device)
    
#     test_dataset.submit(preds)
