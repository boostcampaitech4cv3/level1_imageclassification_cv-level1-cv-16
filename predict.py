from dataset.dataset import TestDataset, CustomDataset
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

from model.models import *
import json


def inference(model, test_loader, device):
    model.to(device)
    model.eval()

    age_preds = []
    gender_preds = []
    mask_preds = []

    with torch.no_grad():
        for age_img, gender_img, mask_img in tqdm(iter(test_loader)):
            age_img = age_img.float().to(device)
            gender_img = gender_img.float().to(device)
            mask_img = mask_img.float().to(device)

            age_pred, gender_pred, mask_pred = model(age_img, gender_img, mask_img)
            
            age_preds += age_pred.argmax(1).detach().cpu().numpy().tolist()
            gender_preds += gender_pred.argmax(1).detach().cpu().numpy().tolist()
            mask_preds += mask_pred.argmax(1).detach().cpu().numpy().tolist()

    model_preds = list(map(lambda preds: calc_ans(*preds), list(zip(age_preds, gender_preds, mask_preds))))
            
    print('Done.')
    return model_preds

def competition_metric(true, pred):
    return f1_score(true, pred, average="macro")


def calc_ans(age, gender, mask):
    return mask*6 + gender*3 + age

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings(action='ignore')
    
    cfg = json.load(open("cfg.json", "r"))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model = Ensemble(15) ## exp dir
    
    test_dataset = TestDataset()
    test_loader = DataLoader(test_dataset, batch_size=cfg['test']['BATCH_SIZE'], shuffle=False, num_workers=0)

    preds = inference(model, test_loader, device)
    
    test_dataset.submit(preds)