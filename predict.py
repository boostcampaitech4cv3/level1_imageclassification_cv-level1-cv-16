from sklearn.metrics import f1_score
from tqdm.auto import tqdm
import torch
import numpy as np

from dataloader.dataloaders import *
from model.models import *
from predict import *
from cfg import *
from data.dataset import *

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


def inference(model, test_loader, device):
    model.to(device)
    model.eval()

    model_preds = []

    with torch.no_grad():
        for img in tqdm(iter(test_loader)):
            img = img.float().to(device)

            model_pred = model(img)
            model_preds += model_pred.argmax(1).detach().cpu().numpy().tolist()

    print('Done.')
    return model_preds

if __name__ == "__main__":
    from cfg import *
    import warnings

    warnings.filterwarnings(action='ignore')

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    CFG = load_cfg()
    model = torch.load("model/model.pt")
    artists = Artists(CFG)

    test_dataset = CustomDataset(artists.test_img_paths, None, artists.test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

    preds = inference(model, test_loader, device)
    preds = artists.le.inverse_transform(preds)  # LabelEncoder로 변환 된 Label을 다시 화가이름으로 변환

    submit = pd.read_csv('./sample_submission.csv')
    submit['artist'] = preds
    submit.to_csv('./submit.csv', index=False)