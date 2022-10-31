import argparse
import multiprocessing
import os
from importlib import import_module

import pandas as pd
import torch
from torch.utils.data import DataLoader

from datasetcopy2 import TestDataset, BaseAugmentation, MaskDataset, MaskGenderDataset, IncorrectGenderDataset, NormalGenderDataset, MaskMaleDataset, IncorrectMaleDataset, NormalMaleDataset, MaskFemaleDataset, IncorrectFemaleDataset, NormalFemaleDataset



def load_model(saved_model, num_classes, device, model_type):
    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)
    
    if model_type == 'MaskModel':
        model_cls = getattr(import_module("modelcopy2"), 'MaskModel')
        model = model_cls(num_classes=num_classes)
                      
        model_path = os.path.join(saved_model, 'best_MaskModel.pth')
        model.load_state_dict(torch.load(model_path, map_location=device))
        
        #model_cls = getattr(import_module("model"), 'MaskModel')
        #model = model_cls(num_classes=num_classes)
        #model_path = os.path.join(saved_model, 'best_MaskModel.pth')
        #saved_checkpoint = torch.load(model_path)
        #model.load_state_dict(saved_checkpoint)
    elif model_type == 'MaskGenderModel':
        model_cls = getattr(import_module("modelcopy2"), 'MaskGenderModel')
        model = model_cls(num_classes=num_classes)
        
        model_path = os.path.join(saved_model, 'best_MaskGenderModel.pth')
        model.load_state_dict(torch.load(model_path, map_location=device))
        #model_cls = getattr(import_module("model"), 'MaskGenderModel')
        #model = model_cls(num_classes=num_classes)
        #model_path = os.path.join(saved_model, 'best_MaskGenderModel.pth')
        #saved_checkpoint = torch.load(model_path)
        #model.load_state_dict(saved_checkpoint)
    elif model_type == 'IncorrectGenderModel':
        model_cls = getattr(import_module("modelcopy2"), 'IncorrectGenderModel')
        model = model_cls(num_classes=num_classes)
        
        model_path = os.path.join(saved_model, 'best_IncorrectGenderModel.pth')
        model.load_state_dict(torch.load(model_path, map_location=device))
        #model_cls = getattr(import_module("model"), 'IncorrectGenderModel')
        #model = model_cls(num_classes=num_classes)
        #model_path = os.path.join(saved_model, 'best_IncorrectGenderModel.pth')
        #saved_checkpoint = torch.load(model_path)
        #model.load_state_dict(saved_checkpoint)
    elif model_type == 'NormalGenderModel':
        model_cls = getattr(import_module("modelcopy2"), 'NormalGenderModel')
        model = model_cls(num_classes=num_classes)
        
        model_path = os.path.join(saved_model, 'best_NormalGenderModel.pth')
        model.load_state_dict(torch.load(model_path, map_location=device))
        #model_cls = getattr(import_module("model"), 'NormalGenderModel')
        #model = model_cls(num_classes=num_classes)
        #model_path = os.path.join(saved_model, 'best_NormalGenderModel.pth')
        #saved_checkpoint = torch.load(model_path)
        #model.load_state_dict(saved_checkpoint)
    elif model_type == 'MaskMaleModel':
        model_cls = getattr(import_module("modelcopy2"), 'MaskMaleModel')
        model = model_cls(num_classes=num_classes)
        
        model_path = os.path.join(saved_model, 'best_MaskMaleModel.pth')
        model.load_state_dict(torch.load(model_path, map_location=device))
        #model_cls = getattr(import_module("model"), 'MaskMaleModel')
        #model = model_cls(num_classes=num_classes)
        #model_path = os.path.join(saved_model, 'best_MaskMaleModel.pth')
        #saved_checkpoint = torch.load(model_path)
        #model.load_state_dict(saved_checkpoint)
    elif model_type == 'IncorrectMaleModel':
        model_cls = getattr(import_module("modelcopy2"), 'IncorrectMaleModel')
        model = model_cls(num_classes=num_classes)
        
        model_path = os.path.join(saved_model, 'best_IncorrectMaleModel.pth')
        model.load_state_dict(torch.load(model_path, map_location=device))
        #model_cls = getattr(import_module("model"), 'IncorrectMaleModel')
        #model = model_cls(num_classes=num_classes)
        #model_path = os.path.join(saved_model, 'best_IncorrectMaleModel.pth')
        #saved_checkpoint = torch.load(model_path)
        #model.load_state_dict(saved_checkpoint)
    elif model_type == 'NormalMaleModel':
        model_cls = getattr(import_module("modelcopy2"), 'NormalMaleModel')
        model = model_cls(num_classes=num_classes)
        
        model_path = os.path.join(saved_model, 'best_NormalMaleModel.pth')
        model.load_state_dict(torch.load(model_path, map_location=device))
        #model_cls = getattr(import_module("model"), 'NormalMaleModel')
        #model = model_cls(num_classes=num_classes)
        #model_path = os.path.join(saved_model, 'best_NormalMaleModel.pth')
        #saved_checkpoint = torch.load(model_path)
        #model.load_state_dict(saved_checkpoint)
    elif model_type == 'MaskFemaleModel':
        model_cls = getattr(import_module("modelcopy2"), 'MaskFemaleModel')
        model = model_cls(num_classes=num_classes)
        
        model_path = os.path.join(saved_model, 'best_MaskFemaleModel.pth')
        model.load_state_dict(torch.load(model_path, map_location=device))
        #model_cls = getattr(import_module("model"), 'MaskFemaleModel')
        #model = model_cls(num_classes=num_classes)
        #model_path = os.path.join(saved_model, 'best_MaskFemaleModel.pth')
        #saved_checkpoint = torch.load(model_path)
        #model.load_state_dict(saved_checkpoint)
    elif model_type == 'IncorrectFemaleModel':
        model_cls = getattr(import_module("modelcopy2"), 'IncorrectFemaleModel')
        model = model_cls(num_classes=num_classes)
        
        model_path = os.path.join(saved_model, 'best_IncorrectFemaleModel.pth')
        model.load_state_dict(torch.load(model_path, map_location=device))
        #model_cls = getattr(import_module("model"), 'IncorrectFemaleModel')
        #model = model_cls(num_classes=num_classes)
        #model_path = os.path.join(saved_model, 'best_IncorrectFemaleModel.pth')
        #saved_checkpoint = torch.load(model_path)
        #model.load_state_dict(saved_checkpoint)
    elif model_type == 'NormalFemaleModel':
        model_cls = getattr(import_module("modelcopy2"), 'NormalFemaleModel')
        model = model_cls(num_classes=num_classes)
        
        model_path = os.path.join(saved_model, 'best_NormalFemaleModel.pth')
        model.load_state_dict(torch.load(model_path, map_location=device))
        #model_cls = getattr(import_module("model"), 'NormalFemaleModel')
        #model = model_cls(num_classes=num_classes)
        #model_path = os.path.join(saved_model, 'best_NormalFemaleModel.pth')
        #saved_checkpoint = torch.load(model_path)
        #model.load_state_dict(saved_checkpoint)
        
    return model


@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):
    """
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    num_classes = MaskDataset.num_classes  # 3
    mask_model = load_model(model_dir, num_classes, device, model_type='MaskModel').to(device)
    mask_model.eval()
    
    num_classes = MaskGenderDataset.num_classes  # 2
    mask_gender_model = load_model(model_dir, num_classes, device, model_type='MaskGenderModel').to(device)
    mask_gender_model.eval()
    
    num_classes = IncorrectGenderDataset.num_classes  # 2
    incorrect_gender_model = load_model(model_dir, num_classes, device, model_type='IncorrectGenderModel').to(device)
    incorrect_gender_model.eval()
    
    num_classes = NormalGenderDataset.num_classes  # 2
    normal_gender_model = load_model(model_dir, num_classes, device, model_type='NormalGenderModel').to(device)
    normal_gender_model.eval()
    
    num_classes = MaskMaleDataset.num_classes  # 3
    mask_male_model = load_model(model_dir, num_classes, device, model_type='MaskMaleModel').to(device)
    mask_male_model.eval()
    
    num_classes = IncorrectMaleDataset.num_classes  # 3
    incorrect_male_model = load_model(model_dir, num_classes, device, model_type='IncorrectMaleModel').to(device)
    incorrect_male_model.eval()
    
    num_classes = NormalMaleDataset.num_classes  # 3
    normal_male_model = load_model(model_dir, num_classes, device, model_type='NormalMaleModel').to(device)
    normal_male_model.eval()
    
    num_classes = MaskFemaleDataset.num_classes  # 3
    mask_female_model = load_model(model_dir, num_classes, device, model_type='MaskFemaleModel').to(device)
    mask_female_model.eval()
    
    num_classes = IncorrectFemaleDataset.num_classes  # 3
    incorrect_female_model = load_model(model_dir, num_classes, device, model_type='IncorrectFemaleModel').to(device)
    incorrect_female_model.eval()
    
    num_classes = NormalFemaleDataset.num_classes  # 3
    normal_female_model = load_model(model_dir, num_classes, device, model_type='NormalFemaleModel').to(device)
    normal_female_model.eval()
    

    img_root = os.path.join(data_dir, 'images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDataset(img_paths, args.resize)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
    preds = []
    with torch.no_grad():
        for idx, images in enumerate(loader):
            images = images.to(device)
            mask_pred = mask_model(images)
            mask_pred = mask_pred.argmax(dim=-1)
            if mask_pred.cpu().numpy() == 0:
                gender_pred = mask_gender_model(images)
                gender_pred = gender_pred.argmax(dim=-1)
                if gender_pred.cpu().numpy() == 0:
                    age_pred = mask_male_model(images)
                    age_pred = age_pred.argmax(dim=-1)
                elif gender_pred.cpu().numpy() == 1:
                    age_pred = mask_female_model(images)
                    age_pred = age_pred.argmax(dim=-1)
            elif mask_pred.cpu().numpy() == 1:
                gender_pred = incorrect_gender_model(images)
                gender_pred = gender_pred.argmax(dim=-1)
                if gender_pred.cpu().numpy() == 0:
                    age_pred = incorrect_male_model(images)
                    age_pred = age_pred.argmax(dim=-1)
                elif gender_pred.cpu().numpy() == 1:
                    age_pred = incorrect_female_model(images)
                    age_pred = age_pred.argmax(dim=-1)
            elif mask_pred.cpu().numpy() == 2:
                gender_pred = normal_gender_model(images)
                gender_pred = gender_pred.argmax(dim=-1) 
                if gender_pred.cpu().numpy() == 0:
                    age_pred = normal_male_model(images)
                    age_pred = age_pred.argmax(dim=-1)
                elif gender_pred.cpu().numpy() == 1:
                    age_pred = normal_female_model(images)
                    age_pred = age_pred.argmax(dim=-1)
                # print(idx, age_pred)
                
            print(idx, mask_pred, gender_pred, age_pred)
            pred = (mask_pred * 6) + (gender_pred * 3) + (age_pred)
            # print(idx, pred)
            preds.extend(pred.cpu().numpy())

    info['ans'] = preds
    save_path = os.path.join(output_dir, f'output.csv')
    info.to_csv(save_path, index=False)
    print(f"Inference Done! Inference result saved at {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size for validing (default: 1000)')
    parser.add_argument('--resize', type=tuple, default=(128, 96), help='resize size for image when you trained (default: (96, 128))')
    # parser.add_argument('--model', type=str, default='MaskModel', help='model type (default: MaskModel)')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model/exp47'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    inference(data_dir, model_dir, output_dir, args)
