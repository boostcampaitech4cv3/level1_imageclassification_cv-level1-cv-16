import torch
import torch.nn as nn
import timm
import torchvision.models as models

class Age_Model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
#         self.backbone = models.mobilenet_v3_large(pretrained=True)
        self.backbone = models.efficientnet_v2_l(pretrained=True)
    
        self.fc1 = nn.Linear(1000, 256)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 43)  # age
        self.dropout2 = nn.Dropout(0.3)
        self.age_classifier = nn.Linear(43, num_classes)  # age split

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.age_classifier(x)
        
        return x
    
class Gender_Model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.mobilenet_v3_large(pretrained=True)
#         self.backbone = models.efficientnet_v2_l(pretrained=True)

        self.fc1 = nn.Linear(1000, 256)
        self.dropout1 = nn.Dropout(0.4)
        self.gender_classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.gender_classifier(x)
        
        return x

class Mask_Model(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.mobilenet_v3_large(pretrained=True)
#         self.backbone = models.efficientnet_v2_l(pretrained=True)

        self.fc1 = nn.Linear(1000, 256)
        self.dropout1 = nn.Dropout(0.4)
        self.mask_classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.mask_classifier(x)
        
        return x

    
class Ensemble(nn.Module):
    def __init__(self, exp):
        super().__init__()
        self.age = torch.load(f"exp/{exp}/age/best.pt")
        self.gender = torch.load(f"exp/{exp}/gender/best.pt")
        self.mask = torch.load(f"exp/{exp}/mask/best.pt")
        
    def forward(self, x, y, z):
        age = self.age(x)
        gender = self.gender(y)
        mask = self.mask(z)
        
        return age, gender, mask
    
if __name__ == "__main__":
    model = Age_Model(3)