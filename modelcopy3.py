import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)
    
    
class MaskModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.efficientnet_v2_l(pretrained=True)

        self.fc1 = nn.Linear(1000, 256)
        self.dropout1 = nn.Dropout(0.4)
        self.age_classifier = nn.Linear(256, num_classes)
        self.gender_classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.age_classifier(x)
        
        return x

    
    
class MaskGenderModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.backbone = models.efficientnet_v2_l(pretrained=True)

        self.fc1 = nn.Linear(1000, 256)
        self.dropout1 = nn.Dropout(0.4)
        self.age_classifier = nn.Linear(256, num_classes)
        self.gender_classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.age_classifier(x)
        
        return x

    
class IncorrectGenderModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.backbone = models.efficientnet_v2_l(pretrained=True)

        self.fc1 = nn.Linear(1000, 256)
        self.dropout1 = nn.Dropout(0.4)
        self.age_classifier = nn.Linear(256, num_classes)
        self.gender_classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.age_classifier(x)
        
        return x


class NormalGenderModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.backbone = models.efficientnet_v2_l(pretrained=True)

        self.fc1 = nn.Linear(1000, 256)
        self.dropout1 = nn.Dropout(0.4)
        self.age_classifier = nn.Linear(256, num_classes)
        self.gender_classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.age_classifier(x)
        
        return x


class MaskMaleModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.backbone = models.efficientnet_v2_l(pretrained=True)

        self.fc1 = nn.Linear(1000, 256)
        self.dropout1 = nn.Dropout(0.4)
        self.age_classifier = nn.Linear(256, num_classes)
        self.gender_classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.age_classifier(x)
        
        return x

"""    
class IncorrectMaleModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.backbone = models.efficientnet_v2_l(pretrained=True)

        self.fc1 = nn.Linear(1000, 256)
        self.dropout1 = nn.Dropout(0.4)
        self.age_classifier = nn.Linear(256, num_classes)
        self.gender_classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.age_classifier(x)
        
        return x

    
class NormalMaleModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.backbone = models.efficientnet_v2_l(pretrained=True)

        self.fc1 = nn.Linear(1000, 256)
        self.dropout1 = nn.Dropout(0.4)
        self.age_classifier = nn.Linear(256, num_classes)
        self.gender_classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.age_classifier(x)
        
        return x

    
class MaskFemaleModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.backbone = models.efficientnet_v2_l(pretrained=True)

        self.fc1 = nn.Linear(1000, 256)
        self.dropout1 = nn.Dropout(0.4)
        self.age_classifier = nn.Linear(256, num_classes)
        self.gender_classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.age_classifier(x)
        
        return x

    
class IncorrectFemaleModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.backbone = models.efficientnet_v2_l(pretrained=True)

        self.fc1 = nn.Linear(1000, 256)
        self.dropout1 = nn.Dropout(0.4)
        self.age_classifier = nn.Linear(256, num_classes)
        self.gender_classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.age_classifier(x)
        
        return x

    
class NormalFemaleModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.backbone = models.efficientnet_v2_l(pretrained=True)

        self.fc1 = nn.Linear(1000, 256)
        self.dropout1 = nn.Dropout(0.4)
        self.age_classifier = nn.Linear(256, num_classes)
        self.gender_classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.age_classifier(x)
        
        return x
"""