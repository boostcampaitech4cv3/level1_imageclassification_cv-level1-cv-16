import torch
import torch.nn as nn
import timm
import torchvision.models as models

class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super(BaseModel, self).__init__()
        ### 모델 불러온 후 학습 추가 진행
        # self.backbone = torch.load("model/model.pt")

        # self.backbone = models.efficientnet_v2_l(pretrained=True)# 23퍼 주변
        self.backbone = models.mobilenet_v3_large(pretrained=True)

        # self.backbone = timm.create_model('swin_large_patch4_window7_224', pretrained=True).cuda()
        # self.backbone = timm.create_model('swinv2_large_window12_192_22k', pretrained=True,num_classes=num_classes).cuda()

        self.fc1 = nn.Linear(1000, 512)
        self.dropout1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(256, 128)
        self.dropout3 = nn.Dropout(0.2)
        self.classifier = nn.Linear(128, num_classes)
    def forward(self, x):
        x = self.backbone(x)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.dropout3(x)
        x = self.classifier(x)
        return x