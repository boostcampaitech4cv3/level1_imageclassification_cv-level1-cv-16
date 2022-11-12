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
        # self.backbone = models.mobilenet_v3_large(pretrained=True)

        self.backbone = timm.create_model('swin_large_patch4_window7_224', pretrained=True).cuda()

        self.fc1 = nn.Linear(1000, 256)
        self.dropout1 = nn.Dropout(0.4)
        self.classifier = nn.Linear(256, num_classes)
    def forward(self, x):
        x = self.backbone(x)
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.classifier(x)
        return x