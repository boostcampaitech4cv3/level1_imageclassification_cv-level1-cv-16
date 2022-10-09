import torch
import torch.nn as nn
import timm
import torchvision.models as models

class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super(BaseModel, self).__init__()
        ### 모델 불러온 후 학습 추가 진행
        # self.backbone = torch.load("model/model.pt")

        self.backbone = models.efficientnet_v2_l(pretrained=True)# 23퍼 주변
        self.classifier = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x