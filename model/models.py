import torch.nn as nn
import timm
import torchvision.models as models

class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super(BaseModel, self).__init__()
        self.backbone = models.mobilenet_v3_large(pretrained=True)
        # self.backbone = timm.create_model('swin_large_patch4_window7_224', pretrained=True,num_classes=num_classes).cuda()

    #         self.classifier1 = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        #         x = self.classifier(x)
        return x