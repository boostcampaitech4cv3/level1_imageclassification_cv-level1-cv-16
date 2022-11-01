import torch.nn as nn
import torch.nn.functional as F
from timm import create_model, list_models

# # type 1 : MetaFormer
# class BaseModel(nn.Module):
#     def __init__(self, num_classes):
#         super().__init__()
#         self.net = create_model('poolformer_m48', pretrained=True)
#         self.net.head = nn.Linear(self.net.head.in_features, num_classes)

#     def forward(self, x):
#         return self.net(x)


# type 2 : rexnet
class BaseModel(nn.Module): 
    def __init__(self, num_classes):
        super().__init__()
        self.net = create_model('rexnet_200', pretrained=True)
        self.net.head.fc = nn.Linear(self.net.head.fc.in_features, num_classes)

    def forward(self, x):
        return self.net(x)


# # type 3
# class BaseModel(nn.Module): 
#     def __init__(self, num_classes):
#         super().__init__()
#         self.net = create_model('tf_efficientnet_b7_ns', pretrained=True)
#         self.net.classifier = nn.Linear(self.net.classifier.in_features, num_classes)
#         # self.net.head.fc = nn.Linear(self.net.head.fc.in_features, num_classes)

#     def forward(self, x):
#         return self.net(x)

# a = BaseModel(18)
# print(a)


# Custom Model Template

# print(list_models('*',pretrained=True))
# class MyModel(nn.Module):
#     def __init__(self, num_classes):
#         super().__init__()

#         """
#         1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
#         2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
#         3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
#         """

#     def forward(self, x):
#         """
#         1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
#         2. 결과로 나온 output 을 return 해주세요
#         """
#         return x
