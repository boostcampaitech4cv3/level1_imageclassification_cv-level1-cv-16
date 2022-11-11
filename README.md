## Boostcamp AI Tech: CV-16
<table>
    <th><p align=center>팀명<p></th>
    <th colspan=5><p align=center>KKKimch</p></th>
    <tr>
        <td align=center>멘토</td>
        <td align=center colspan=5>
            <a href="https://github.com/animilux"><img height="120px" width="120px" src="https://avatars.githubusercontent.com/u/77085163?v=4"/></a>
            <br/>
            <a href="https://github.com/animilux"><strong>👨‍🔬 정용기</strong></a>
            <br />
        </td>
    </tr>
    <tr height="160px">
        <td align="center" width="150px">
            <p>구성원</p>
        </td>
        <td align="center" width="150px">
            <a href="https://github.com/kdoyoon"><img height="120px" width="120px" src="https://avatars.githubusercontent.com/u/93971443?s=100&v=4"/></a>
            <br/>
            <a href="https://github.com/kdoyoon"><strong>⚽ 김도윤</strong></a>
            <br />
        </td>
        <td align="center" width="150px">
            <a href="https://github.com/dbsgh431"><img height="120px" width="120px" src="https://avatars.githubusercontent.com/u/39187226?v=4"/></a>
            <br/>
            <a href="https://github.com/dbsgh431"><strong>🥵 김윤호</strong></a>
            <br />
        </td>
        <td align="center" width="150px">
            <a href="https://github.com/juye-ops"><img height="120px" width="120px" src="https://avatars.githubusercontent.com/u/103459155?v=4"/></a>
            <br/>
            <a href="https://github.com/juye-ops"><strong>☕ 김주엽</strong></a>
            <br />
        </td>
        <td align="center" width="150px">
            <a href="https://github.com/choipp"><img height="120px" width="120px" src="https://avatars.githubusercontent.com/u/103131249?v=4"/></a>
            <br />
            <a href="https://github.com/choipp"><strong>🎬 최동혁</strong></a>
            <br />
        </td>
        <td align="center" width="150px">
            <a href="https://github.com/soonyoung-hwang"><img height="120px" width="120px" src="https://avatars.githubusercontent.com/u/78343941?v=4"/></a>
            <br />
            <a href="https://github.com/soonyoung-hwang"><strong>🏸 황순영</strong></a>
            <br />
        </td>
    </tr>
</table>

## 실행 방법
### Train
```
$ python train.py
```

### Inference
```
$ python inference.py
```
## 설명
### 대회 결과
- Public
  - Val F1 Score: 0.7488
  - Accuracy: 80.6190%
- Private
  - Val F1 Score: 0.7229
  - Accuracy: 79.0635%

### 적용 기술
**Augmentations**
- Albumentations 라이브러리를 이용한 Augmentation 진행
- 나이
    |Augmentation|Parameter|Description|
    |:-:|:-:|:-:|
    |GrayScale||흑백 이미지 변환|
    |CenterCrop|384|중앙 영역 잘라내기|
    |Sharpening|alpha = (0.5, 1)|이미지 선명화|
    |Horizontal Flip|p = 0.3|p 확률에 따른 좌우 반전|
    |Normalize|mean = (0.485, 0.456, 0.406) <br> std = (0.229, 0.224, 0.225)|이미지 선명화|

- 성별
    |Augmentation|Parameter|Description|
    |:-:|:-:|:-:|
    |GrayScale||흑백 이미지 변환|
    |UnderCrop|img_size = 384|하단 중앙 영역 잘라내기|
    |Sharpening|alpha = (0.5, 1)|이미지 선명화|
    |Horizontal Flip|p = 0.3|p 확률에 따른 좌우 반전|
    |Normalize|mean = (0.485, 0.456, 0.406) <br> std = (0.229, 0.224, 0.225)|이미지 선명화|

- 마스크
  |Augmentation|Parameter|Description|
  |:-:|:-:|:-:|
  |GrayScale||흑백 이미지 변환|
  |RandomResizedCrop|img_size = 384 <br> scale = (0.8, 1.0)|중앙 영역 잘라내기|
  |Sharpening|alpha = (0.5, 1)|이미지 선명화|
  |Horizontal Flip|p = 0.3|p 확률에 따른 좌우 반전|
  |Normalize|mean = (0.485, 0.456, 0.406) <br> std = (0.229, 0.224, 0.225)|이미지 선명화|

**모델링: Ensemble**
- 나이, 성별, 마스크 분류 모델을 각각 학습하여 Inference 레벨에서 Ensemble 진행
- MobileNet V3를 통하여 신속한 모델링 완성
- S.O.T.A.의 Image Classification 정보를 바탕으로 Pretrained Model 선정
- EfficientNet V2, Swin, ViT 등의 모델 이용
- EfficientNet V2 Large
  - S.O.T.A.의 Cifar 100 데이터셋 기준 SAM Optimizer와 함께 2위에 등극

**학습 방법**
- 나이와 연령의 경우 2700명에 대해 각각 동일한 정보를 보유하므로 CSV의 각 Row에 해당하는 7장의 이미지 중 1장을 랜덤으로 선택하여 학습
- 마스크의 경우 2700명에 대해 7장의 이미지가 분류 대상이므로 모든 이미지$(2700*7=18900)$에 대해 학습
- Validation은 각 도메인에서 8:2로 적용
- 각 도메인에 대한 모델은 EfficientNet V2 Large 모델을 이용
- Hyper-Parameter
  - Loss Function: Focal with Label Smoothing
  - Optimizer: SAM
  - Learning rate: 0.01

**학습 결과**
- 나이
  - Val F1 Score: 86.4%
  - 대부분 범주의 경계에 있는 20대 후반, 50대 후반에서 오분류

- 성별
  - Val F1 Score: 99.3%
  - 머리카락 길이나 체형에 따라 오분류

- 마스크
  - Val F1 Score: 99.5%
  - 마스크의 경우, 대부분 입술이 보이면 이상 착용자도 미착용으로 오분류
