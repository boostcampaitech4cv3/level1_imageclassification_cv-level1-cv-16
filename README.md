## Boostcamp AI Tech: CV-16
<table>
    <th><p align=center>νλͺ<p></th>
    <th colspan=5><p align=center>KKKimch</p></th>
    <tr>
        <td align=center>λ©ν </td>
        <td align=center colspan=5>
            <a href="https://github.com/animilux"><img height="120px" width="120px" src="https://avatars.githubusercontent.com/u/77085163?v=4"/></a>
            <br/>
            <a href="https://github.com/animilux"><strong>π¨βπ¬ μ μ©κΈ°</strong></a>
            <br />
        </td>
    </tr>
    <tr height="160px">
        <td align="center" width="150px">
            <p>κ΅¬μ±μ</p>
        </td>
        <td align="center" width="150px">
            <a href="https://github.com/kdoyoon"><img height="120px" width="120px" src="https://avatars.githubusercontent.com/u/93971443?s=100&v=4"/></a>
            <br/>
            <a href="https://github.com/kdoyoon"><strong>β½ κΉλμ€</strong></a>
            <br />
        </td>
        <td align="center" width="150px">
            <a href="https://github.com/dbsgh431"><img height="120px" width="120px" src="https://avatars.githubusercontent.com/u/39187226?v=4"/></a>
            <br/>
            <a href="https://github.com/dbsgh431"><strong>π₯΅ κΉμ€νΈ</strong></a>
            <br />
        </td>
        <td align="center" width="150px">
            <a href="https://github.com/juye-ops"><img height="120px" width="120px" src="https://avatars.githubusercontent.com/u/103459155?v=4"/></a>
            <br/>
            <a href="https://github.com/juye-ops"><strong>β κΉμ£Όμ½</strong></a>
            <br />
        </td>
        <td align="center" width="150px">
            <a href="https://github.com/choipp"><img height="120px" width="120px" src="https://avatars.githubusercontent.com/u/103131249?v=4"/></a>
            <br />
            <a href="https://github.com/choipp"><strong>π¬ μ΅λν</strong></a>
            <br />
        </td>
        <td align="center" width="150px">
            <a href="https://github.com/soonyoung-hwang"><img height="120px" width="120px" src="https://avatars.githubusercontent.com/u/78343941?v=4"/></a>
            <br />
            <a href="https://github.com/soonyoung-hwang"><strong>πΈ ν©μμ</strong></a>
            <br />
        </td>
    </tr>
</table>

## μ€ν λ°©λ²
### Train
```
$ python train.py
```

### Inference
```
$ python inference.py
```
## μ€λͺ
### λν κ²°κ³Ό
- Public
  - Val F1 Score: 0.7488
  - Accuracy: 80.6190%
- Private
  - Val F1 Score: 0.7229
  - Accuracy: 79.0635%

### μ μ© κΈ°μ 
**Augmentations**
- Albumentations λΌμ΄λΈλ¬λ¦¬λ₯Ό μ΄μ©ν Augmentation μ§ν
- λμ΄
    |Augmentation|Parameter|Description|
    |:-:|:-:|:-:|
    |GrayScale||νλ°± μ΄λ―Έμ§ λ³ν|
    |CenterCrop|384|μ€μ μμ­ μλΌλ΄κΈ°|
    |Sharpening|alpha = (0.5, 1)|μ΄λ―Έμ§ μ λͺν|
    |Horizontal Flip|p = 0.3|p νλ₯ μ λ°λ₯Έ μ’μ° λ°μ |
    |Normalize|mean = (0.485, 0.456, 0.406) <br> std = (0.229, 0.224, 0.225)|μ΄λ―Έμ§ μ λͺν|

- μ±λ³
    |Augmentation|Parameter|Description|
    |:-:|:-:|:-:|
    |GrayScale||νλ°± μ΄λ―Έμ§ λ³ν|
    |UnderCrop|img_size = 384|νλ¨ μ€μ μμ­ μλΌλ΄κΈ°|
    |Sharpening|alpha = (0.5, 1)|μ΄λ―Έμ§ μ λͺν|
    |Horizontal Flip|p = 0.3|p νλ₯ μ λ°λ₯Έ μ’μ° λ°μ |
    |Normalize|mean = (0.485, 0.456, 0.406) <br> std = (0.229, 0.224, 0.225)|μ΄λ―Έμ§ μ λͺν|

- λ§μ€ν¬
  |Augmentation|Parameter|Description|
  |:-:|:-:|:-:|
  |GrayScale||νλ°± μ΄λ―Έμ§ λ³ν|
  |RandomResizedCrop|img_size = 384 <br> scale = (0.8, 1.0)|μ€μ μμ­ μλΌλ΄κΈ°|
  |Sharpening|alpha = (0.5, 1)|μ΄λ―Έμ§ μ λͺν|
  |Horizontal Flip|p = 0.3|p νλ₯ μ λ°λ₯Έ μ’μ° λ°μ |
  |Normalize|mean = (0.485, 0.456, 0.406) <br> std = (0.229, 0.224, 0.225)|μ΄λ―Έμ§ μ λͺν|

**λͺ¨λΈλ§: Ensemble**
- λμ΄, μ±λ³, λ§μ€ν¬ λΆλ₯ λͺ¨λΈμ κ°κ° νμ΅νμ¬ Inference λ λ²¨μμ Ensemble μ§ν
- MobileNet V3λ₯Ό ν΅νμ¬ μ μν λͺ¨λΈλ§ μμ±
- S.O.T.A.μ Image Classification μ λ³΄λ₯Ό λ°νμΌλ‘ Pretrained Model μ μ 
- EfficientNet V2, Swin, ViT λ±μ λͺ¨λΈ μ΄μ©
- EfficientNet V2 Large
  - S.O.T.A.μ Cifar 100 λ°μ΄ν°μ κΈ°μ€ SAM Optimizerμ ν¨κ» 2μμ λ±κ·Ή

**νμ΅ λ°©λ²**
- λμ΄μ μ°λ Ήμ κ²½μ° 2700λͺμ λν΄ κ°κ° λμΌν μ λ³΄λ₯Ό λ³΄μ νλ―λ‘ CSVμ κ° Rowμ ν΄λΉνλ 7μ₯μ μ΄λ―Έμ§ μ€ 1μ₯μ λλ€μΌλ‘ μ ννμ¬ νμ΅
- λ§μ€ν¬μ κ²½μ° 2700λͺμ λν΄ 7μ₯μ μ΄λ―Έμ§κ° λΆλ₯ λμμ΄λ―λ‘ λͺ¨λ  μ΄λ―Έμ§$(2700*7=18900)$μ λν΄ νμ΅
- Validationμ κ° λλ©μΈμμ 8:2λ‘ μ μ©
- κ° λλ©μΈμ λν λͺ¨λΈμ EfficientNet V2 Large λͺ¨λΈμ μ΄μ©
- Hyper-Parameter
  - Loss Function: Focal with Label Smoothing
  - Optimizer: SAM
  - Learning rate: 0.01

**νμ΅ κ²°κ³Ό**
- λμ΄
  - Val F1 Score: 86.4%
  - λλΆλΆ λ²μ£Όμ κ²½κ³μ μλ 20λ νλ°, 50λ νλ°μμ μ€λΆλ₯

- μ±λ³
  - Val F1 Score: 99.3%
  - λ¨Έλ¦¬μΉ΄λ½ κΈΈμ΄λ μ²΄νμ λ°λΌ μ€λΆλ₯

- λ§μ€ν¬
  - Val F1 Score: 99.5%
  - λ§μ€ν¬μ κ²½μ°, λλΆλΆ μμ μ΄ λ³΄μ΄λ©΄ μ΄μ μ°©μ©μλ λ―Έμ°©μ©μΌλ‘ μ€λΆλ₯
