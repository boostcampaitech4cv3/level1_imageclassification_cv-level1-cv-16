import albumentations as A
import cv2
import numpy as np
import random
import pandas as pd

from glob import glob
from albumentations.core.transforms_interface import ImageOnlyTransform
from matplotlib import pyplot as plt


def undercrop(image, **kwargs):
    h, w, c = image.shape
    x_start = w - 384
    y_start = -384
    
    
    return image[y_start:, x_start:x_start+384]

if __name__ == "__main__":
    import random
    from glob import glob
    
    file_dir = "../../input/data/train/images/"
    files = glob(f"{file_dir}*/*")
    file = random.choice(files)
    
    img = cv2.imread(file)
    cv2.imwrite("원본.jpg", img)
    
    transform = A.Compose([
        A.Lambda(image=undercrop)
    ])
    img = transform(image=img)["image"]
    print(img.shape)
    cv2.imwrite("sample.jpg", img)