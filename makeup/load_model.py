import os.path

import numpy as np
from skimage.filters import gaussian
from model.model import BiSeNet
import torch
import torchvision.transforms as transforms
import cv2
import os.path as osp


class model():
    def __init__(self,cp):
        self.model = BiSeNet(n_classes=9)
        save_pth = osp.join('pre_train', cp)
        self.model.load_state_dict(torch.load(save_pth))
        print(torch.cuda.is_available())

a = model(osp.join("best_model","전이학습_best모델.pth"))
