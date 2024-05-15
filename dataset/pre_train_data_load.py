
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import os.path as osp
import os
from PIL import Image
import numpy as np
import json
import cv2
import matplotlib.pyplot as plt

from transform import *
class pretrain_Facemask(Dataset):
    def __init__(self, rootpth, cropsize=(640, 480), mode='train', *args, **kwargs):
        super(pretrain_Facemask, self).__init__(*args, **kwargs)
        assert mode in ('train', 'val', 'test')
        self.mode = mode
        self.ignore_lb = 255
        self.rootpth = rootpth
        self.imgs = os.listdir(osp.join(self.rootpth,"image"))
        #  pre-processing
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        self.trans_train = Compose([
            ColorJitter(
                brightness=0.5,
                contrast=0.5,
                saturation=0.5),
            HorizontalFlip(),
            RandomScale((0.75, 1.0, 1.25, 1.5, 1.75, 2.0)),
            RandomCrop(cropsize)
            ])

    def __getitem__(self, idx):
        color_map = [
            (0, 0, 0),  # void
            (60, 0, 255),  # hair
            (51, 255, 255),  # brow
            (255, 0, 255),  # eye
            (255, 255, 0),  # lip
            (0, 255, 0),  # mouth
            (0, 153, 0),  # nose
            (255, 0, 0),  # skin
            (255, 204, 204)  # ear
        ]
        grayscale_map = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        impth = self.imgs[idx]
        img = Image.open(osp.join(self.rootpth,"image", impth))
        img = img.resize((512, 512), Image.BILINEAR)
        label = Image.open(osp.join(self.rootpth, 'mask', impth[:-3]+'png'))
        label = label.resize((512,512), Image.BILINEAR)
        canvas = np.zeros((512,512),dtype=np.uint8)
        for i, color in enumerate(color_map):
            # 매칭되는 픽셀 찾기
            matching_pixels = np.all(np.array(label) == np.array(color), axis=-1)
            canvas[np.where(matching_pixels)] += grayscale_map[i]
        label = Image.fromarray(canvas, mode='P')
        if self.mode == 'train':
            im_lb = dict(im=img, lb=label)
            im_lb = self.trans_train(im_lb)
            img, label = im_lb['im'], im_lb['lb']
        img = self.to_tensor(img)
        label = np.array(label).astype(np.int64)
        return img, label

    def __len__(self):
        return len(self.imgs)




