#!/usr/bin/python
# -*- encoding: utf-8 -*-

from model.model import BiSeNet
import torch
import os.path as osp
import numpy as np
import torchvision.transforms as transforms
import cv2

def vis_parsing_maps(im, parsing_anno, stride):
    part_colors = [
        [0, 0, 0],  # void
        [60, 0, 255],  # hair
        [51, 255, 255],  # brow
        [255, 0, 255],  # eye
        [255, 255, 0],  # lip
        [0, 255, 0],  # mouth
        [0, 153, 0],  # nose
        [255, 0, 0],  # skin
        [255, 204, 204]  # ear
    ]


    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)

    # Save result or not
    # if save_im:
    #     cv2.imwrite(save_path[:-4] +'.png', vis_parsing_anno)
    #     cv2.imwrite(save_path, vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    return vis_im
    # return vis_im

def evaluate(cp='model_final_diss.pth'):
    n_classes = 9
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    save_pth = osp.join('../res/cp', cp)
    net.load_state_dict(torch.load(save_pth))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 가로
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # 세로
    with torch.no_grad():
        if cap.isOpened():
            while True:
                ret, img = cap.read()
                if ret:
                    # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                    # img = cv2.resize(img,(512,512))
                    image = img.copy()
                    img = to_tensor(img)
                    img = torch.unsqueeze(img, 0)
                    img = img.cuda()
                    out = net(img)[0]
                    parsing = out.squeeze(0).cpu().numpy().argmax(0)
                    # input로 512, 512 입력

                    # print(parsing)
                    #print(np.unique(parsing))
                    pred = vis_parsing_maps(image, parsing, stride=1)

                    cv2.imshow('pic', pred)
                k = cv2.waitKey(1)
                # ord('q'): input q exit
                if k == 27:  # esc
                    break
            cap.release()  # 캡쳐 자원 반납
            cv2.destroyAllWindows()





if __name__ == "__main__":
    evaluate(cp='examplemodel.pth')


