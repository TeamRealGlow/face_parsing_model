import cv2
import os
import numpy as np
from skimage.filters import gaussian
from model.model import BiSeNet
import torch
import os
import os.path as osp
from PIL import Image
import torchvision.transforms as transforms
import cv2
def sharpen(img):
    img = img * 1.0
    gauss_out = gaussian(img, sigma=5, channel_axis=-1)

    alpha = 1.5
    img_out = (img - gauss_out) * alpha + img

    img_out = img_out / 255.0

    mask_1 = img_out < 0
    mask_2 = img_out > 1

    img_out = img_out * (1 - mask_1)
    img_out = img_out * (1 - mask_2) + mask_2
    img_out = np.clip(img_out, 0, 1)
    img_out = img_out * 255
    return np.array(img_out, dtype=np.uint8)


def hair(image, parsing, part=1, color=[230, 50, 20]):
    #colors = [[100, 200, 100]]
    b, g, r = color      #[10, 50, 250]       # [10, 250, 10]
    tar_color = np.zeros_like(image)#이미지와 같은 크기의 0 배열 생성
    tar_color[:, :, 0] = b
    tar_color[:, :, 1] = g
    tar_color[:, :, 2] = r

    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    tar_hsv = cv2.cvtColor(tar_color, cv2.COLOR_BGR2HSV)

    if part == 4:
        image_hsv[:, :, 0:2] = tar_hsv[:, :, 0:2]
    else:
        image_hsv[:, :, 0:1] = tar_hsv[:, :, 0:1]

    changed = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)

    if part == 1:
        changed = sharpen(changed)

    changed[parsing != part] = image[parsing != part]
    # changed = cv2.resize(changed, (512, 512))
    return changed


# def lip(image, parsing, part=4, color=[0, 0, 255]):
#     b, g, r = color      #[10, 50, 250]
#     # [10, 250, 10]
#     tar_color = np.zeros_like(image)
#     tar_color[:, :, 0] = b
#     tar_color[:, :, 1] = g
#     tar_color[:, :, 2] = r
#
#     image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
#     il, ia, ib = cv2.split(image_lab)
#
#     tar_lab = cv2.cvtColor(tar_color, cv2.COLOR_BGR2Lab)
#     tl, ta, tb = cv2.split(tar_lab)
#
#     image_lab[:, :, 0] = np.clip(il - np.mean(il) + tl, 0, 100)
#     image_lab[:, :, 1] = np.clip(ia - np.mean(ia) + ta, -127, 128)
#     image_lab[:, :, 2] = np.clip(ib - np.mean(ib) + tb, -127, 128)
#
#     changed = cv2.cvtColor(image_lab, cv2.COLOR_Lab2BGR)
#
#     if part == 4:
#         changed = sharpen(changed)
#
#     changed[parsing != part] = image[parsing != part]
#     # changed = cv2.resize(changed, (512, 512))
#     return changed


if __name__ == '__main__':
    table = {
        'hair': 1,
        'lip': 4,
    }
    n_classes = 9
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    cp = "18_iter_class9.pth"
    save_pth = osp.join('../res/cp', cp)
    net.load_state_dict(torch.load(save_pth))
    net.eval()
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    cap = cv2.VideoCapture(0)
    with torch.no_grad():
        if cap.isOpened():
            while True:
                ret, img = cap.read()
                if ret:
                    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    image = img.copy()
                    img = to_tensor(img)
                    img = torch.unsqueeze(img, 0)
                    img = img.cuda()
                    out = net(img)[0]
                    parsing = out.squeeze(0).cpu().numpy().argmax(0)
                    parts = [table['hair'], table['lip']]
                    # colors = [[20, 20, 200], [100, 100, 230], [100, 100, 230]]
                    colors = [[0, 0, 255]]
                    for part, color in zip(parts, colors):
                        image2 = hair(image, parsing, part, color)
                    cont = cv2.hconcat([image, image2])
                    cv2.imshow('pic', cont)
                k = cv2.waitKey(1)
                # ord('q'): input q exit
                if k == 27:  # esc
                    break
            cap.release()  # 캡쳐 자원 반납
            cv2.destroyAllWindows()















