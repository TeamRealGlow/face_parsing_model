from model.model import BiSeNet
import torch
import os.path as osp
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
class Parser():
    def __init__(self,cp):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        """
        :param cp:
        input model path
        """
        self.model = BiSeNet(n_classes=9)
        save_pth = osp.join('..','pre_train', cp)
        self.model.load_state_dict(torch.load(save_pth))
        self.model = self.model.to(self.device)
        self.model.eval()
    def _preprocessing(self,imgPath):
        to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        img = Image.open(imgPath)
        w, h = (img.width, img.height)
        image = img.resize((512, 512), Image.BILINEAR)
        image = to_tensor(image)
        image = torch.unsqueeze(image, 0)
        return image,w,h
    def _make_face_seg(self,image,w,h):
        with torch.no_grad():
            image = image.to(self.device)
            y_pred = self.model(image)[0]
            parsing = y_pred.squeeze(0).cpu().numpy().argmax(0)
            parsingimage = Image.fromarray((parsing).astype(np.uint8))
            parsingimage = parsingimage.resize((w, h))
            return parsingimage

    def out_parsing(self, imgPath):
        image, w, h = self._preprocessing(imgPath)
        parsing = self._make_face_seg(image, w, h)
        return parsing

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    a = Parser(osp.join("best_model","examplemodel.pth"))
    parsing = a.out_parsing(osp.join("..","example","65.jpg"))
    plt.imshow(parsing)
    plt.show()