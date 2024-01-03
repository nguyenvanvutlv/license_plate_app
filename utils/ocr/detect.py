import torch
from utils.ocr.net import OCRnet
import cv2
import numpy as np


class OCRLicence:
    def __init__(self, weight, device) -> None:
        self.plateName = r"                                          0123456789ABCDEFGHJKLMNPQRSTUVWXYZ  "
        self.mean_value, self.std_value = (0.588, 0.193)
        check_point = torch.load(weight, map_location=device)
        model_state = check_point['state_dict']
        cfg = check_point['cfg']
        color_classes = 5
        self.model = OCRnet(num_classes=len(self.plateName),
                            export=True, cfg=cfg, color_num=color_classes)

        self.model.load_state_dict(model_state, strict=False)
        self.model.to(device)
        self.model.eval()
        self.device = device

    def image_processing(self, img, device):
        img = cv2.resize(img, (168, 48))
        img = np.reshape(img, (48, 168, 3))

        # normalize
        img = img.astype(np.float32)
        img = (img / 255. - self.mean_value) / self.std_value
        img = img.transpose([2, 0, 1])
        img = torch.from_numpy(img)

        img = img.to(device)
        img = img.view(1, *img.size())
        return img

    def decodePlate(self, preds):
        pre = 0
        newPreds = []
        index = []
        for i in range(len(preds)):
            if preds[i] != 0 and preds[i] != pre:
                newPreds.append(preds[i])
                index.append(i)
            pre = preds[i]
        return newPreds, index

    def run(self, image):
        image = self.image_processing(image, self.device)
        preds, color_preds = self.model(image)
        color_preds = torch.softmax(color_preds, dim=-1)
        color_conf, color_index = torch.max(color_preds, dim=-1)
        color_conf = color_conf.item()
        preds = torch.softmax(preds, dim=-1)
        prob, index = preds.max(dim=-1)
        index = index.view(-1).detach().cpu().numpy()
        prob = prob.view(-1).detach().cpu().numpy()

        newPreds, new_index = self.decodePlate(index)
        prob = prob[new_index]
        plate = ""
        for i in newPreds:
            plate += self.plateName[i]
        return plate
