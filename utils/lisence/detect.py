import torch
from utils.opencv.draw import draw_boxes
from super_gradients.training import models
from super_gradients.common.object_names import Models
import utils.lisence.darknet as dn
from utils.lisence.darknet import detect
from model.wpod_process.label import dknet_label_conversion
from model.wpod_process.utils import nms


class ObjectDetection:
    def __init__(self, source) -> None:
        self.source = source
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def predict(self, image, conf=0.7):
        """_summary_
        Args:
            image (_type_): numpy array
        Returns:
            bboxes  : bounding box xyxy of lisence plate
            confidences: conf foe each lisence
        """
        prediction = self.model.predict(
            image, fuse_model=False, conf=conf)[0].prediction
        bboxes = prediction.bboxes_xyxy.astype(int)
        confidences = prediction.confidence.astype(float)
        return bboxes, confidences

    def load(self):
        pass

    def toDevice(self):
        self.model = self.model.to(self.device)


class YoloNAS(ObjectDetection):
    def __init__(self, source) -> None:
        super().__init__(source)

    def load(self):
        self.model = models.get('yolo_nas_l',
                                num_classes=1,
                                checkpoint_path=self.source)
        self.toDevice()

    def predict(self, image):
        bboxes, confs = super().predict(image)
        if confs.shape[0] > 0:
            pos = None
            conf = 0
            for index, value in enumerate(confs):
                if value > conf:
                    conf = value
                    pos = bboxes[index]
            if pos is None:
                return None, image, None
            image_draw = image.copy()
            image_draw = draw_boxes(image_draw, [pos])

            # pos0, pos1, pos2, pos3
            #   x1,   y1,   x2,   y2
            croped = image.copy()
            croped = croped[pos[1]: pos[3], pos[0]: pos[2]]
            return 1, image_draw, croped
        else:
            return None, image, None


class OCRLicense:
    def __init__(self, weights, netcfg, dataset, ocr_threshold) -> None:
        ocr_weights = weights.encode('utf-8')
        ocr_netcfg = netcfg.encode('utf-8')
        ocr_dataset = dataset.encode('utf-8')

        self.ocr_net = dn.load_net(ocr_netcfg, ocr_weights, 0)
        self.ocr_meta = dn.load_meta(ocr_dataset)
        self.ocr_threshold = ocr_threshold

    def run(self, img):
        R, (_w, _h) = detect(self.ocr_net, self.ocr_meta,
                             img, thresh=self.ocr_threshold, nms=None)

        # print(R, _w, _h)
        if len(R):
            L = dknet_label_conversion(R, _w, _h)
            L = nms(L, 0.45)
            L.sort(key=lambda x: x.tl()[0])
            lp_str = ''.join([chr(l.cl()) for l in L])
            return lp_str
        return "NONE"
