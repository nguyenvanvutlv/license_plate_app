from ultralytics import YOLO
from utils.opencv.draw import draw_boxes
import torch


class ObjectDetection:
    def __init__(self, source, device) -> None:
        self.model = YOLO(source)
        self.device = device

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


class YOLOv8(ObjectDetection):
    def __init__(self, source, device) -> None:
        super().__init__(source, device)
        self.toDevice()

    def predict(self, image, conf=0.7):
        xyxys = []
        confs = 0
        results = self.model(image, verbose=False, conf=conf)
        for result in results:
            xyxy = list(result.boxes.xyxy.cpu().numpy().astype(int))
            conf = list(result.boxes.conf.cpu().numpy())
            if len(xyxy):
                xyxys = [xyxy[0]]
                confs = conf[0]

        return xyxys, confs

    def draw(self, origin, xyxys):
        image = origin.copy()
        image = draw_boxes(image, xyxys)
        return image

