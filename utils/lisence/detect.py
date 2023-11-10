import torch
from utils.opencv.draw import draw_boxes
from super_gradients.training import models
from super_gradients.common.object_names import Models


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
