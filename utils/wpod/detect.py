from model.wpod_process.utils import im2single
from model.wpod_process.keras_utils import load_model, detect_lp
import cv2


class Wpodnet:
    def __init__(self, source="model/wpod_model/wpod-net_update1.h5"):
        self.model = load_model(source)

    def predict(self, frame):
        ratio = float(max(frame.shape[:2])) / min(frame.shape[:2])
        side = int(ratio * 288.)
        bound_dim = min(side + (side % (2 ** 4)), 608)
        _, LlpImgs, _ = detect_lp(self.model, im2single(
            frame), bound_dim, 2 ** 4, (240, 80), 0.5)
        return LlpImgs

    def run(self, frame):
        expanded_image = cv2.copyMakeBorder(
            frame, 200, 200, 200, 200, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        results = self.predict(expanded_image)
        if len(results) > 0:
            return results[0]
        else:
            return frame
