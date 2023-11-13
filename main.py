import paddle
import cv2
import os
import imutils
import numpy as np
import matplotlib.pyplot as plt
from utils.opencv import (
    camera,
    draw,
    fps
)
import time
from utils.lisence.detect import YoloNAS
from utils.wpod.detect import Wpodnet
from utils.lisence.detect import OCRLicense
import torch
torch.cuda.set_device(0)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class App:
    def __init__(self, video_source=0, window_name="lisence", flip=False) -> None:
        # self.video = camera.Camera(video_source)
        self.video = cv2.VideoCapture(video_source)
        self.fps = fps.FPS()
        self.window_name = window_name
        self.flip = flip
        self.max_fps = 144

        self.yolonas = YoloNAS("model/lisence/ckpt_best.pth")
        self.yolonas.load()
        self.wpodnet = Wpodnet()
        self.width = 1024
        self.height = 720

        self.ocr = OCRLicense(weights="model/digits/ocr-net.weights",
                              netcfg="model/digits/ocr-net.cfg",
                              dataset="model/digits/ocr-net.data",
                              ocr_threshold=0.4)

        self.cropped = None
        self.output_wpod = None
        self.thresh = None

    def process(self, image):
        img = image.copy()
        # img = imutils.resize(img, self.width, self.height)
        _status, frame, _croped = self.yolonas.predict(img)
        if _status is None:
            return
        self.cropped = _croped
        self.output_wpod = self.wpodnet.run(
            self.cropped).astype(np.float32) * 255
        print(np.max(self.output_wpod), np.min(self.output_wpod))

        return _status

    def run(self):
        # self.video = self.video.start()
        try:
            while True:
                self.fps.start()
                # frame = self.video.read()
                res, frame = self.video.read()
                # if self.flip:
                # frame = cv2.flip(frame, 1)
                if frame is not None:
                    # CODE HERE

                    _status = self.process(frame)
                    if _status is not None:
                        self.ocr.run(self.output_wpod)

                    # CODE HERE
                    self.fps.stop()
                    self.fps.update()
                    frame = draw.putText(
                        frame, f'FPS: {self.fps.fps()}', (20, 20))
                    cv2.imshow(self.window_name, frame)
                    if self.thresh is not None:
                        cv2.imshow("thresh", self.thresh)

                    time.sleep(1 / self.max_fps)
                    if cv2.waitKey(1) & 0xFF == ord('x'):
                        break
                else:
                    break
        except KeyboardInterrupt:
            print("Interrupt")
        except RuntimeError as error:
            print(error)
        finally:
            # self.video.stop()
            self.video.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    app = App(video_source="data/test.mp4", flip=True)
    app.run()
