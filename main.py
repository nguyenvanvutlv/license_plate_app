import cv2
import os
from utils.opencv import (
    camera,
    draw,
    fps
)
import time
from utils.lisence.detect import YoloNAS
from utils.lisence.detect import OCRLicense
import torch
torch.cuda.set_device(0)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class App:
    def __init__(self, video_source=0, window_name="lisence") -> None:
        self.video = camera.Camera(video_source)
        # self.video = cv2.VideoCapture(video_source)
        self.fps = fps.FPS()
        self.window_name = window_name

        self.yolonas = YoloNAS("model/lisence/ckpt_best.pth")
        self.yolonas.load()

        self.ocr = OCRLicense(weights="model/digits/ocr-net.weights",
                              netcfg="model/digits/ocr-net.cfg",
                              dataset="model/digits/ocr-net.data",
                              ocr_threshold=0.4)

        self.cropped = None
        self.text_ocr = ""
        self.frame = None

    def process(self, image):
        img = image.copy()
        _status, self.frame, self.cropped = self.yolonas.predict(img)
        if _status is not None:
            self.text_ocr = self.ocr.run(self.cropped)
            _h, _w = self.cropped.shape[:2]
            _ratio = max(_h, _w) / min(_h, _w)
            self.frame = draw.putText(
                self.frame, f'h: {_h}, w: {_w}, ratio: {round(_ratio, 2)}', (20, 50))
            self.frame = draw.putText(self.frame, f'Lisence: {self.text_ocr}', (20, 80))


    def run(self):
        self.video = self.video.start()
        try:
            while True:
                self.fps.start()
                self.frame = self.video.read()
                # res, self.frame = self.video.read()
                # if self.flip:
                if self.frame is not None:
                    # CODE HERE

                    self.process(self.frame)

                    # CODE HERE
                    self.fps.stop()
                    self.fps.update()
                    self.frame = draw.putText(
                        self.frame, f'FPS: {self.fps.fps()}', (20, 20))



                    cv2.imshow(self.window_name, self.frame)
                    if cv2.waitKey(1) & 0xFF == ord('x'):
                        break
                else:
                    break
        except KeyboardInterrupt:
            print("Interrupt")
        except RuntimeError as error:
            print(error)
        finally:
            self.video.stop()
            self.video.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    # app = App(video_source="/home/vu/Desktop/license_plate_app/data/test_3.mp4")
    app = App(video_source=0)
    app.run()
