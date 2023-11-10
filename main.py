import cv2
import os
from utils.opencv import (
    camera,
    draw,
    fps
)
import time
from utils.lisence.detect import YoloNAS


class App:
    def __init__(self, video_source=0, window_name="lisence", flip=False) -> None:
        self.video = camera.Camera(video_source)
        self.fps = fps.FPS()
        self.window_name = window_name
        self.flip = flip
        self.max_fps = 144

        self.yolonas = YoloNAS("model/ckpt_best.pth")
        self.yolonas.load()

    def process(self, image):
        pass

    def run(self):
        self.video = self.video.start()
        try:
            while True:
                self.fps.start()
                frame = self.video.read()
                # if self.flip:
                # frame = cv2.flip(frame, 1)
                if frame is not None:
                    # CODE HERE

                    _status, frame, _croped = self.yolonas.predict(frame)
                    # CODE HERE
                    self.fps.stop()
                    self.fps.update()
                    frame = draw.putText(
                        frame, f'FPS: {self.fps.fps()}', (20, 20))
                    cv2.imshow(self.window_name, frame)
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
            self.video.stop()
            self.video.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    app = App(flip=True)
    app.run()
