from threading import Thread
import cv2


class Camera:
    def __init__(self, video_source=0) -> None:
        self.video_source = video_source
        self.video = cv2.VideoCapture(self.video_source)
        self.res, self.frame = self.video.read()
        self.stopped = False
        

    def start(self):
        Thread(target=self.update, args=()).start()
        return self
    
    def update(self):
        while True:
            if self.stopped:
                return
            self.res, self.frame = self.video.read()
    
    def read(self):
        return self.frame
    
    def stop(self):
        self.stopped = True

    def restart(self):
        self.video = cv2.VideoCapture(self.video_source)

    def release(self):
        self.video.release()