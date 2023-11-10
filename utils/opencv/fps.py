import time
import collections
import numpy as np

class FPS:
    def __init__(self) -> None:
        self.__processing_times = collections.deque()
        self.__processing_time = 0
        self.__fps = 0
        self.__start = None
        self.__end = None

    def start(self):
        self.__start = time.time()
    
    def stop(self):
        self.__end = time.time()

    def update(self):
        self.__processing_times.append(self.__end - self.__start)
        if len(self.__processing_times) > 200:
            self.__processing_times.popleft()
        self.__processing_time = np.mean(self.__processing_times) * 1100
        self.__fps = 1000 / self.__processing_time
        

    def fps(self):
        try:
            return int(round(self.__fps, 2))
        except:
            return -1