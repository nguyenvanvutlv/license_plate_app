#!/usr/bin/env python

'''
Floodfill sample.

Usage:
  floodfill.py [<image>]

  Click on the image to set seed point

Keys:
  f     - toggle floating range
  c     - toggle 4/8 connectivity
  ESC   - exit
'''

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv

from skimage.segmentation import flood_fill


class FloodFill():
    def __init__(self, window_name) -> None:
        self.window_name = window_name
        # cv.createTrackbar('lo', self.window_name, 20, 255, self.update)
        # cv.createTrackbar('hi', self.window_name, 20, 255, self.update)

    def run(self, img, seed_x, seed_y, color_code):
        filled_checkers = flood_fill(img, (seed_x, seed_y), color_code)
        return filled_checkers
