import cv2
import numpy as np


def getColor(label):
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return color


def putText(origin, text, pos):
    img = origin.copy()
    color = getColor(0)
    x, y = pos
    t_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
    cv2.rectangle(
        img, (x, y), (x + t_size[0] + 3, y + t_size[1] + 4), color, -1)
    cv2.putText(img, text, (x, y + t_size[1] + 4),
                cv2.FONT_HERSHEY_PLAIN, 1.6, [255, 255, 255], 2)
    return img


def draw_boxes(origin, bbox):
    img = origin.copy()
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        color = getColor(0)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    return img


def expand(frame):
    h, w, c = frame.shape
    cx, cy = w//2, h//2
    newImage = np.zeros(((w//2) * 8, (h//2) * 8, c), dtype=frame.dtype)
    x, y = -2 * cx, -2 * cy
    newImage[-x: (-x)+h, -y: (-y)+w] = frame.copy()
    return newImage
