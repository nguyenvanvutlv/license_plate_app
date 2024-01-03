import cv2


def difference_frame(before_frame, after_frame):
    if before_frame is None or after_frame is None:
        return None
    diff = cv2.absdiff(before_frame, after_frame)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return len(contours)