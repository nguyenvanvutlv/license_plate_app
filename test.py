import cv2
import numpy as np

cap = cv2.VideoCapture(0)
_, first_frame = cap.read()

# Convert the first frame to grayscale
first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# Detect corners in the first frame
corners = cv2.goodFeaturesToTrack(first_gray, maxCorners=100, qualityLevel=0.01, minDistance=10)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the current frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow using Lucas-Kanade method
    new_corners, status, _ = cv2.calcOpticalFlowPyrLK(first_gray, gray, corners, None)

    # Filter out points with poor tracking quality
    good_new = new_corners[status == 1]
    good_old = corners[status == 1]

    # Display the result
    frame_copy = frame.copy()
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        cv2.line(frame_copy, (a, b), (c, d), (0, 255, 0), 2)
        cv2.circle(frame_copy, (a, b), 5, (0, 255, 0), -1)

    cv2.imshow('Feature Tracking', frame_copy)

    if cv2.waitKey(30) & 0xFF == 27:  # Press 'Esc' to exit
        break

    # Update the points for the next iteration
    corners = good_new.reshape(-1, 1, 2)

cap.release()
cv2.destroyAllWindows()
