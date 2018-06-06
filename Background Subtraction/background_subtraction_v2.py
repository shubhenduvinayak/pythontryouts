import cv2
import numpy as np

cap = cv2.VideoCapture(0)

subtractor = cv2.createBackgroundSubtractorMOG2(history=20, varThreshold=25)

while True:
    _, frame = cap.read()

    mask = subtractor.apply(frame)

    cv2.imshow("frame",frame)
    cv2.imshow("mask",mask)

    key = cv2.waitKey(30) & 0xFF
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()