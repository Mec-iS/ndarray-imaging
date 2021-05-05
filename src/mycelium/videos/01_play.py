"""
CUDA and OpenCV

Simple play
"""

from os.path import join
import cupy as np
import cv2

from __init__ import assetsdir_1

cap = cv2.VideoCapture(join(assetsdir_1, "glass_1__sample_1_left.mkv"))

while(cap.isOpened()):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()