"""
CUDA and OpenCV

Simple edit: flip vertically
"""

from os.path import join

import cupy as np
import cv2

from __init__ import assetsdir_1

cap = cv2.VideoCapture(join(assetsdir_1, "glass_1__sample_1_left.mkv"))

# https://docs.opencv.org/4.5.0/d4/d15/group__videoio__flags__base.html#gaeb8dd9c89c10a5c63c139bf7c4f5704d
fourcc = cv2.VideoWriter_fourcc(*'XVID')

fps = cap.get(cv2.CAP_PROP_FPS)
frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
out = cv2.VideoWriter(join(assetsdir_1, 'output.avi'),fourcc, fps, size)

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret==True:
        frame = cv2.flip(frame,0)

        # write the flipped frame
        out.write(frame)

        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()