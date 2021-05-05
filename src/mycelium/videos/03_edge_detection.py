"""
CUDA and OpenCV

Apply OTSU to visualise edges changes
"""

"""
CUDA and OpenCV

Simple edit: flip vertically
"""

from os.path import join

import cupy as np
import cv2
from skimage.filters import threshold_otsu, threshold_local, threshold_mean

from __init__ import assetsdir_1


cap = cv2.VideoCapture(join(assetsdir_1, "movie.mp4"))
fourcc = cv2.VideoWriter_fourcc(*'fmp4')

fps = cap.get(cv2.CAP_PROP_FPS)
frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
out = cv2.VideoWriter(join(assetsdir_1, 'output.mp4'),fourcc, fps, size)

index = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret is True:
        _, edited = cv2.threshold(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        final = cv2.cvtColor(edited,cv2.COLOR_GRAY2RGB)

        cv2.putText(final, 'fps: ' + str(fps), (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,255), 5)
        cv2.putText(final, 'frame: ' + str(index), (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,255), 5)
        cv2.putText(final, 'time: ' + str(round(index / 24.0, 2)) + "s", (0, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,255), 5)

        out.write(final)
        print(type(final), final.shape)
        index += 1

        cv2.imshow('frame', final)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()