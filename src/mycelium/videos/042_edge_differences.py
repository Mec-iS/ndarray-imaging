"""
CUDA and OpenCV

"Never Stops Exploring" video
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Compute binary threshold differences frame by frame.

Video and stats generator

For optimizations see https://jamesbowley.co.uk/accelerating-opencv-with-cuda-streams-in-python/
For use of `cv2.cuda_GpuMat` see https://www.simonwenkel.com/2020/12/30/opencv-python-api-and-custom-cuda-kernels.html
"""
import sys
from os.path import join
from math import sqrt
from copy import copy

import numpy as np
import cv2
from skimage.filters import threshold_otsu, threshold_local, threshold_mean

from __init__ import assetsdir_1


cap = cv2.VideoCapture(join(assetsdir_1, "raw_trimmed_MOTION.mp4"))
fourcc = cv2.VideoWriter_fourcc(*'fmp4')

fps = cap.get(cv2.CAP_PROP_FPS)
frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
size = (660, 330)
out = cv2.VideoWriter(join(assetsdir_1, 'output.mp4'),fourcc, fps, size, 0)

index = 0
diff = None

#
# Stats
# https://courses.lumenlearning.com/boundless-statistics/chapter/describing-variability/
#
total_pixels = size[0] * size[1]
print(total_pixels)
mean_acc = 0
t_0_sq = 0.0

max_diff = (None, None)  # (value, frame number)
min_diff = (None, None)  # (value, frame number)

while(cap.isOpened()):
    ret, frame = cap.read()
    frame = np.asarray(frame)
    if ret is True:
        index += 1

        # https://stackoverflow.com/a/62049804/2536357
        image = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)[400:730,400:730]
        se = cv2.getStructuringElement(cv2.MORPH_RECT , (8,8))
        bg = cv2.morphologyEx(image, cv2.MORPH_DILATE, se)
        out_gray = cv2.divide(image, bg, scale=255)
        edited = cv2.threshold(out_gray, 0, 255, cv2.THRESH_OTSU)[1]

        spotted = np.full(edited.shape, 255, np.uint8)
        if diff is not None:
            spotted[edited != diff] = 0
            t_1_diff = np.count_nonzero(spotted == 0)
            t_1_diff_perc = t_1_diff / total_pixels
            mean_acc += t_1_diff_perc
            
            # mean of percent of diff
            mean_perc = mean_acc / index

            # variance
            t_1_sq = pow((t_1_diff_perc - mean_perc), 2)
            t_1_delta_variance = (t_1_sq / index) - (t_0_sq / (index - 1))

            t_0_sq = copy(t_1_sq)

            if index > 500:
                # mid-range of diff and max variance
                max_diff = (t_1_diff_perc, index) if max_diff[0] is None or t_1_diff_perc > max_diff[0] else max_diff
                min_diff = (t_1_diff_perc, index) if min_diff[0] is None or t_1_diff_perc < min_diff[0] else min_diff
                mid_range =  (max_diff[0] + min_diff[0]) / 2
        else:
            diff = edited
            mean_perc = 0.0
            mid_range = 0.0
            t_1_delta_variance = sys.maxsize

        separator = np.full(edited.shape, 0, np.uint8)
        horizontal = np.hstack((image, separator, edited))
        horizontal_concat = np.concatenate((image, separator, edited), axis=1)

        cv2.putText(horizontal_concat,
            f'frame: {str(index)}', (350, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,255), 5)
        cv2.putText(horizontal_concat,
            f'time: {str(round(index / 24.0, 2))}s', (350, 110), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,255), 5)

        if max_diff[0] is not None:
            cv2.putText(horizontal_concat,
                f'max-var-diff: {str(round(max_diff[0], 6))}, {str(max_diff[1])}, {str(index - max_diff[1])}',
                (325, edited.shape[0]-175), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 5)
            # cv2.putText(horizontal_concat,
            #     f'min-var-diff: {str(round(min_diff [0], 6))}, {str(min_diff [1])}, {str(index - min_diff [1])}',
            #     (25, edited.shape[0]-145), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 5)

        cv2.putText(horizontal_concat,
            'mean: ' + str(round(mean_perc, 4)), (350, edited.shape[0]-100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 5)
        cv2.putText(horizontal_concat,
            'mid-range: ' + str(round(mid_range, 4)), (350, edited.shape[0]- 66), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 5)
        cv2.putText(horizontal_concat,
            'variance: ' + str(round(t_1_delta_variance, 10)), (350, edited.shape[0]- 33), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 5)

        out.write(horizontal_concat)

        cv2.imshow('frame', horizontal_concat)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        from time import sleep
        sleep(0.07)

    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()