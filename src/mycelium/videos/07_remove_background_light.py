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
from os.path import join, dirname
from math import sqrt
from copy import copy
import csv

import numpy as np
import cv2

from __init__ import assetsdir_1

csv_path = join(dirname(__file__), "its-alive-zoom-stats3.csv")

cap = cv2.VideoCapture(join(assetsdir_1, "glass_1__sample_1_left.mkv"))
fourcc = cv2.VideoWriter_fourcc(*'fmp4')

fps = cap.get(cv2.CAP_PROP_FPS)
frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
size = (490 * 3, 1000)
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

        # crop image in the area of interest
        image = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)[650:1650, 1580:2070]
        # create a kernel
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6,8))
        # try sort out background noise by morph transform
        bg = cv2.morphologyEx(image, cv2.MORPH_OPEN, se)
        # threshold
        edited = cv2.threshold(bg, 2, 250, cv2.THRESH_OTSU)[1]

        #gradient = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, se)

        #out_gray = cv2.divide(gradient, bg, scale=255)

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
            

            with open(csv_path, 'a') as myfile:
                wr = csv.writer(myfile, quoting=csv.QUOTE_ALL, delimiter=',')
                wr.writerow(
                    [index, round(index / 24.0, 2), 
                    t_1_diff, t_1_diff_perc, mean_perc, 
                    t_1_delta_variance, max_diff, mid_range]
                )
        else:
            mean_perc = 0.0
            mid_range = 0.0
            t_1_delta_variance = sys.maxsize

            with open(csv_path, 'w') as myfile:
                wr = csv.writer(myfile, quoting=csv.QUOTE_ALL, delimiter=',')
                wr.writerow(
                    ["frame", "time from t0", "pixels diffs", "diffs percentage",
                    "mean diff", "variance from t-1", "(max variance from t-1, frame)",
                    "mid min-max range"]
                )
        
        diff = edited

        horizontal = np.hstack((image, spotted, bg))
        horizontal_concat = np.concatenate((image, spotted, bg), axis=1)

        out.write(horizontal_concat)

        cv2.imshow('frame', horizontal_concat)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # from time import sleep
        # sleep(0.07)

    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()