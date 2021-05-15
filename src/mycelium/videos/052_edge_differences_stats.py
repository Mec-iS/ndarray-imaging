"""
CUDA and OpenCV

"Never Stops Exploring" video
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Compute binary threshold differences frame by frame.

Stats generator and CSV storage

For optimizations see https://jamesbowley.co.uk/accelerating-opencv-with-cuda-streams-in-python/
For use of `cv2.cuda_GpuMat` see https://www.simonwenkel.com/2020/12/30/opencv-python-api-and-custom-cuda-kernels.html
"""
import sys
import csv
from os.path import join, dirname
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

csv_path = join(dirname(__file__), "its-alive-zoom-stats2.csv")
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
            t_1_delta_variance_mod = abs(t_1_delta_variance)

            t_0_sq = copy(t_1_sq)

            if index > 500:
                # mid-range of diff
                max_diff = (t_1_delta_variance_mod, index) if max_diff[0] is None or t_1_delta_variance_mod > max_diff[0] else max_diff
                min_diff = (t_1_delta_variance_mod, index) if min_diff[0] is None or t_1_delta_variance_mod < min_diff[0] else min_diff
                mid_range =  (max_diff[0] + min_diff[0]) / 2
            
            with open(csv_path, 'a') as myfile:
                wr = csv.writer(myfile, quoting=csv.QUOTE_ALL, delimiter=',')
                wr.writerow(
                    [index, round(index / 24.0, 2), t_1_diff_perc,
                    mean_perc, t_1_delta_variance_mod, max_diff, min_diff,
                    int(t_1_delta_variance_mod > mid_range)]
                )
        else:
            diff = edited
            t_1_diff = 0
            t_1_diff_perc = 0.0
            mean_perc = 0.0
            mid_range = sys.maxsize
            t_1_delta_variance = sys.maxsize

            with open(csv_path, 'w') as myfile:
                wr = csv.writer(myfile, quoting=csv.QUOTE_ALL, delimiter=',')
                wr.writerow(
                    ["frame", "time from t0", "diffs percentage",
                    "mean diff", "variance from t-1 abs", "(max variance from t-1, frame)", "(min variance from t-1, frame)",
                    "over/under mid range"]
                )
    else:
        break

# Release everything if job is finished
cap.release()
cv2.destroyAllWindows()