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
        # 1st image
        # extrapolate one channel from a LAB color style and trim to target area in the frame
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)[400:730,400:730,0]
        # 2nd image
        # apply mean threshold to
        edited = cv2.medianBlur(lab, 3)
        edited = cv2.adaptiveThreshold(edited, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)

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

            if index > 50:
                # mid-range of diff
                max_diff = (t_1_diff_perc, index) if max_diff[0] is None or t_1_diff_perc > max_diff[0] else max_diff
                min_diff = (t_1_diff_perc, index) if min_diff[0] is None or t_1_diff_perc < min_diff[0] else min_diff
                mid_range =  (max_diff[0] + min_diff[0]) / 2
        else:
            diff = edited
            t_1_diff = 0
            t_1_diff_perc = 0.0
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

        with open(csv_path, 'a') as myfile:
            wr = csv.writer(myfile, quoting=csv.QUOTE_ALL, delimiter=',')
            wr.writerow(
                [index, round(index / 24.0, 2), 
                t_1_diff, t_1_diff_perc, mean_perc, 
                t_1_delta_variance, max_diff, mid_range]
            )

    else:
        break

# Release everything if job is finished
cap.release()
cv2.destroyAllWindows()