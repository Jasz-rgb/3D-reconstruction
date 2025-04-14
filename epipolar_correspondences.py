import numpy as np
import cv2
from sparse import *

def epipolar_correspondences(im1, im2, F, pts1):
    pts2 = []
    for pt1 in pts1:
        pt1_hom = np.array([pt1[0], pt1[1], 1])
        epipolar_line = F @ pt1_hom

        a, b, c = epipolar_line
        x_vals = np.arange(0, im2.shape[1])
        y_vals = (-c - a * x_vals) / b

        valid_indices = (y_vals >= 0) & (y_vals < im2.shape[0])
        x_candidates = x_vals[valid_indices]
        y_candidates = y_vals[valid_indices]

        best_match = None
        min_distance = float('inf')

        window_size = 10
        x1, y1 = int(pt1[0]), int(pt1[1])
        window1 = im1[max(0, y1 - window_size):min(im1.shape[0], y1 + window_size),
                      max(0, x1 - window_size):min(im1.shape[1], x1 + window_size)]

        if window1.size == 0:
            pts2.append(None)
            continue

        for x2, y2 in zip(x_candidates, y_candidates):
            x2, y2 = int(x2), int(y2)
            window2 = im2[max(0, y2 - window_size):min(im2.shape[0], y2 + window_size),
                          max(0, x2 - window_size):min(im2.shape[1], x2 + window_size)]

            if window2.size == 0:
                continue

            window1_resized = cv2.resize(window1, (window_size * 2, window_size * 2))
            window2_resized = cv2.resize(window2, (window_size * 2, window_size * 2))

            distance = np.sum((window1_resized - window2_resized)**2)

            if distance < min_distance:
                min_distance = distance
                best_match = (x2, y2)

        if best_match:
            pts2.append(best_match)
        else:
            pts2.append(None)

    return np.array(pts2)
im1=cv2.imread(r'C:\Users\JASMINE\Desktop\TASK_6\data\im1.png',0)
im2=cv2.imread(r'C:\Users\JASMINE\Desktop\TASK_6\data\im2.png',0)
pts2=epipolar_correspondences(im1, im2, F, pts1)
cv2.imshow('im1',im1)
cv2.waitKey(0)
