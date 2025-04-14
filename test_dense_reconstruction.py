import numpy as np
import cv2
from rectify_pair import rectify_pair
from compute_disparity import compute_disparity_map
from compute_depth import compute_depth_map
from visualize_dense import visualize_dense

data_extrinsics = np.load("data/extrinsics.npz")
R1 = data_extrinsics['R1']
R2 = data_extrinsics['R2']
t1 = data_extrinsics['t1']
t2 = data_extrinsics['t2']

data_intrinsics = np.load("data/intrinsics.npz")
K1 = data_intrinsics['K1']
K2 = data_intrinsics['K2']

M1, M2, K1p, K2p, R1p, R2p, t1p, t2p = rectify_pair(K1, K2, R1, R2, t1, t2)

im_left = cv2.imread("data/im1.png", cv2.IMREAD_GRAYSCALE)
im_right = cv2.imread("data/im2.png", cv2.IMREAD_GRAYSCALE)

im_left_rectified = cv2.warpPerspective(im_left, M1, (im_left.shape[1], im_left.shape[0]))
im_right_rectified = cv2.warpPerspective(im_right, M2, (im_right.shape[1], im_right.shape[0]))

disparity_map = compute_disparity_map(im_left_rectified, im_right_rectified)
baseline = np.linalg.norm(t1 - t2)
depth_map = compute_depth_map(disparity_map, K1p, baseline)

visualize_dense(disparity_map, depth_map)
