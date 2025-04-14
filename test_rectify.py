import numpy as np
import cv2
from rectify_pair import rectify_pair

data_extrinsics = np.load("data/extrinsics.npz")
R1 = data_extrinsics['R1']
R2 = data_extrinsics['R2']
t1 = data_extrinsics['t1']
t2 = data_extrinsics['t2']

data_intrinsics = np.load("data/intrinsics.npz")
K1 = data_intrinsics['K1']
K2 = data_intrinsics['K2']

M1, M2, K1p, K2p, R1p, R2p, t1p, t2p = rectify_pair(K1, K2, R1, R2, t1, t2)

im1 = cv2.imread("data/im1.png")
im2 = cv2.imread("data/im2.png")

im1_rectified = cv2.warpPerspective(im1, M1, (im1.shape[1], im1.shape[0]))
im2_rectified = cv2.warpPerspective(im2, M2, (im2.shape[1], im2.shape[0]))

cv2.imshow("Image 1 Rectified", im1_rectified)
cv2.imshow("Image 2 Rectified", im2_rectified)
cv2.waitKey(0)
cv2.destroyAllWindows()
