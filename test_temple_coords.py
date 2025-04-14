import numpy as np
import matplotlib.pyplot as plt
import cv2

data = np.load("data/some_corresp.npz")
pts1 = data['pts1']
pts2 = data['pts2']

data_intrinsics = np.load("data/intrinsics.npz")
K1 = data_intrinsics['K1']
K2 = data_intrinsics['K2']

data_temple = np.load("data/temple_coords.npz")
pts1_temple = data_temple['pts1']

im1 = cv2.imread("data/im1.png")
im2 = cv2.imread("data/im2.png")

M = max(im1.shape[0], im1.shape[1])

from eight_point import eight_point
from epipolar_correspondences import epipolar_correspondences
from essential_matrix import essential_matrix
from triangulate import triangulate

F = eight_point(pts1, pts2, M)

pts2_temple = epipolar_correspondences(im1, im2, F, pts1_temple)

valid_indices = [i for i in range(len(pts2_temple)) if pts2_temple[i] is not None]
pts1_temple_valid = pts1_temple[valid_indices]
pts2_temple_valid = [pts2_temple[i] for i in valid_indices]
pts2_temple_valid = np.array(pts2_temple_valid)


E = essential_matrix(F, K1, K2)

def camera2(E):
    U, S, V = np.linalg.svd(E)
    if np.linalg.det(U) < 0:
        U[:, 2] *= -1
    if np.linalg.det(V) < 0:
        V[:, 2] *= -1

    W = np.array([[0, -1, 0],
                  [1, 0, 0],
                  [0, 0, 1]])

    P2s = [np.hstack((U @ W @ V.T, U[:, 2].reshape(3, 1))),
           np.hstack((U @ W @ V.T, -U[:, 2].reshape(3, 1))),
           np.hstack((U @ W.T @ V.T, U[:, 2].reshape(3, 1))),
           np.hstack((U @ W.T @ V.T, -U[:, 2].reshape(3, 1)))]
    return P2s

Ps = camera2(E)
P1 = np.hstack((np.eye(3), np.zeros((3, 1))))
best_P2 = None
best_points3d = None
max_positive_depth = -1

for P2 in Ps:
    points3d = triangulate(K1 @ P1, pts1_temple_valid, K2 @ P2, pts2_temple_valid)

    points3d_camera1 = (np.linalg.inv(K1 @ P1[:3,:3]) @ (points3d.T - K1 @ P1[:3,3].reshape(3,1))).T
    points3d_camera2 = (np.linalg.inv(K2 @ P2[:3,:3]) @ (points3d.T - K2 @ P2[:3,3].reshape(3,1))).T

    num_positive_depth = np.sum((points3d_camera1[:, 2] > 0) & (points3d_camera2[:, 2] > 0))

    if num_positive_depth > max_positive_depth:
        max_positive_depth = num_positive_depth
        best_P2 = P2
        best_points3d = points3d

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(best_points3d[:, 0], best_points3d[:, 1], best_points3d[:, 2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('3D Reconstruction of Temple')
plt.show()

R2 = best_P2[:, :3]
t2 = best_P2[:, 3]

R1 = np.eye(3)
t1 = np.zeros(3)

np.savez("data/extrinsics.npz", R1=R1, R2=R2, t1=t1, t2=t2)

from reprojection_error import reprojection_error
mean_error = reprojection_error(best_points3d, K1 @ P1, pts1_temple_valid)
print(f"The re-projection error is: {mean_error}")
