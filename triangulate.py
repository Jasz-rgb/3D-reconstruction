import numpy as np

def triangulate(P1, pts1, P2, pts2):
    pts3d = []
    for i in range(pts1.shape[0]):
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]

        A = np.array([
            x1 * P1[2, :] - P1[0, :],
            y1 * P1[2, :] - P1[1, :],
            x2 * P2[2, :] - P2[0, :],
            y2 * P2[2, :] - P2[1, :]
        ])

        U, S, V = np.linalg.svd(A)
        point_homogeneous = V[-1]
        point_3d = point_homogeneous[:3] / point_homogeneous[3]

        pts3d.append(point_3d)

    return np.array(pts3d)
