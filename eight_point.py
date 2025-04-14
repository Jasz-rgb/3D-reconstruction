import numpy as np

def eight_point(pts1, pts2, M):
    T = np.array([[1/M, 0, 0],
                  [0, 1/M, 0],
                  [0, 0, 1]])

    pts1_norm = np.hstack((pts1, np.ones((pts1.shape[0], 1))))  #Homogeneous coordinates
    pts2_norm = np.hstack((pts2, np.ones((pts2.shape[0], 1)))) #Homogeneous coordinates

    pts1_norm = (T @ pts1_norm.T).T[:,:2] # Apply transformation
    pts2_norm = (T @ pts2_norm.T).T[:,:2]

    A = np.zeros((pts1.shape[0], 9))
    for i in range(pts1.shape[0]):
        x1, y1 = pts1_norm[i,0], pts1_norm[i,1]
        x2, y2 = pts2_norm[i,0], pts2_norm[i,1]
        A[i] = [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]

    U, S, V = np.linalg.svd(A)
    F_norm = V[-1].reshape(3, 3)

    U, S, V = np.linalg.svd(F_norm)
    S[2] = 0
    F_norm = U @ np.diag(S) @ V

    F = T.T @ F_norm @ T

    return F
