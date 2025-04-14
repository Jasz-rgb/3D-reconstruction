import numpy as np
import  cv2 
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
def essential_matrix(F, K1, K2):
    E = K2.T @ F @ K1
    return E
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
im1=cv2.imread(r'C:\Users\JASMINE\Desktop\TASK_6\data\im1.png',0)
im2=cv2.imread(r'C:\Users\JASMINE\Desktop\TASK_6\data\im2.png',0)
given=np.load(r'C:\Users\JASMINE\Desktop\TASK_6\data\some_corresp.npz')
pts1=given['pts1']
pts2=given['pts2']
h,w=im1.shape[:2]
M=max(h,w)
F=eight_point(pts1, pts2, M)
print("Fundamental matrix is ",F)
#correspondance
coords=np.load(r'C:\Users\JASMINE\Desktop\TASK_6\data\temple_coords.npz')
ptim1=coords['pts1']
ptim2=epipolar_correspondences(im1, im2, F, pts1)
for i, point in enumerate(ptim2):
    x, y = int(point[0]), int(point[1])
    cv2.circle(im2, (x, y), radius=5, color=(0, 0, 255), thickness=-1)  # Red circle
cv2.imshow('im2',im2)
cv2.waitKey(0)
#intrinsic
intrinsic=np.load(r'C:\Users\JASMINE\Desktop\TASK_6\data\intrinsics.npz')
print(intrinsic.files)
K1=intrinsic['K1']
K2=intrinsic['K2']
E=essential_matrix(F, K1, K2)
print("\nEssential matrix is ",E)


