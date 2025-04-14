import numpy as np

def reprojection_error(pts3d, P1, pts1):
    """
    Calculates the mean re-projection error.

    """
    # Project 3D points back into the first image
    pts3d_homogeneous = np.hstack((pts3d, np.ones((pts3d.shape[0], 1))))
    pts2d_projected_homogeneous = (P1 @ pts3d_homogeneous.T).T
    pts2d_projected = pts2d_projected_homogeneous[:, :2] / pts2d_projected_homogeneous[:, 2:]

    # Calculate the Euclidean distance between the projected points and the original points
    errors = np.linalg.norm(pts2d_projected - pts1, axis=1)

    # Calculate the mean error
    mean_error = np.mean(errors)

    return mean_error
