import numpy as np

def rectify_pair(K1, K2, R1, R2, t1, t2):
    c1 = -np.linalg.inv(K1 @ R1) @ (K1 @ t1)
    c2 = -np.linalg.inv(K2 @ R2) @ (K2 @ t2)

    r1 = (c2 - c1) / np.linalg.norm(c2 - c1)
    r2 = np.cross(R1[2, :], r1)
    r2 = r2 / np.linalg.norm(r2)
    r3 = np.cross(r1, r2)

    R_new = np.vstack((r1, r2, r3)).T

    M1 = K2 @ R_new @ np.linalg.inv(K1 @ R1)
    M2 = K2 @ R_new @ np.linalg.inv(K2 @ R2)

    K1p = K2
    K2p = K2
    R1p = R_new
    R2p = R_new
    t1p = np.zeros(3)
    t2p = -R_new @ (c2 - c1)

    return M1, M2, K1p, K2p, R1p, R2p, t1p, t2p
