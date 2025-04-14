import numpy as np
def compute_depth_map(disparity_map, K, baseline):
    f = K[0, 0]  # Focal length (assuming fx and fy are equal)
    depth_map = np.zeros_like(disparity_map)

    depth_map[disparity_map > 0] = (f * baseline) / disparity_map[disparity_map > 0]

    return depth_map
