import numpy as np

def compute_disparity_map(im_left_rectified, im_right_rectified, window_size=5, max_disparity=64):
    height, width = im_left_rectified.shape
    disparity_map = np.zeros((height, width), dtype=np.float32)

    for y in range(window_size // 2, height - window_size // 2):
        for x in range(window_size // 2, width - window_size // 2):
            best_disparity = 0
            min_ssd = float('inf')

            for d in range(max_disparity):
                if x - d < window_size // 2:
                    break

                left_window = im_left_rectified[y - window_size // 2:y + window_size // 2 + 1,
                                                x - window_size // 2:x + window_size // 2 + 1]
                right_window = im_right_rectified[y - window_size // 2:y + window_size // 2 + 1,
                                                  x - d - window_size // 2:x - d + window_size // 2 + 1]

                ssd = np.sum((left_window.astype(np.float32) - right_window.astype(np.float32)) ** 2)

                if ssd < min_ssd:
                    min_ssd = ssd
                    best_disparity = d

            disparity_map[y, x] = best_disparity

    return disparity_map
