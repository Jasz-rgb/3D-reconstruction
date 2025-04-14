import cv2
import matplotlib.pyplot as plt

def visualize_dense(disparity_map, depth_map):
    plt.figure()
    plt.title("Disparity Map")
    plt.imshow(disparity_map, cmap='gray')
    plt.colorbar()
    plt.show()

    plt.figure()
    plt.title("Depth Map")
    plt.imshow(depth_map, cmap='plasma')
    plt.colorbar()
    plt.show()
