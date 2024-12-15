import cv2
import numpy as np
import matplotlib.pyplot as plt


def generate_marker(marker_id, marker_size, directory, filename):
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

    marker_image = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)

    cv2.imwrite(f'{directory}/{filename}', marker_image)
    plt.axis('off')
    plt.imshow(marker_image, interpolation='nearest', cmap='gray', vmin=-100, vmax=50)

    plt.title(f'ArUco Marker {marker_id}')
    plt.show()

if __name__ == '__main__':
    for i in range(1, 15):
        generate_marker(i, 800, './aruco', f'{i}_size800x800.png')
