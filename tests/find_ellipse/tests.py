from utils.find_ellipse import FindEllipseRHT
from cv2 import ellipse, blur
import numpy as np
import time

def _e2e_speed_test(n):
    sum = 0
    for i in range(n):
        image = np.zeros((1000, 1000), dtype=np.uint8)
        center = (np.random.rand((2)) * np.array(image.shape)).astype(int)
        axies = (np.random.rand((2)) * np.array(image.shape) / 2).astype(int)
        # if axies[0]>axies[1]:
        #     axies[0],axies[1] = axies[1],axies[0]
        angle = np.random.random() * 360
        image = ellipse(image, center.tolist(), axies.tolist(), angle, 0., 360., 255, 3)
        image = blur(image, [3, 3])
        mask_binary = np.zeros(image.shape, dtype=bool)
        mask_binary[image == 255] = False
        mask_binary[image != 255] = True
        time1 = time.time()
        test = FindEllipseRHT(image, mask_binary,use_canny=False)
        test.max_iter = 200000
        test.score_threshold = 10
        data = test.run(plot_mode=False, debug_mode=False)
        data = sorted(data, key=lambda item: -item[-1])
        # data_to_diff = np.concat(
        #     [list(map(float, data[0][0][:2])), list(map(float, data[0][0][2:4])), [float(data[0][0][4])]])
        # data_to_diff = np.reshape(data_to_diff, (len(data_to_diff)))
        # other_to_diff = np.concat([center.astype(float), axies.astype(float), np.array([angle])])
        # print(data[0], [center, axies, angle], np.linalg.norm(data_to_diff - other_to_diff))
        # image = ellipse(image, list(map(int, data[0][0][:2])), list(map(int, data[0][0][2:4])),
        #                 float(data[0][0][4]) / np.pi * 180, 0., 360., 140, 3)
        time2 = time.time()
        sum+=(time2 - time1)
    return sum/n


if __name__ == "__main__":
    print(_e2e_speed_test(4))