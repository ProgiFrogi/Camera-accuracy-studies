import  numpy as np
import cv2
from functions import  correct_point


def draw_cluster(image, clusters):
    for cluster in clusters:
        cluster.draw(image)

def draw_center(image, points):
    correct = []
    for point in points:
        if correct_point(image, point):
            correct.append(point)
    correct = np.array(correct)

    if len(correct) > 0:
        center = np.mean(correct, 0)
        cv2.circle(image, (int(center[0]), int(center[1])), 25, (0, 0, 255), 2)

def draw_point(image, point):

    if correct_point(image, point):
        cv2.circle(image, (int(point[0]),int(point[1])), radius=5, color=(0, 0, 0), thickness=-1)

    return  image