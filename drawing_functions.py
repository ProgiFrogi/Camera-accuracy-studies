import  numpy as np
import cv2
from functions import  Corrcet


def draw_cluster(image, clusters, filt = None):
    for cluster in clusters:
        if filt == None:
            #print("OK")
            cluster.draw(image)
            continue
        if filt(cluster):
            #print("OK")
            cluster.draw(image)

def draw_center(image, points):
    correct = []
    for point in points:
        if Corrcet(image).point(point):
            correct.append(point)
    correct = np.array(correct)

    if len(correct) > 0:
        center = np.mean(correct, 0)
        cv2.circle(image, (int(center[0]), int(center[1])), 25, (0, 0, 255), 2)

def draw_point(image, points):
    for point  in points:
        if Corrcet(image).point(point):
            cv2.circle(image, (int(point[0]),int(point[1])), radius=5, color=(0, 0, 0), thickness=-1)
