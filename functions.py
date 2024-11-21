import cv2
import numpy as np
from random import randint
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import HDBSCAN, DBSCAN
from mpl_toolkits import mplot3d
import numpy as np
import sys
import matplotlib.pyplot as plt

from hyper_params import *

def convert_image(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    height, width, _ = image_rgb.shape
    pixels = image_rgb.reshape(-1, 3)

    k = 3
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(pixels)

    clustered_pixels = kmeans.cluster_centers_[kmeans.labels_].astype(np.uint8)
    clustered_frame = clustered_pixels.reshape(height, width, 3)
    return clustered_frame

def points_to_param(points):
    x1, y1, x2, y2 = points
    a, b = None, None
    if (abs(x1 - x2) > 0.01):
        b = 1
        a = -(y1-y2)/(x1-x2)
    else:
        a = 1
        b = -(x1-x2)/(y1-y2)
    c = -a*x1-b*y1

    l = ((x2-x1)**2 + (y2-y1)**2)**0.5
    return a, b, c, l

def contour_cords(image, line, method = "calc"):
    height, width, _ = image.shape
    a,b,c = line.a, line.b, line.c
    x_up, x_down = None, None
    y_left, y_right = None, None
    point = []
    if abs(a) > 0.00001:
        x_up, x_down = -c/a, -(c+b*height)/a
    else:
        x_up, x_down  = -1, -1
    if abs(b) > 0.00001:
        y_left, y_right = -c/b, -(c+a*width)/b
    else:
        y_left, y_right = -1, -1

    if 0 < x_up < width:
        point.append(x_up)
    if 0 < y_right < height:
        point.append(width + y_right)
    if 0 < x_down < width:
        point.append(width + height + x_down)
    if 0 < y_left < height:
        point.append(2*width + height + y_left)
    if len(point) != 2:
        print(point)
        print(a,b,c)
        print("Line point Error!")
        return False
    if method  == 'check':
        return True
    return np.array(point)

'''
def concatenate_line(image, lines):
    a_arr, b_arr, c_arr = [], [], []
    points = []
    for line in lines:
        a, b, c, l = points_to_param(line[0])
        a_arr.append(a)
        b_arr.append(b)
        c_arr.append(c)
        line = Line_cluster([line])
        point = 
        points.append(contour_cords(image, line))
    a_arr = np.array(a_arr) / np.max(np.abs(a_arr))
    b_arr = np.array(b_arr) / np.max(np.abs(b_arr))
    c_arr = np.array(c_arr) / np.max(np.abs(c_arr))

    S = Metric(image)
    points = 2 * np.array(points) / S.perimetr
    dbscan = DBSCAN(eps=eps_dbscan_for_clusters/8, min_samples=1, metric=S.count)
    labels = dbscan.fit_predict(points)
    #plt.scatter(points[:,0], points[:,1], c= labels)
    #plt.show()

    concatenate_lines = {}

    for i in range(len(lines)):
        ind = labels[i]
        if ind in concatenate_lines :
            concatenate_lines[ind].append(lines[i])
        else:
            concatenate_lines[ind] = [lines[i]]

    return concatenate_lines
'''

def concatenate_line(image, lines):
    points = []
    correct_lines = []
    for line in lines:
        line_p = Line_cluster([line])
        if contour_cords(image, line_p, method= 'check'):
            point = contour_cords(image, line_p)
            points.append(point)
            correct_lines.append(line)

    #print(len(lines), np.shape(points))
    S = Metric(image)
    points = np.array(points) / S.perimetr
    dbscan = DBSCAN(eps=eps_dbscan_for_small_lines, min_samples=1, metric=S.count)
    labels = dbscan.fit_predict(points)
    #plt.scatter(points[:,0], points[:,1], c= labels)
    #plt.show()

    concatenate_lines = {}

    for i in range(len(correct_lines)):
        ind = labels[i]
        if ind in concatenate_lines :
            concatenate_lines[ind].append(correct_lines[i])
        else:
            concatenate_lines[ind] = [correct_lines[i]]

    return concatenate_lines

def merge_in_clusters(image, lines):
    global eps_dbscan_for_clusters
    global min_cluster_len

    Line_clusters = []
    points = []

    for label in lines:
        Lines = Line_cluster(lines[label])

        if Lines.l > min_cluster_len  and contour_cords(image, Lines, method= 'check'):
            Line_clusters.append(Lines)
            points.append(contour_cords(image, Lines))

    S = Metric(image)
    points = np.array(points) / S.perimetr
    dbscan = DBSCAN(eps=eps_dbscan_for_clusters, min_samples=1, metric= S.count)
    labels = dbscan.fit_predict(points)
    #plt.scatter(points[:,0], points[:,1], c= labels)
    #plt.show()

    concatenate_clusters = {}

    for i in range(len(Line_clusters)):
        ind = labels[i]
        if ind in concatenate_clusters:
            concatenate_clusters[ind].append(Line_clusters[i])
        else:
            concatenate_clusters[ind] = [Line_clusters[i]]

    Clusters = [ Cluster(lines_dict) for lines_dict in concatenate_clusters.values()]

    return Clusters

def correct(image, line):
    if contour_cords(image, line, method= 'check') == False:
        return False
    else:
        return True

def intersection(line1, line2):
    a1, b1, c1 = line1
    a2, b2, c2 = line2

    a = np.array([[a1, b1], [a2, b2]])
    b = np.array([c1, c2])
    if np.linalg.cond(a) < 1 / sys.float_info.epsilon:
        return np.linalg.solve(a,b)*np.array([-1, -1])
    else:
        return [-1,-1]

def intersections_clusters(cluster1, cluster2):
    cluster1_a_b_c = cluster1.get_lines_params()
    cluster2_a_b_c = cluster2.get_lines_params()
    points = []
    for line1 in cluster1_a_b_c:
        for line2 in cluster2_a_b_c:
            point = intersection(line1, line2)
            points.append(point)
    return points

def correct_point(image, point):
    height, width, _ = image.shape
    x, y = point
    if 0 < x < width and 0 < y < height:
        return True
    else:
        return  False

def find_lines(rgb_image):
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    edges = cv2.Canny(blurred_image, 50, 150)
    lines = cv2.HoughLinesP(edges,
                            rho=rho,
                            theta=theta,
                            threshold=threshold,
                            minLineLength=minLineLength,
                            maxLineGap=maxLineGap)

    return lines

def sort_lines_by_length(lines):
    if lines is None:
        return []

    # Вычисляем длины линий и сохраняем их вместе с координатами
    line_lengths = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)  # Длина линии
        line_lengths.append((length, line))

    # Сортируем по длине в порядке убывания
    sorted_lines = sorted(line_lengths, key=lambda x: x[0], reverse=True)

    # Извлекаем отсортированные линии
    sorted_lines = [line for length, line in sorted_lines]

    return sorted_lines

'''def is_double_line(cluster):
    S = []
    lines = cluster.get_lines()
    l = len(lines)
    for i in range(1, l):
        for j in range(i):
            S.append(my_metric(lines[i], lines[j]))
    if 20 > max(S) > 9:
        return False
    return True'''

class Line_cluster():

    def __init__(self, lines):
        self.lines = lines
        self.a, self.b, self.c, self.l = self.mean_line_param()

    def get_section(self):
        x_start = min([line[0][0] for line in self.lines])
        x_end = max([line[0][2] for line in self.lines])
        return x_start, x_end

    def mean_line_param(self):
        a_lines, b_lines, c_lines, l_lines = [], [], [], 0

        for line in self.lines:
            a, b, c, l = points_to_param(line[0])
            a_lines.append(a)
            b_lines.append(b)
            c_lines.append(c)
            l_lines += l

        b_lines, a_lines = np.array(b_lines), np.array(a_lines)
        b, a = np.mean(b_lines), np.mean(a_lines)

        return a, b, c, l_lines

    def print(self,image, color):
        for line in self.lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(image, (x1, y1), (x2, y2), color, 2)

class Cluster():
    def __init__(self, cluster_lines):
        self.clusters_params, self.l =  self.get_param(cluster_lines)
        self.lines = self.get_lines(cluster_lines)

    def get_param(self, clusters_lines):
        params = []
        l_cluster = 0
        for cluster in clusters_lines:
            a,b,c, l = cluster.a, cluster.b, cluster.c, cluster.l
            params.append([a,b,c])
            l_cluster += l
        return params, l_cluster

    def get_lines(self, cluster_lines):
        lines_array = []
        for line in cluster_lines:
            lines_array.append(line)
        return lines_array

    def draw(self, image):
        color = (randint(10, 255), randint(10, 255), randint(10, 255))
        for lines in self.lines:
            lines.print(image, color= color)

    def max_norm(self, image):
        S = Metric(image).line_metric
        n  = len(self.lines)
        l = 0
        for i in range(1,n):
            for j in range(i):
                l = max(l, S(self.lines[i], self.lines[j]))
        return l

    def get_lines_params(self):
        return self.clusters_params

class Metric():
    def __init__(self, image):
        height, width, _ = image.shape
        self.perimetr = 2*(height + width)
    def count(self, x,y):
        dx = abs(x[0] - y[0]) % self.perimetr
        dy = abs(x[1] - y[1]) % self.perimetr
        return (dx ** 2 + dy ** 2) ** 0.5
    def line_metric(self, line1, line2):
        x_start_1, x_end_1 = line1.get_section()
        x_start_2, x_end_2 = line2.get_section()
        x_start = min(x_end_1, x_end_2)
        x_end = max(x_start_1, x_start_2)
        a1, b1, c1 = line1.a, line1.b, line1.c
        a2, b2, c2 = line2.a, line2.b, line2.c
        return (abs(-(c1 + a1*x_start)/b1 + (c2 + a2*x_start)/b2) + abs(-(c1 + a1*x_end)/b1 + (c2 + a2*x_end)/b2))/2


