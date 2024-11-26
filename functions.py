import cv2
from random import randint
from sklearn.cluster import KMeans
from sklearn.cluster import HDBSCAN, DBSCAN
from mpl_toolkits import mplot3d
import numpy as np
import sys

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

def one_line_metric(line1, line2):
    x_start_1, x_end_1 = line1.get_section_x()
    x_start_2, x_end_2 = line2.get_section_x()

    x_start = min(x_start_1, x_start_2)
    x_end = max(x_end_1, x_end_2)

    y1, y2 = line1.y(x_start), line2.y(x_start)
    dy1 = abs(y1 - y2)

    y1, y2 = line1.y(x_end), line2.y(x_end)
    dy2 = abs(y1 - y2)

    l1 = abs(line1.l * (x_end - x_start) / (x_end_1 - x_start_1))
    l2 = abs(line2.l * (x_end - x_start) / (x_end_2 - x_start_2))

    return  (dy1 + dy2) / (l1 + l2)
def line_metric(line1, line2):
    a1, b1, c1, l1 = points_to_param(line1)
    a2, b2, c2, l2 = points_to_param(line2)
    x_start_1, x_end_1 = line1[0], line1[2]
    x_start_2, x_end_2 = line2[0], line2[2]
    x_start = min(x_start_1, x_start_2)
    x_end = max(x_end_1, x_end_2)
    y1 = -(c1 + a1 * x_start) / b1
    y2 = -(c2 + a2 * x_start) / b2
    dy1 = abs(y1 - y2)
    y1 = -(c1 + a1 * x_end) / b1
    y2 = -(c2 + a2 * x_end) / b2
    dy2 = abs(y1 - y2)
    l1 = abs(l1 * (x_end - x_start) / (x_end_1 - x_start_1))
    l2 = abs(l2 * (x_end - x_start) / (x_end_2 - x_start_2))
    return (dy1 + dy2) / (l1 + l2)

def warp_perspective_to_top_view(image, src_points, dst_points, output_size):
    # Вычисляем матрицу преобразования перспективы
    src = np.array(src_points, dtype=np.float32)
    dst = np.array(dst_points, dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(src, dst)

    # Применяем преобразование к изображению
    warped_image = cv2.warpPerspective(image, matrix, output_size)
    return warped_image
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

class Line():

    def __init__(self, lines):
        self.lines = lines
        self.a, self.b, self.c, self.segment_len = self.mean_line_param()
        self.l = self.get_len()
    def get_len(self):
        x1, x2 = self.get_section_x()
        y1, y2 = self.y(x1), self.y(x2)
        S = ((x2 -x1)**2 + (y2 - y1)**2)**0.5
        if S == np.inf:
            y1, y2 = self.get_section_y()
            x1, x2 = self.x(y1), self.x(y2)
            S = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

        return S

    def get_section_x(self):
        x_start = min([line[0][0] for line in self.lines])
        x_end = max([line[0][2] for line in self.lines])
        return x_start, x_end

    def get_section_y(self):
        y_start = min([line[0][1] for line in self.lines])
        y_end = max([line[0][3] for line in self.lines])
        return y_start, y_end

    def y(self, x):
        return -(self.c + self.a*x)/self.b
    def x(self, y):
        return -(self.c + self.b * y) / self.a

    def mean_line_param(self):
        a_lines, b_lines, c_lines, l  = [], [], [], 0

        for line in self.lines:
            a, b, c, l = points_to_param(line[0])
            a_lines.append(a)
            b_lines.append(b)
            c_lines.append(c)
            l += l

        b_lines, a_lines = np.array(b_lines), np.array(a_lines)
        c, b, a = np.mean(c_lines), np.mean(b_lines), np.mean(a_lines)

        return a, b, c, l

    def print(self,image, color, type = "all", d = 2):
        if type == "all":
            for line in self.lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(image, (x1, y1), (x2, y2), color, d)
        if type == "one":
            if abs(self.b) > 0.00001:
                x1, x2 = self.get_section_x()
                y1, y2 = self.y(x1), self.y(x2)
                cv2.line(image, (x1, int(y1)), (x2, int(y2)), color, d)
            else:
                y1, y2 = self.get_section_y()
                x1, x2 = self.x(y1), self.x(y2)
                cv2.line(image, ( int(x1),y1), (int(x2), y2), color, d)

class Cluster():
    def __init__(self, lines):
        self.clusters_params, self.l =  self.get_param(lines)
        self.lines = self.get_lines(lines)
    def segment_len(self):
        l = 0
        for line in self.lines:
            l += line.segment_len
        return l
    def get_param(self, lines):
        params = []
        l_cluster = 0
        for line in lines:
            a,b,c, l = line.a, line.b, line.c, line.l
            params.append([a,b,c])
            l_cluster += l
        return params, l_cluster

    def get_lines(self, lines):
        lines_array = []
        for line in lines:
            lines_array.append(line)
        return lines_array
    def max_norm(self):
        S = []
        l = len(self.lines)
        for i in range(1, l):
            for j in range(i):
                S.append(one_line_metric(self.lines[i], self.lines[j]))
        return max(S)

    def draw(self, image, color = None):
        if color == None:
            color = (randint(10, 255), randint(10, 255), randint(10, 255))
        for lines in self.lines:
            lines.print(image, color= color, type = 'one', d = 4)

    def get_lines_params(self):
        return self.clusters_params

    def __gt__(self, other):
        if self.l > other.l:
            return True
        return False

class Metric():
    def __init__(self, image):
        self.height, self.width, _ = image.shape
        self.perimetr = 2*(self.height + self.width)
    def count(self, x,y):
        dx = abs(x[0] - y[0]) % self.perimetr
        dy = abs(x[1] - y[1]) % self.perimetr
        return (dx ** 2 + dy ** 2) ** 0.5
    def cluster_metric(self, x,y):
        dx = abs(x[0] - y[0]) % self.perimetr
        dy = abs(x[1] - y[1]) % self.perimetr
        S = (dx ** 2 + dy ** 2) ** 0.5
        return S/x[2] + S/y[2]

class Corrcet():
    def __init__(self, image):
        self.image = image

    def point(self, point):
        height, width, _ = self.image.shape
        x, y = point
        if 0 < x < width and 0 < y < height:
            return True
        else:
            return False
    #возможно следующий метод не требуется
    def line(self, line):
        if contour_cords(self.image, line, method='check') == False:
            return False
        else:
            return True
