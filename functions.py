import cv2
from random import randint
from sklearn.cluster import KMeans
from sklearn.cluster import HDBSCAN, DBSCAN
from mpl_toolkits import mplot3d
import numpy as np

import sys

def draw_cluster(image, clusters, filt = None):
    for cluster in clusters:
        if filt == None:
            #print("OK")
            cluster.draw(image)
            continue
        if filt(cluster):
            #print("OK")
            cluster.draw(image)
def point_on_height_line(cluster):
    func = lambda line: -line.c/line.a
    points = [func(line) for line in cluster.lines]
    return points

def left_and_right_point(image, clusters):
    height, width, _ = image.shape
    right_m, left_m = 2000, 2000
    for cluster in clusters:
        func = lambda line: -line.c / line.b
        left = min([func(line) for line in cluster.lines])
        left_m = min(left, left_m)

        func = lambda line: -(line.c + line.a*height)/ line.b
        right = min([func(line) for line in cluster.lines])
        right_m = min(right, right_m)


    return right_m, left_m

def nearest_point(target_point , points):
    metric = lambda p1, p2: (p1[0]- p2[0])**2 + (p1[1] - p2[1])**2
    R_min = 10000000000000
    nearest = None
    for point in points:
        R = metric(target_point, point)
        if R < R_min:
            nearest = point
            R_min = R
    return nearest

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
    return warped_image, matrix

def inverse(point, matrix):
    point_img2_homogeneous = np.array([point[0], point[1], 1])
    point_img1_homogeneous = np.dot(matrix, point_img2_homogeneous)
    x = point_img1_homogeneous[0] / point_img1_homogeneous[2]
    y = point_img1_homogeneous[1] / point_img1_homogeneous[2]
    return np.array([x, y])

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
        maxx = 0
        l = len(self.clusters_params)
        for i in range(1,l):
            for j in range(i):
                c1 = self.clusters_params[i][2]
                c2 = self.clusters_params[j][2]
                maxx = max(abs(c1-c2), maxx)


        maxx_other = 0
        l = len(other.clusters_params)
        for i in range(1, l):
            for j in range(i):
                c1 = other.clusters_params[i][2]
                c2 = other.clusters_params[j][2]
                maxx_other = max(abs(c1 - c2), maxx_other)
        file = open('statistic.txt', 'a')
        file.write(str(self.l) + " " + str(maxx) + "\n")
        file.close()
        if self.l*maxx > other.l*maxx_other:
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


class line_detect():
    def __init__(self, image):
        self.image = image
        self.rho = 3
        self.theta = np.pi / (140)
        self.threshold = 60
        self.minLineLength = 60
        self.maxLineGap = 15
        self.eps_dbscan = 0.0125
        self.eps_dbscan_for_clusters = 0.0001 / 2.2
        self.Canny_1 = 80
        self.Canny_2 = 150
        self.blur = 5

    def find_lines(self, rgb_image):
        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        blurred_image = cv2.GaussianBlur(gray_image, (self.blur, self.blur), 0)
        edges = cv2.Canny(blurred_image, self.Canny_1, self.Canny_2)
        lines = cv2.HoughLinesP(edges,
                                rho=self.rho,
                                theta=self.theta,
                                threshold=self.threshold,
                                minLineLength=self.minLineLength,
                                maxLineGap=self.maxLineGap)

        return lines

    def one_line(self, lines):
        points = []
        for line in lines:
            x1, x2, x3, x4 = line[0]
            points.append([x1, x2, x3, x4])
        points = np.array(points)
        dbscan = DBSCAN(eps=self.eps_dbscan, min_samples=1, metric=line_metric)
        labels = dbscan.fit_predict(points)

        concatenate_lines = {}
        for i in range(len(lines)):
            ind = labels[i]
            if ind in concatenate_lines:
                concatenate_lines[ind].append(lines[i])
            else:
                concatenate_lines[ind] = [lines[i]]
        lines = [Line(value) for value in concatenate_lines.values()]

        return lines

    def merge_in_clusters(self, lines, criteria=lambda x: True):
        Line_clusters = []
        points = []
        l_arr = []
        for line in lines:
            if contour_cords(self.image, line, method='check') and criteria(line):
                Line_clusters.append(line)
                points.append(contour_cords(self.image, line))
                l_arr.append(line.l)

        S = Metric(self.image)

        l_arr = np.array(l_arr).reshape(-1, 1)
        points = np.array(points) / S.perimetr
        if len(points) ==0 or len(l_arr) == 0:
            return False
        X = np.hstack((points, l_arr))
        dbscan = DBSCAN(eps=self.eps_dbscan_for_clusters, min_samples=1, metric=S.cluster_metric)
        labels = dbscan.fit_predict(X)
        # plt.scatter(points[:, 0], points[:, 1], c=labels)
        # plt.show()

        concatenate_clusters = {}

        for i in range(len(Line_clusters)):
            ind = labels[i]
            if ind in concatenate_clusters:
                concatenate_clusters[ind].append(Line_clusters[i])
            else:
                concatenate_clusters[ind] = [Line_clusters[i]]

        Clusters = [Cluster(lines_dict) for lines_dict in concatenate_clusters.values()]

        return Clusters

    def detect(self):
        lines = self.find_lines(self.image)
        lines = self.one_line(lines)
        Clusters = self.merge_in_clusters(lines)
        return Clusters


# возможно случайно изменил класс, провеить
class Main_Line_detect(line_detect):
    def __init__(self, image):
        super().__init__(image)
        self.rho = 3
        self.theta = np.pi / (360)
        self.threshold = 150
        self.minLineLength = 100
        self.maxLineGap = 30
        self.eps_dbscan = 0.0125
        self.eps_dbscan_for_clusters = 0.0001 / 4

    def criteria(self, cluster):
        if len(cluster.lines) != 2:
            return False
        return True

    def func(self, line):
        phi = np.arctan(line.a / line.b)
        if abs(phi) < np.pi / 20 or abs(phi - np.pi) < np.pi / 20:
            return True
        return False

    def detect(self):
        lines = self.find_lines(self.image)
        lines = self.one_line(lines)
        Clusters = self.merge_in_clusters(lines, self.func)
        Clusters = list(filter(self.criteria, Clusters))
        Clusters.sort(reverse=True)
        if len(Clusters) < 1:
            return None

        return Clusters[0]


class Long_lines(line_detect):
    def __init__(self, image):
        super().__init__(image)
        self.rho = 3
        self.theta = np.pi / (360)
        self.threshold = 150
        self.minLineLength = 100
        self.maxLineGap = 40
        self.eps_dbscan = 0.025  # 0.02 - 0.03
        self.eps_dbscan_for_clusters = 0.00004
        #0.000036  # 0.000032 - 0.000041

    def criteria(self, cluster):
        if len(cluster.lines) != 2:
            return False
        if cluster.max_norm() > 0.08: #0.065
            return False

        return True

    def detect(self):
        lines = self.find_lines(self.image)
        lines = self.one_line(lines)

        def func(line):
            phi = np.arctan(line.b / line.a)
            if abs(phi) < np.pi / 5 or abs(phi - np.pi) < np.pi / 5:
                return True
            return False

        Clusters = self.merge_in_clusters(lines, func)
        #draw_cluster(self.image, Clusters)
        Clusters = list(filter(self.criteria, Clusters))
        Clusters.sort(reverse=True)
        if len(Clusters) < 2:
            return None, None

        return Clusters[0], Clusters[1]


class Last_line(Main_Line_detect):
    def __init__(self, image):
        super().__init__(image)
        self.rho = 1
        self.theta = np.pi / (360)
        self.threshold = 50
        self.minLineLength = 150
        self.maxLineGap = 30
        self.eps_dbscan = 0.0085
        self.eps_dbscan_for_clusters = 0.0001 / 4

    def criteria(self, cluster):
        if len(cluster.lines) != 2:
            return False

        return True

    def detect(self):
        lines = self.find_lines(self.image)
        lines = self.one_line(lines)

        for line in lines:
            color = (255, 255, 255)
            #line.print(self.image, color)
        Clusters = self.merge_in_clusters(lines, self.func)
        if type(Clusters) == bool:
            return False
        Clusters = list(filter(self.criteria, Clusters))
        if len(Clusters) <1:
            return False

        Clusters.sort(reverse=True)
        return Clusters
