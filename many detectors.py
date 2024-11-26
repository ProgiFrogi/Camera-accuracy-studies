from functions import *
from drawing_functions import *

#иногда все же выбираются фигово вертикальные линии, это потом пофикистт можно
#главное проверить основную гипотезу
class line_detect():
    def __init__(self, image):
        self.image = image
        self.rho = 3
        self.theta = np.pi / (140)
        self.threshold = 60
        self.minLineLength = 60
        self.maxLineGap = 15
        self.eps_dbscan = 0.0125
        self.eps_dbscan_for_clusters = 0.0001/2.2

    def find_lines(self, rgb_image):
        gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
        edges = cv2.Canny(blurred_image, 80, 150)
        lines = cv2.HoughLinesP(edges,
                                rho= self.rho,
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
        dbscan = DBSCAN(eps= self.eps_dbscan, min_samples=1, metric= line_metric)
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

    def merge_in_clusters(self, lines, criteria = lambda x: True):
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

        X = np.hstack((points, l_arr))
        dbscan = DBSCAN(eps=self.eps_dbscan_for_clusters, min_samples=1, metric=S.cluster_metric)
        labels = dbscan.fit_predict(X)
        #plt.scatter(points[:, 0], points[:, 1], c=labels)
        #plt.show()

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

#возможно случайно изменил класс, провеить
class Main_Line_detect(line_detect):
    def __init__(self, image):
        super().__init__(image)
        self.rho = 3
        self.theta = np.pi / (360)
        self.threshold = 150
        self.minLineLength = 100
        self.maxLineGap = 30
        self.eps_dbscan = 0.0125
        self.eps_dbscan_for_clusters = 0.0001 /4

    def criteria(self, cluster):
        if len(cluster.lines) != 2:
            return False
        return True

    def func(self,line):
        phi = np.arctan(line.a / line.b)
        if abs(phi) < np.pi / 20 or abs(phi - np.pi) < np.pi / 20:
            return True
        return False

    def detect(self):
        lines = self.find_lines(self.image)
        lines = self.one_line(lines)

        Clusters = self.merge_in_clusters(lines, self.func)
        Clusters  = list(filter(self.criteria, Clusters))
        Clusters.sort(reverse= True)
        if len(Clusters) <1:
            return  None

        return Clusters[0]

class Long_lines(line_detect):
    def __init__(self, image):
        super().__init__(image)
        self.rho = 3
        self.theta = np.pi / (360)
        self.threshold = 150
        self.minLineLength = 100
        self.maxLineGap = 40
        self.eps_dbscan = 0.025 #0.02 - 0.03
        self.eps_dbscan_for_clusters = 0.000036 # 0.000032 - 0.000041

    def criteria(self, cluster):
        if len(cluster.lines) != 2:
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
        Clusters  = list(filter(self.criteria, Clusters))
        Clusters.sort(reverse= True)
        if len(Clusters) < 2:
            return None, None

        return Clusters[0], Clusters[1]


class Last_line(Main_Line_detect):
    def __init__(self, image):
        super().__init__(image)
        self.rho = 1
        self.theta = np.pi / (360)
        self.threshold = 50
        self.minLineLength = 200
        self.maxLineGap = 30
        self.eps_dbscan = 0.0085
        self.eps_dbscan_for_clusters = 0.0001/5

    def detect(self):
        lines = self.find_lines(self.image)
        lines = self.one_line(lines)
        Clusters = self.merge_in_clusters(lines, self.func)
        if len(Clusters) < 1:
            return None
        Clusters = list(filter(self.criteria, Clusters))
        Clusters.sort(reverse=True)
        return Clusters[0]


'''

input_video_path = 'video_cuted/4_right_up_cut.mp4'
output_video_path = 'video/4_right_up_processed.mp4'

cap = cv2.VideoCapture(input_video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Кодек для сохранения видео

# Создаем объект для записи нового видео
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
while True:
    ret, image = cap.read()
    if not ret:
        break

    clustered_frame = image

    Detector = Main_Line_detect( clustered_frame)
    Main_Line = Detector.detect()
    if not Main_Line == None:
        Main_Line.draw(clustered_frame, color=(0, 255, 0))

    Detector = Long_lines(clustered_frame)
    line1, line2 = Detector.detect()
    if not line1 == None:
        line1.draw(clustered_frame, color=(255, 0, 0))
        line2.draw(clustered_frame, color=(255, 0, 0))



    out.write(clustered_frame)

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Видео сохранено по адресу: {output_video_path}")'''

def point_on_height_line(cluster):
    func = lambda line: -line.c/line.a
    points = [func(line) for line in cluster.lines]
    return points

def left_and_right_point(image, cluster):
    height, width, _ = image.shape
    func = lambda line: -line.c / line.b
    left = min([func(line) for line in cluster.lines])
    func = lambda line: -(line.c + line.a*height)/ line.b
    right = min([func(line) for line in cluster.lines])
    return left, right

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





image = cv2.imread("picture/4_right_up_cut.jpg")

Detector = Main_Line_detect(image)
Main_Line = Detector.detect()
Main_Line.draw(image, color = (0,255,0))
Detector = Long_lines(image)
line1,line2 = Detector.detect()
line1.draw(image, color = (255,0,0))
line2.draw(image, color = (255,0,0))


#дальше идет какой-то бред
angels = []

points1 = intersections_clusters(Main_Line,line1)
points2 = intersections_clusters(Main_Line,line2)

points1 = list((filter(Corrcet(image).point, points1)))
points2 = list((filter(Corrcet(image).point, points2)))

draw_point(image, points1)
draw_point(image, points2)
draw_center(image, points1)
draw_center(image, points2)


img_for_print = cv2.resize(image, (0,0), fx=0.5, fy=0.5)
cv2.imshow('Cluster', img_for_print)
cv2.waitKey()

height, width, _ = image.shape

#-------------------------------------------------------------------------------------------------
up_points = point_on_height_line(line1) + point_on_height_line(line2)
down_points = intersections_clusters(line1, Main_Line) + intersections_clusters(line2, Main_Line)
left_up_point = (min(up_points), 0)
right_up_point = (max(up_points), 0)
right_down_point = nearest_point([width, height], down_points)
left_down_point = nearest_point([0, height], down_points)
#-------------------------------------------------------------------------------------------------

conv = lambda my_list: (my_list[0], my_list[1])
src_points = [left_up_point, right_up_point, conv(right_down_point),conv(left_down_point) ] #первые две фиксированы

# Координаты, куда мы хотим проецировать эти точки (вид сверху)
# Обычно задаётся прямоугольником
print(width, height)
width = int(right_down_point[0] - left_down_point[0])
dst_points = [(0, 0), (width, 0), (width, height),(0, height) ]


output_size = (width, height)  # Ширина x Высота

# Выполняем преобразование
image = warp_perspective_to_top_view(image, src_points, dst_points, output_size)

#увеличиваем контраст изображения
contrast = 5
brightness = int(round(255*(1-contrast)/2))
image = cv2.addWeighted(image, contrast, image, 0, brightness)

#а вот ту может для надежности просто все точки все-таки искать?
Detector = Last_line(image)
cluster = Detector.detect()
cluster.draw(image, color = (0,255,0))

left, right = left_and_right_point(image, cluster)

src_points = [(0, left), (width, right), (width, height),(0, height) ]
dst_points = [(0, 0), (width, 0), (width, height),(0, height) ]
output_size = (width, height)
image = warp_perspective_to_top_view(image, src_points, dst_points, output_size)

# Сохраняем и показываем результат
top_view_image = cv2.resize(image, (0,0), fx=0.5, fy=0.5)
cv2.imshow('Cluster', top_view_image)
cv2.waitKey()

cv2.destroyAllWindows()






