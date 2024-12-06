import warnings
warnings.filterwarnings("ignore")
from functions import *
from drawing_functions import *

def inverse(point, matrix):
    point_img2_homogeneous = np.array([point[0], point[1], 1])
    point_img1_homogeneous = np.dot(matrix, point_img2_homogeneous)
    x = point_img1_homogeneous[0] / point_img1_homogeneous[2]
    y = point_img1_homogeneous[1] / point_img1_homogeneous[2]
    return np.array([x, y])

image = cv2.imread("picture/4_right_up_cut.jpg")

Detector = Main_Line_detect(image)
Main_Line = Detector.detect()
Main_Line.draw(image, color = (0,255,0))
Detector = Long_lines(image)
line1,line2 = Detector.detect()
line1.draw(image, color = (255,0,0))
line2.draw(image, color = (255,0,0))


points1 = intersections_clusters(Main_Line,line1)
points2 = intersections_clusters(Main_Line,line2)

points1 = list((filter(Corrcet(image).point, points1)))
points2 = list((filter(Corrcet(image).point, points2)))

draw_point(image, points1)
draw_point(image, points2)
draw_center(image, points1)
draw_center(image, points2)

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
down_points = points1 + points2
left_up_point = (min(up_points), 0)
right_up_point = (max(up_points), 0)
right_down_point = nearest_point([width, height], down_points)
left_down_point = nearest_point([0, height], down_points)
#-------------------------------------------------------------------------------------------------

conv = lambda my_list: (my_list[0], my_list[1])
src_points = [left_up_point, right_up_point, conv(right_down_point),conv(left_down_point) ]

print(height, width)
width = int(max(up_points) - min(up_points))
height = int(4*width/3)
print(height, width)

dst_points = [(0, 0), (width, 0), (width, height),(0, height) ]

output_size = (width, height)  # Ширина x Высота

# Выполняем преобразование
image_help, matrix = warp_perspective_to_top_view(image, src_points, dst_points, output_size)
print(np.shape(matrix), matrix)
contrast = 5
brightness = int(round(255 * (1 - contrast) / 2))
image_help = cv2.addWeighted(image_help, contrast, image_help, 0, brightness)

cv2.imshow('Cluster', image_help)
cv2.waitKey()


Detector = Last_line(image_help)
clusters = Detector.detect()

cluster = clusters[0]
cluster.draw(image_help, color = (0,255,0))

cv2.imshow('Cluster', image_help)
cv2.waitKey()


matrix = np.linalg.inv(matrix)
right,left = left_and_right_point(image_help, [cluster])
print(left, right)
#print(inverse([0, left],matrix), inverse([width, right],matrix))
point_l = inverse([0, left],matrix)
point_r = inverse([width, right],matrix)
print(point_l, point_r)
print(right_down_point, left_down_point)
cv2.circle(image, (int(point_l[1]),int(point_l[0])), radius=15, color=(255, 0, 0), thickness=-1)
cv2.circle(image, (int(point_r[1]),int(point_r[0])), radius=15, color=(255, 0, 0), thickness=-1)
cv2.line(image,(int(point_l[0]),int(point_l[1])), (int(point_r[0]),int(point_r[1])), (0,255,0), 5)
warped_image = cv2.warpPerspective(image_help, matrix, (1920, 1080))
cv2.imshow('Cluster', image )
cv2.waitKey()

x= 300
y = 400

rect_3d = np.array([
    [0.0, 0.0, 0.0],  # Точка 1
    [x, 0.0, 0.0],  # Точка 2
    [x,  y, 0.0],  # Точка 3
    [0.0,y, 0.0]   # Точка 4
], dtype=np.float32)

# Соответствующие 2D-точки на изображении (например, координаты пикселей)
rect_2d = np.array([
    point_l,  # Точка 1
    point_r,  # Точка 2
    right_down_point, # Точка 3
    left_down_point,   # Точка 4
], dtype=np.float32)

camera_matrix = np.array([
    [1920, 0, 960],  # fx, 0, cx
    [0, 1080, 540],  # 0, fy, cy
    [0, 0, 1]       # 0, 0, 1
], dtype=np.float32)

# Коэффициенты дисторсии (здесь предполагаем их нулевыми, если известны — укажите их)
dist_coeffs = np.zeros((4, 1), dtype=np.float32)

# Решаем задачу PnP
success, rvec, tvec = cv2.solvePnP(rect_3d, rect_2d, camera_matrix, dist_coeffs)
# Печатаем результат
if success:
    print("Rotation Vector (rvec):")
    print(rvec)
    print("\nTranslation Vector (tvec):")
    print(tvec)

    # Переходим из координат прямоугольника в мировую систему (камера относительно центра)
    # Для этого центр прямоугольника: (0.5, 0.25, 0) в системе объекта
    center_3d = np.array([[x/2, y/2, 0]], dtype=np.float32)

    # Преобразуем вектор вращения в матрицу
    R, _ = cv2.Rodrigues(rvec)
    for vec in rect_3d:

        print(np.shape(tvec))
        print(vec,-R.T @ tvec)


    print("\nCamera Position in Rectangle's Coordinate System:")

else:
    print("Не удалось вычислить положение камеры.")

