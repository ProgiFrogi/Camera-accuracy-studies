import numpy as np
import warnings
warnings.filterwarnings("ignore")
from functions import *
from drawing_functions import *

bad_v, bad_m, bad_height, error_size, flag_err = 0, 0, 0, 0, 0

color = [(randint(10, 255), randint(10, 255), randint(10, 255))]*100
last_img = np.zeros((1920,1080, 3))
img_count = 0
count = 0

def get_bird_vision(image):

    global bad_v, bad_m, bad_height, img_count
    img_count += 1
    print("r", img_count)
    clustered_frame = image
    Detector = Main_Line_detect(clustered_frame)
    Main_Line = Detector.detect()

    if Main_Line == None:
        bad_m +=1
        print("Bad Main Line")
        return False, 0
    else:
        Main_Line.draw(image, color = (0,255,0))

    Detector = Long_lines(clustered_frame)
    line1, line2 = Detector.detect()

    if line1 == None:
        bad_v +=1
        print("Bad vertical line")
        return False, 0
    else:
        line1.draw(image, color=(255, 0, 0))
        line2.draw(image, color=(255, 0, 0))

    points1 = intersections_clusters(Main_Line, line1)
    points2 = intersections_clusters(Main_Line, line2)
    points1 = list((filter(Corrcet(image).point, points1)))
    points2 = list((filter(Corrcet(image).point, points2)))

    height, width, _ = image.shape

    up_points = point_on_height_line(line1) + point_on_height_line(line2)
    down_points = points1 + points2
    left_up_point = (min(up_points), 0)
    right_up_point = (max(up_points), 0)
    right_down_point = nearest_point([width, height], down_points)
    left_down_point = nearest_point([0, height], down_points)

    conv = lambda my_list: (my_list[0], my_list[1])
    src_points = [left_up_point, right_up_point, conv(right_down_point), conv(left_down_point)]

    width = int(right_down_point[0] - left_down_point[0])
    width = 400
    height = int(width*4/3)
    dst_points = [(0, 0), (width, 0), (width, height), (0, height)]

    output_size  = (width, height)
    print(width, height)# Ширина x Высота

    # Выполняем преобразование
    image_help, matrix = warp_perspective_to_top_view(image, src_points, dst_points, output_size)

    # увеличиваем контраст изображения
    contrast = 5
    brightness = int(round(255 * (1 - contrast) / 2))
    image_help = cv2.addWeighted(image_help, contrast, image_help, 0, brightness)


    Detector = Last_line(image_help)
    clusters = Detector.detect()
    if type(clusters) == bool:
        bad_height += 1
        print("fail")
        return False, 0
    #clusters[0].draw(image, color=(0, 255, 0)) # тут нужно что-то другое

    matrix = np.linalg.inv(matrix)
    right, left = left_and_right_point(image_help, clusters)
    print(left, right, "координаты сбоку")
    point_l = inverse([0, left], matrix)
    point_r = inverse([width, right], matrix)
    print(point_l , point_r, "координаты сбоку но в старой картинке")
    cv2.line(image, (int(point_l[0]), int(point_l[1])), (int(point_r[0]), int(point_r[1])), (0, 255, 0), 5)


    img_for_print = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow('Cluster', img_for_print)
    cv2.waitKey()


    print(np.shape(image), "fff")

    return True, image



if __name__ == '__main__':
    input_video_path = 'video_cuted/4_right_up_cut.mp4'
    output_video_path = 'video/4_right_up_processed_last.mp4'

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

        flag, clustered_frame = get_bird_vision(image)
        #print(type(clustered_frame), np.size(clustered_frame))
        if flag and np.size(clustered_frame) > 1080*1500:
            last_img = clustered_frame
        else:
            if np.size(clustered_frame) < 1080*1500:
                error_size +=1
                print("very small")
            if not flag:
                flag_err += 1
            count += 1
            clustered_frame = last_img

        clustered_frame =cv2.resize(clustered_frame, (1920, 1080), interpolation=cv2.INTER_LINEAR)

        out.write(clustered_frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    #прблемы с сохранением видео, скорее всего, разный размер кадров
    print(f"Видео сохранено по адресу: {output_video_path}", count)
    print('Bad detect Main line: ', bad_m)
    print('Bad detect Long lines: ', bad_v)
    print('Bad detect Height line: ', bad_height)
    print('Bad size or flag error: ', error_size, flag_err)


