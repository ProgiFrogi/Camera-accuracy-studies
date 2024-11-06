from ArUcoDetecion.get_position import get_markerB_position_from_markerA
from utils.cut_image import get_cut_frame_from_frame
from utils.metrics import RMSE, MSE
import numpy as np
import cv2
from matplotlib import pyplot as plt
from ArUcoDetecion.detect_marks import detect_markers_return_ids


def get_RMSE_from_2_cameras_by_images_many_marks(main_marker,
                                                 img_1, img_2,
                                                 aruco_type, marker_length,
                                                 camera_1_setting, camera_2_setting):
    camera_1_matrix = camera_1_setting['camera_matrix']
    camera_2_matrix = camera_2_setting['camera_matrix']
    camera_1_dist_coef = camera_1_setting['dist_coef']
    camera_2_dist_coef = camera_2_setting['dist_coef']

    markers_1 = detect_markers_return_ids(img_1, aruco_type)
    markers_2 = detect_markers_return_ids(img_2, aruco_type)
    list1 = list(map(lambda x: x[0], markers_1.tolist()))
    list2 = list(map(lambda x: x[0], markers_2.tolist()))
    inter = list(set(list1).intersection(list2))
    if len(inter) == 0:
        return 0

    result = 0
    for marker in inter:
        result += get_MSE_from_2_cameras_by_images(main_marker, marker,
                                                   img_1, img_2,
                                                   aruco_type, marker_length,
                                                   camera_1_setting, camera_2_setting)
    return (result**0.5, len(inter))

def example_get_RMSE_from_2_cameras_by_images_many_marks():
    marker_id_1 = 0
    marker_id_2 = 5
    aruco_type = cv2.aruco.DICT_5X5_250
    camera_matrix = np.array(
        [[800, 0, 640],
         [0, 800, 360],
         [0, 0, 1]], dtype=np.float64)
    dist_coef = np.array([0, 0, 0, 0, 0], dtype=np.float64)

    camera_1_setting = {'camera_matrix': camera_matrix, 'dist_coef': dist_coef}
    camera_2_setting = {'camera_matrix': camera_matrix, 'dist_coef': dist_coef}

    errors = []
    time = []
    marks_counter = []
    i = 0

    path = '../materials_part1/1.mkv'
    cam = cv2.VideoCapture(path)
    while True:
        success, frame = cam.read()
        if not success:
            cam = cv2.VideoCapture(path)
            continue
        frame1 = get_cut_frame_from_frame(frame, 1)
        frame2 = get_cut_frame_from_frame(frame, 2)

        rmse, marks = get_RMSE_from_2_cameras_by_images_many_marks(marker_id_1,
                                                 frame1, frame2,
                                                 aruco_type, 0.27,
                                                 camera_1_setting, camera_2_setting)

        cv2.imshow('frame', frame1)
        print(rmse)
        errors.append(rmse)
        time.append(i)
        marks_counter.append(marks)

        i += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    fig, axes = plt.subplots(nrows=1, ncols=2)

    axes[0].plot(time, errors)
    axes[0].set_title('RMSE 2 camers')
    axes[0].set_xlabel('cadre')
    axes[0].set_ylabel('Error')

    axes[1].plot(time, marks_counter)
    axes[1].set_title('marks in camers')
    axes[1].set_xlabel('cadre')
    axes[1].set_ylabel('count marks in cadre')

    plt.show()


def get_MSE_from_2_cameras_by_images(marker_0_id, marker_1_id,
                                     img_1, img_2,
                                     aruco_type, marker_length,
                                     camera_1_setting, camera_2_setting):
    arrays = get_arrays_from_2_cameras_by_images(marker_0_id, marker_1_id,
                                                 img_1, img_2,
                                                 aruco_type, marker_length,
                                                 camera_1_setting, camera_2_setting)
    return MSE(arrays[0], arrays[1])


def get_RMSE_from_2_cameras_by_images(marker_0_id, marker_1_id,
                                      img_1, img_2,
                                      aruco_type, marker_length,
                                      camera_1_setting, camera_2_setting):
    arrays = get_arrays_from_2_cameras_by_images(marker_0_id, marker_1_id,
                                                 img_1, img_2,
                                                 aruco_type, marker_length,
                                                 camera_1_setting, camera_2_setting)
    return RMSE(arrays[0], arrays[1])


def get_arrays_from_2_cameras_by_images(marker_0_id, marker_1_id,
                                        img_1, img_2,
                                        aruco_type, marker_length,
                                        camera_1_setting, camera_2_setting):
    camera_1_matrix = camera_1_setting['camera_matrix']
    camera_2_matrix = camera_2_setting['camera_matrix']
    camera_1_dist_coef = camera_1_setting['dist_coef']
    camera_2_dist_coef = camera_2_setting['dist_coef']

    position_1_marker_1_from_marker_0 = get_markerB_position_from_markerA(marker_0_id, marker_1_id,
                                                                          img_1, aruco_type,
                                                                          camera_1_matrix, camera_1_dist_coef,
                                                                          marker_length)
    position_2_marker_1_from_marker_0 = get_markerB_position_from_markerA(marker_0_id, marker_1_id,
                                                                          img_2, aruco_type,
                                                                          camera_2_matrix, camera_2_dist_coef,
                                                                          marker_length)

    return position_1_marker_1_from_marker_0, position_2_marker_1_from_marker_0


def example_get_RMSE_from_2_cameras_by_images():
    marker_id_1 = 0
    marker_id_2 = 5
    aruco_type = cv2.aruco.DICT_5X5_250
    camera_matrix = np.array(
        [[800, 0, 640],
         [0, 800, 360],
         [0, 0, 1]], dtype=np.float64)
    dist_coef = np.array([0, 0, 0, 0, 0], dtype=np.float64)

    camera_1_setting = {'camera_matrix': camera_matrix, 'dist_coef': dist_coef}
    camera_2_setting = {'camera_matrix': camera_matrix, 'dist_coef': dist_coef}

    errors = []
    time = []
    i = 0

    path = '../materials_part1/1.mkv'
    cam = cv2.VideoCapture(path)
    while True:
        success, frame = cam.read()
        if not success:
            cam = cv2.VideoCapture(path)
            continue
        frame1 = get_cut_frame_from_frame(frame, 1)
        frame2 = get_cut_frame_from_frame(frame, 2)

        rmse = get_RMSE_from_2_cameras_by_images(marker_id_1, marker_id_2,
                                                 frame1, frame2,
                                                 aruco_type, 0.27,
                                                 camera_1_setting, camera_2_setting)

        cv2.imshow('frame', frame1)
        print(rmse)
        errors.append(rmse)
        time.append(i)
        i += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    plt.plot(time, errors)
    plt.title('RMSE 2 camers')
    plt.xlabel('cadre')
    plt.ylabel('Error')
    plt.show()


if __name__ == "__main__":
    example_get_RMSE_from_2_cameras_by_images_many_marks()
