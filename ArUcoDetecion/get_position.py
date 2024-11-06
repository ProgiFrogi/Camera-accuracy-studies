import cv2
import numpy as np
from utils import get_cut_frame_from_frame
from cv2 import aruco

# Setting of camera
camera_matrix = np.array(
    [[800, 0, 640],
     [0, 800, 360],
     [0, 0, 1]], dtype=np.float64)
dist_coeffs = np.array([0, 0, 0, 0, 0], dtype=np.float64)

def get_camera_position(img, dictionary, parameters, camera_matrix, dist_coeffs, marker_length):
    '''
    :param img - input img:
    :param dictionary - dict ArUco markers:
    :param parameters - params of ArUco marker detector:
    :param camera_matrix - camera calibration matrix :
    :param dist_coeffs - camera distortion coeffs:
    :param marker_length - len of marker:
    :return: frame
    '''
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    markerCorners, markerIds, _ = detector.detectMarkers(img)

    if markerIds is not None:
        obj_points = np.array([
            [-marker_length / 2, marker_length / 2, 0],
            [marker_length / 2, marker_length / 2, 0],
            [marker_length / 2, -marker_length / 2, 0],
            [-marker_length / 2, -marker_length / 2, 0]
        ], dtype=np.float64)

        for i in range(len(markerIds)):
            success, rvec, tvec = cv2.solvePnP(
                obj_points, markerCorners[i][0], camera_matrix, dist_coeffs
            )

            if success:
                R, _ = cv2.Rodrigues(rvec)
                camera_position = -np.dot(R.T, tvec).reshape(3)

                print(f"Положение камеры относительно метки {markerIds[i]}: {camera_position}")

                cv2.drawFrameAxes(img, camera_matrix, dist_coeffs, rvec, tvec, marker_length / 2)

        return img
    else:
        print("Метки не найдены.")
        return img

def get_cam_position_from_mark(path, marker_length = 0.27, aruco_type = aruco.DICT_5X5_1000, part=1):

    dictionary = aruco.getPredefinedDictionary(aruco_type)
    parameters = aruco.DetectorParameters()

    cam = cv2.VideoCapture(path)

    while True:
        success, frame = cam.read()
        if not success:
            cam.release()
            cam = cv2.VideoCapture(path)
        frame = get_cut_frame_from_frame(frame, part)
        output_image = get_camera_position(frame, dictionary, parameters, camera_matrix, dist_coeffs, marker_length)

        cv2.imshow("Detected Markers", output_image)
        cv2.waitKey(1)

def get_markerB_position_from_markerA(marker_A_id, marker_B_id,
                                      image, aruco_type,
                                      camera_matrix, dist_coeffs,
                                      marker_length):
    dictionary = aruco.getPredefinedDictionary(aruco_type)
    parameters = aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    marker_corners, marker_ids, _ = detector.detectMarkers(image)

    if marker_ids is not None and marker_A_id in marker_ids and marker_B_id in marker_ids:
        obj_points = np.array([
            [-marker_length / 2, marker_length / 2, 0],
            [marker_length / 2, marker_length / 2, 0],
            [marker_length / 2, -marker_length / 2, 0],
            [-marker_length / 2, -marker_length / 2, 0]
        ], dtype=np.float64)

        # Get tvec and rvec for marker A
        index_A = np.where(marker_ids == marker_A_id)[0][0]
        success_A, rvec_A, tvec_A = cv2.solvePnP(
            obj_points, marker_corners[index_A][0], camera_matrix, dist_coeffs
        )

        # Get tvec and rvec for marker B
        index_B = np.where(marker_ids == marker_B_id)[0][0]
        success_B, rvec_B, tvec_B = cv2.solvePnP(
            obj_points, marker_corners[index_B][0], camera_matrix, dist_coeffs
        )

        if success_A and success_B:
            # Преобразуем векторы вращения в матрицы вращения
            R_A, _ = cv2.Rodrigues(rvec_A)
            R_B, _ = cv2.Rodrigues(rvec_B)

            # Положение метки B в мировой системе координат
            P_B = np.dot(R_B, np.zeros((3, 1))) + tvec_B

            # Положение метки B относительно метки A
            P_BA = np.dot(R_A.T, (P_B - tvec_A))

            return P_BA.flatten()  # Возвращаем положение метки B относительно метки A

        return None  # Если не нашли метки или не удалось получить tvec и rvec

def output_get_marker5_position_from_marker0():
    path = '../materials_part1/1.mkv'
    cam = cv2.VideoCapture(path)

    dictionary = aruco.getPredefinedDictionary(aruco.DICT_5X5_1000)
    parameters = aruco.DetectorParameters()
    marker_length = 0.27

    while True:
        success, frame = cam.read()
        if not success:
            cam.release()
            cam = cv2.VideoCapture(path)
            continue
        frame = get_cut_frame_from_frame(frame, 1)
        position_relative = get_markerB_position_from_markerA(0, 5, frame, cv2.aruco.DICT_5X5_1000, camera_matrix, dist_coeffs, marker_length)
        print(f'Позиция второй метки относительно первой:{position_relative}')
        position_relative = get_markerB_position_from_markerA(5, 0, frame, cv2.aruco.DICT_5X5_1000, camera_matrix,
                                                              dist_coeffs, marker_length)
        print(f'Позиция первой метки относительно второй:{position_relative}')
        cv2.imshow('frame', frame)
        cv2.waitKey(1)


if __name__ == "__main__":
    path = '../materials_part1/1.mkv'
    img = cv2.VideoCapture(path)
    img = img.read()

    output_get_marker5_position_from_marker0()

