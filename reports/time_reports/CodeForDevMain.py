from runtime_calibraiton import FieldParam
from runtime_calibraiton import RuntimeCalibration as RCalib
from print_errors import print_errors
import logging
import numpy as np
import cv2

def DetectMarkersHandler(calib, path):
    cam = cv2.VideoCapture(path)
    errors = []
    time = []
    i = 0
    while True:
        success, frame = cam.read()
        if not success:
            cam = cv2.VideoCapture(path)
            continue
        frame_deb = frame.copy()
        calib.detect_markers(frame, frame_deb)
        if calib.is_update_mse:
            i += 1
        cv2.imshow('frame', frame_deb)
        time.append(i)
        if cv2.waitKey(2) & 0xFF == ord('q'):
            break
        if i != len(calib.reprojection_errors_log):
            print_errors(i)
            break
        elif (i != 0):
            errors.append(calib.reprojection_errors_log[i-1])
    return errors, time

if __name__ == '__main__':
    logging.basicConfig(
        filename="logfile.log",
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger("field")
    intrinsic = np.array([[596.66029956, 0, 316.90197034],
                          [0, 597.98886626, 266.08610358],
                          [0, 0, 1]], dtype=float)
    distortion_coef = np.array([0.07776152, -0.2656504, 0.00959812, 0.00041337, 0.31470695], dtype=float)

    fp = FieldParam(
        corner_tag_size_m=0.175,  # Размер метки 7 см
        corner_tag_border_m=0.175,  # Граница метки 0 см
        field_length_m=1.0,  # Длина поля 32.6 см
        field_width_m=1.0,  # Ширина поля 32.6 см
        border_size_m=0.0
    )
    aruco_dict_id = cv2.aruco.DICT_6X6_250


    calib = RCalib(
        logger=logger,
        intrinsic=intrinsic,
        distortion_coef=distortion_coef,
        field_param=fp,
        aruco_dict_id=aruco_dict_id
    )
    num = 12

    errors, time = DetectMarkersHandler(calib=calib, path=f'video/{num}.avi')
    print(len(errors), len(time))
    print_errors(
        errors,
        time,
        title=f"Configuration {num}",
        xlabel="Time",
        ylabel="MSE"
    )

