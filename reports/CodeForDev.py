from dataclasses import dataclass
import time
import numpy as np
import cv2
import typing as tp


@dataclass
class FieldParam:
    corner_tag_size_m: float
    corner_tag_border_m: float
    field_length_m: float
    field_width_m: float
    border_size_m: float = 0.17


class RuntimeCalibration:
    """
    Handle the camera field calibration
    """
    # Checked
    def __init__(
        self, logger, intrinsic, distortion_coef, field_param: FieldParam, aruco_dict_id
    ) -> None:
        self.is_update_mse = False
        self.logger = logger  # logging.getLogger("field")
        self.calibration_status = False
        self.reprojection_errors_log = []
        self.field_param: FieldParam = field_param # Информация о поле
        self.configure_field() # высчитывает теоретическое положение меток
        self.intrinsic = intrinsic # считываем матрицу камеры
        self.distortion_coef = distortion_coef # считываем коэффициенты дистросии камеры
        self.should_calibrate = True

        self.aruco_dictionary = cv2.aruco.Dictionary_get(aruco_dict_id)
        self.aruco_parameters = cv2.aruco.DetectorParameters_create()
        self.arucoItems = {
            # Corners
            0: ["c1", (128, 128, 0)],
            1: ["c2", (128, 128, 0)],
            2: ["c3", (128, 128, 0)],
            3: ["c4", (128, 128, 0)],
            4: ["c5", (128, 128, 0)],
            5: ["c6", (128, 128, 0)],
        }

    # Checked
    def configure_field(self):
        '''
        Вызывается в конструкторе
        Рассчитывается предположительное положение углов поля

        :return:
        '''
        self.corner_field_positions = {}
        # Пытаемся найти теоретическое положение центра метки
        for c, sx, sy in (
            ["c1", 0, 1],
            ["c2", 0, -1],
            ["c3", -1, 1],
            ["c4", -1, -1],
            ["c5", 1, 1],
            ["c6", 1, -1],
        ):  # координата x центра
            cX = sx * (
                self.field_param.field_length_m / 2
                + (self.field_param.corner_tag_size_m / 2)
                + self.field_param.corner_tag_border_m
            )
            # координата y центра
            cY = sy * (
                self.field_param.field_width_m / 2
                + (self.field_param.corner_tag_size_m / 2)
                + self.field_param.corner_tag_border_m
            )

            # углы метки
            self.corner_field_positions[c] = [
                # Top left
                (
                    cX + self.field_param.corner_tag_size_m / 2,
                    cY + self.field_param.corner_tag_size_m / 2,
                ),
                # Top right
                (
                    cX + self.field_param.corner_tag_size_m / 2,
                    cY - self.field_param.corner_tag_size_m / 2,
                ),
                # Bottom right
                (
                    cX - self.field_param.corner_tag_size_m / 2,
                    cY - self.field_param.corner_tag_size_m / 2,
                ),
                # Bottom left
                (
                    cX - self.field_param.corner_tag_size_m / 2,
                    cY + self.field_param.corner_tag_size_m / 2,
                ),
            ]
        # self.logger.info(f"corner field positions: {self.corner_field_positions}")
        # Position of corners on the image
        self.corner_gfx_positions: dict[str, tp.Any] = {}

        # Is the field calibrated ?
        self.is_calibrated = False

        # Do we see the whole field ?
        self.see_whole_field = False

        # Extrinsic (4x4) transformations
        self.extrinsic_mat = None

        # Camera intrinsic and distortion
        self.intrinsic = None
        self.distortion = None
        self.errors = 0
    # Checked
    def calibrated(self):
        # Просто функция, что возвращает статус калибровки
        #  бесполоезна, так как нигде не меняется статус на True
        # recalculate calibration status
        return self.calibration_status

    def set_corner_position(self, corner: str, corners: list):
        """
        Sets the position of a corner

        :param str corner: the corner name (c1, c2, c3 or c4)
        :param list corners: the corner position
        """
        self.corner_gfx_positions[corner] = corners

    def update_calibration(self, image, image_debug=None):
        """
        If the field should be calibrated, compute a calibration from the detected corners.
        This will use corner positions previously passed with set_corner_positions.

        :param image: the (OpenCV) image used for calibration
        """
        self.is_update_mse = False
        if len(self.corner_gfx_positions) >= 3 and self.should_calibrate:
            # Computing point-to-point correspondance
            object_points = []
            graphics_positions = []
            for key in self.corner_gfx_positions:
                for gfx, real in zip(
                    self.corner_gfx_positions[key], self.corner_field_positions[key]
                ):
                    #######################
                    if not (image_debug is None):
                        projected_position = self.pixel_to_position(gfx)
                        reprojection_error = np.linalg.norm(np.array(real) - np.array(projected_position))
                        cv2.circle(image_debug, tuple(projected_position[:2]), 5, (0, 0, 255), -1)
                        cv2.putText(
                            image_debug,
                            f"{reprojection_error:.3f}",
                            tuple(projected_position[:2]),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            1,
                        )
                    ######################3

                    graphics_positions.append(gfx)
                    object_points.append([*real, 0.0])

            object_points = np.array(object_points, dtype=np.float32)
            graphics_positions = np.array(graphics_positions, dtype=np.float32)

            # Calibrating camera
            flags = (
                cv2.CALIB_USE_INTRINSIC_GUESS
                + cv2.CALIB_FIX_FOCAL_LENGTH
                + cv2.CALIB_FIX_PRINCIPAL_POINT
            )

            # No distortion
            flags += cv2.CALIB_FIX_TANGENT_DIST
            flags += (
                cv2.CALIB_FIX_K1
                + cv2.CALIB_FIX_K2
                + cv2.CALIB_FIX_K3
                + cv2.CALIB_FIX_K4
                + cv2.CALIB_FIX_K5
            )
            ret, _, _, rvecs, tvecs = cv2.calibrateCamera(
                [object_points],
                [graphics_positions],
                image.shape[:2][::-1],
                np.array(self.intrinsic, dtype=np.float32),
                self.distortion_coef,
                flags=flags,
            )

            # Computing extrinsic matrices
            transformation = np.eye(4)
            transformation[:3, :3], _ = cv2.Rodrigues(rvecs[0])
            transformation[:3, 3] = tvecs[0].T

            self.extrinsic_mat_inv = transformation

            self.extrinsic_mat = np.eye(4)
            self.extrinsic_mat = np.linalg.inv(transformation)
            # We are now calibrated
            self.is_calibrated = True
            self.should_calibrate = False
            self.errors = 0

            # Checking if we can see the whole fields
            # self.logger.info(f"IMAGE SHAPE{image.shape}")
            image_height, image_width, _ = image.shape
            image_points = []
            self.see_whole_field = True
            for sx, sy in [(-1, 1), (1, 1), (1, -1), (-1, -1)]:
                x = sx * (
                    (self.field_param.field_length_m / 2)
                    + self.field_param.border_size_m
                )
                y = sy * (
                    (self.field_param.field_width_m / 2)
                    + self.field_param.border_size_m
                )

                img = self.position_to_pixel([x, y, 0.0])
                image_points.append((int(img[0]), int(img[1])))

                if (
                    img[0] < 0
                    or img[0] > image_width
                    or img[1] < 0
                    or img[1] > image_height
                ):
                    self.see_whole_field = False

        # We check that calibration is consistent, this can happen be done with only a few corners
        # The goal is to avoid recalibrating everytime for performance reasons
        if len(self.corner_gfx_positions) >= 3:
            if self.is_calibrated:
                has_error = False
                reprojection_errors = []
                for key in self.corner_gfx_positions:
                    for gfx, real in zip(
                        self.corner_gfx_positions[key], self.corner_field_positions[key]
                    ):
                        projected_position = self.pixel_to_position(gfx)
                        reprojection_distance = np.linalg.norm(
                            np.array(real) - np.array(projected_position)
                        )
                        if reprojection_distance > 0.025:
                            reprojection_errors.append(reprojection_distance)
                            has_error = True
                self.is_update_mse = True
                mean_error = np.mean(reprojection_errors)
                self.logger.info(f"Mean reprojection error: {mean_error}")
                self.reprojection_errors_log.append(mean_error)

                if not has_error:
                    self.errors = 0
                else:
                    self.errors += 1
                    if self.errors > 8:
                        # self.logger.warning("Calibration seems wrong, re-calibrating")
                        # self.logger.warning(
                        #     f"Reprojection threshold is 0.025 corner error is {reprojection_errors}"
                        # )
                        self.should_calibrate = True
        # self.logger.debug(f"HUIHUIHUI!!!!{self.extrinsic_mat}")
        self.corner_gfx_positions = {}

    def detect_markers(self, image, image_debug=None):
        (corners, ids, rejected) = cv2.aruco.detectMarkers(
            image, self.aruco_dictionary, parameters=self.aruco_parameters
        )

        new_markers = {}

        if len(corners) > 0:
            for markerCorner, markerID in zip(corners, ids.flatten()):
                if markerID not in self.arucoItems:
                    continue

                corners = markerCorner.reshape((4, 2))

                # Draw the bounding box of the ArUCo detection
                item = self.arucoItems[markerID][0]

                if item[0] == "c":
                    self.set_corner_position(item, corners)

                if image_debug is not None:
                    (topLeft, topRight, bottomRight, bottomLeft) = corners
                    topRight = (int(topRight[0]), int(topRight[1]))
                    bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                    bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
                    topLeft = (int(topLeft[0]), int(topLeft[1]))
                    itemColor = self.arucoItems[markerID][1]
                    if True:
                        cv2.line(image_debug, topLeft, topRight, itemColor, 2)
                        cv2.line(image_debug, topRight, bottomRight, itemColor, 2)
                        cv2.line(image_debug, bottomRight, bottomLeft, itemColor, 2)
                        cv2.line(image_debug, bottomLeft, topLeft, itemColor, 2)

                        # Compute and draw the center (x, y)-coordinates of the
                        # ArUco marker
                        cX = int((topLeft[0] + bottomRight[0]) / 2.0)
                        cY = int((topLeft[1] + bottomRight[1]) / 2.0)
                        cv2.circle(image_debug, (cX, cY), 4, (0, 0, 255), -1)
                        fX = int((topLeft[0] + topRight[0]) / 2.0)
                        fY = int((topLeft[1] + topRight[1]) / 2.0)
                        cv2.line(
                            image_debug,
                            (cX, cY),
                            (cX + 2 * (fX - cX), cY + 2 * (fY - cY)),
                            (0, 0, 255),
                            2,
                        )

                        text = item
                        if item.startswith("blue") or item.startswith("green"):
                            text = item[-1]
                        cv2.putText(
                            image_debug,
                            text,
                            (cX - 4, cY + 4),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            6,
                        )
                        cv2.putText(
                            image_debug,
                            text,
                            (cX - 4, cY + 4),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            itemColor,
                            2,
                        )

                if self.calibrated() and item[0] != "c":
                    new_markers[item] = self.pose_of_tag(corners)
                    self.last_updates[item] = time.time()

        self.update_calibration(image, image_debug)

    def position_to_pixel(self, pos: list) -> list:
        """
        Given a position (3D, will be assumed on the ground if 2D), find its position on the screen

        :param list pos: position in field frame (2D or 3D)
        :return list: position on the screen
        """
        if len(pos) == 2:
            # If no z is provided, assume it is a ground position
            pos = [*pos, 0.0]

        point_position_camera = self.field_to_camera(pos)
        position, J = cv2.projectPoints(
            point_position_camera,
            np.zeros(3),
            np.zeros(3),
            self.intrinsic,
            self.distortion,
        )
        position = position[0][0]

        return [int(position[0]), int(position[1])]

    def field_to_camera(self, point: list) -> np.ndarray:
        """
        Transforms a point from field frame to camera frame

        :param list point: point in field frame (3d)
        :return np.ndarray: point in camera frame (3d)
        """
        return (self.extrinsic_mat @ np.array([*point, 1.0]))[:3]

    def camera_to_field(self, point: list) -> np.ndarray:
        """
        Transforms a point from camera frame to field frame

        :param list point: point in camera frame (3d)
        :return np.ndarray: point in field frame (3d)
        """
        return (self.extrinsic_mat_inv @ np.array([*point, 1.0]))[:3]

    def pixel_to_position(self, pixel: list, z: float = 0) -> list:
        """
        Transforms a pixel on the image to a 3D point on the field, given a z

        :param list pos: pixel
        :param float z: the height to intersect with, defaults to 0
        :return list: point coordinates (x, y)
        """

        # Computing the point position in camera frame
        point_position_camera = cv2.undistortPoints(
            np.array(pixel), self.intrinsic, self.distortion
        )[0][0]

        # Computing the point position in the field frame and solving for given z
        point_position_field = self.camera_to_field([*point_position_camera, 1.0])
        camera_center_field = self.camera_to_field(np.array([0.0, 0.0, 0.0]))
        delta = point_position_field - camera_center_field
        length = (z - camera_center_field[2]) / delta[2]

        return list(camera_center_field + length * delta)[:2]

    @property
    def extrinsic(self) -> np.ndarray:
        return self.extrinsic_mat
