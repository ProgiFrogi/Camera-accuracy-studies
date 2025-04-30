import numpy as np
import cv2
import math

def homography_matrix(segs):
    """deprecated"""
    def homography_matrix_by_4_vec(vecs):  # vecs = [a,b,c,d],  a,b -> (1,0) c,d ->(0,1)
        for i in range(len(vecs)):
            vecs[i] = np.append(vecs[i], 1)
        # notice that a-b = alpha is eigenvector with value 0 as well as c-d = beta
        # than:
        # A = V*L*V^(-1)
        # L = [[0,0,0],[0,0,0],[0,0,x]]
        # and V = [alpha,beta,alpha cross beta]
        alpha = vecs[0] - vecs[1]
        beta = vecs[2] - vecs[3]
        gamma = np.cross(alpha, beta)

    # idea v2: try all matricies and choose one which maximizes metric.
    # for all  segments generate vectors length of 1
    # for all pairs of this vectors find matrix that minimizes some function
    vectors = list()
    for i in segs:
        point = (i[0] - i[1]) / np.linalg.norm(i[0] - i[1])
        vectors.append(point)
    vectors_v2 = np.array(vectors).reshape((len(vectors), 1, 2))
    # def get_metric(vecs,)
    best_val = -1e9
    best_m = np.identity(3)
    for i in range(len(vectors)):
        for j in range(len(vectors)):
            if i != j:
                for k in range(len(vectors)):
                    if i != k and j != k:
                        for f in range(len(vectors)):
                            if f != k and f != j and f != k:
                                mat, mask = cv2.findHomography(
                                    np.array([vectors[f], vectors[i], vectors[j], vectors[k]]),
                                    np.array([[0, 100], [100, 0], [100, 0], [0, 100]]))
                                # print(vectors.shape,mat.shape)
                                vecs = cv2.perspectiveTransform(vectors_v2, mat).reshape((len(vectors), 2))
                                groups_size = [0, 0, 0]  # hor,vert,bad
                                for t in vecs:
                                    t_p = t / np.linalg.norm(t)
                                    if np.abs(np.abs(np.dot(t_p, np.array([1, 0]))) - 1) < 1 / 8:
                                        groups_size[0] += 1
                                    elif np.abs(np.abs(np.dot(t_p, np.array([0, 1]))) - 1) < 1 / 8:
                                        groups_size[1] += 1
                                    else:
                                        groups_size[2] += 1
                                val = groups_size[0] * groups_size[1] - groups_size[2]
                                # print(groups_size)
                                if val > best_val:
                                    print(groups_size)
                                    # print()
                                    best_val = val
                                    best_m = mat

    print(best_val, best_m)

    return best_m



# homograpy matrix, for transfromation that minimizes sum of distances between each end of segments and line that goes trough center of segment and "horizontal" or "vertical" point at the horizon, where this points represents points to which are all parallel horizontal/vertical lines are converging
def homography_matrix_v2(hor_point1 ,hor_point2 ,center_pos ,only_points = False,angle_between_lines=math.pi/2)  :  # ,fov_px,fov_angle):
    """
    given 2 horizon points and center of image generates homography matrix that convert image from perspective view to bird-view

    :param hor_point1: horizon point for vertical lines
    :param hor_point2: horizon point for horizontal lines
    :param center_pos: center of image
    :param only_points: if True, return only 2 ordered set of points that are converted from on to another
    :param angle_between_lines: angle(in radians) between "horizontal" and "vertical" lines. in some cases angle=pi/2 may not work and you will need smaller angle
    :return:
    """
    # hereinafter assuming center of image is (0,0), i.e. (0,0) is point through perpendicular from "focal point" is goes
    # todo:convert edges to p1 and p2 solution: watch in deepseek, has intersting way of searching with ransak. also points p1 and p2 are called vanishing points and there are some papers in internet about them.
    def find_distance_fp_to_plane_v1(fov_pixels,
                                     field_of_view):  # hor[0]*x+hor[1]*y = hor[2] ; field_of_view in degrees - angle at which edges of image displayed, fov_pixels is number of pixels corresponding to fov
        return fov_pixels / (2 * math.tan(field_of_view / 2))

    # do not work if horizontal lines are parallel to horizon.
    # assuming that perpendicular line from "focal point" (point from which rays are shoot in point-and-plane approximation of camera) onto "focal plane" (plane of image in approximation ...) goes through center of image
    def find_distance_fp_to_plane_v2(p1, p2, p3,
                                     angle=math.pi / 2) -> float:  # p1 is horizont point for vertical lines,p2 is horizont point for horizontal lines, p3 is nearest point on horizon to center of image. returns distance in pixels
        # let's derive formula:
        # look on triangle FP, p1,p2:
        # let segment p1-p3 called a, p3-p2 called b, angle fp-p1-p2 is beta, angle p2-fp-p1 is alpha, segment fp-p3 is h
        # angle fp-p3-p1 is 90 deg based on assumption
        # then tg(beta) = h/a, tg(180-alpha-beta) = -tg(alpha + beta) = h/b
        # then -tg(beta)/tg(alpha+beta) = b/a
        # then -tg(beta)/((tg(alpha)+tg(beta))/(tg(1-tg(alpha)tg(beta))) = b/a
        # then tg(beta) = x, tg(alpha) = c
        # (-x+c*x**2)/(x+c) = b/a
        # then -x+c*x**2 -x*b/a-c*b/a = 0
        # same as: c*x**2 +(-1-b/a)*x-c*b/a = 0
        # d = 1+2*b/a+(b/a)**2+4*c**2*b/a
        # x = ...: x is positive
        a = np.linalg.norm(p1 - p3)
        b = np.linalg.norm(p2 - p3)
        c_p = math.tan(
            math.pi / 2 - angle)  # c_p = ctg(angle) = tg(90-angle) = 1/tg(angle) needed because tg(90) -> inf
        # d = 1+2*b/a+(b/a)**2+4*b/a*c*c
        # x = ((1+b/a)+math.sqrt(d))/(2*c)
        h = 0
        if angle != math.pi /2:
            x = ((1 + b / a) * c_p + math.sqrt((1 + 2 * b / a + (b / a) ** 2) * c_p ** 2 + 4 * b / a)) / 2
            h = x * a
        else:
            h = math.sqrt \
                ( a *b  )  # len of perpendicular to hypotenuse is sqrt of multiplication of projections of sides onto hypotenuse

        print("h and p3" ,h, (np.linalg.norm(p3)), (p3))
        ans = math.sqrt(h ** 2 - np.linalg.norm(
            p3) ** 2)  # h is distance from p3 to focal point, because this line is not necessary perpendicular to focal plane and line that goes through center is, this step is needd
        return ans

    def calculate_p3(p1, p2):
        return np.dot(p1, p1 - p2) / np.linalg.norm(p1 - p2) ** 2 * (p2 - p1) + p1  # simplification of following

        # a = np.linalg.norm(p1)
        # b = np.linalg.norm(p2)
        # c = np.linalg.norm(p1-p2)

        a_p = np.dot(p1,
                     p2 - p1)  # same in case of relation as: np.dot(-p1,p2-p1)/c# same as: (np.dot(-p1,p2-p1)/a/c)*a
        b_p = np.dot(p2,
                     p1 - p2)  # same in case of relation as: np.dot(-p2,p1-p2)/c# same as: (np.dot(-p2,p1-p2)/b/c)*b
        return a_p / (a_p + b_p) * p1 + b_p / (a_p + b_p) * p2

    def angle_of_camera_v1(distance_to_fp, p3):
        """
        :return: angle between camera axis and horisontal line (if camera look at the horison, angle is 0)
        """
        return math.atan(np.linalg.norm(p3) / distance_to_fp)

    def homography_matrix_internal(p1, angle, scale, p3, distance_to_fp):
        """
        assuming camera axis lower than horizon :todo fix this
        :param p1:  is horizont point for vertical lines
        :param angle: angle of tilt perpendicular to horizon
        :param scale: relation of real field(in pixels) to pixels on image
        :param p3: p3
        :param distance_to_fp: one extensive parameter, distance to focal point
        :return: homography matrix
        """
        hdir = np.array([-p3[1], p3[0]])
        hdir /= np.linalg.norm(hdir)
        hp1 = hdir * 1000
        hp2 = -hdir * 1000
        hpr1 = hp1 * scale
        hpr2 = hp2 * scale
        # we dont want to accidentaly put one of point on horizon so we halfing angle between p3 and camera axis
        hp3 = p3 * math.tan(angle / 2) / math.tan(angle)
        p3_normed = p3 / np.linalg.norm(p3)
        hpr3 = distance_to_fp * scale * p3_normed  # because of new_angle * 2 = angle, which creates isosceles triangle
        hp4 = - distance_to_fp / math.tan(angle) * p3_normed
        hpr4 = -p3_normed * math.cos(angle) * scale * distance_to_fp
        # decomposition onto orthogonal basis
        p1_for_scale = p1 / 2
        p1_p_hdir = np.dot(hdir, p1_for_scale) * hdir
        p1_p_p3 = p1 - p1_p_hdir
        # projection of basis
        vert_angle = np.arctan(np.linalg.norm(p1_p_p3) / distance_to_fp)
        p1_p_p3r = p1_p_p3 * math.cos(vert_angle) / math.sin(angle - vert_angle) * scale  # some geometry
        p1_p_hdirr = p1_p_hdir * scale * math.sin(angle) * math.cos(vert_angle) / math.sin(
            angle - vert_angle)
        p1r = p1_p_p3r + p1_p_hdirr  # decompose vector into hp3 and hdir, project hdir with scale2(where scale2 computed from sinus theorem and scale), and hp3 with scale of (smth)

        cos_rot = np.dot(p1r, np.array((1, 0))) / np.linalg.norm(p1r)
        sin_rot = math.sqrt(1 - cos_rot ** 2)
        rot_mat = np.array([[cos_rot, sin_rot], [-sin_rot, cos_rot]])  # todo maybe wrong direction

        print(np.array([hp1, hp2, hp3, hp4]) + center_pos, np.array([hpr1, hpr2, hpr3, hpr4]) + center_pos, rot_mat,
              angle)
        hpr1 = rot_mat.dot(hpr1)
        hpr2 = rot_mat.dot(hpr2)
        hpr3 = rot_mat.dot(hpr3)
        hpr4 = rot_mat.dot(hpr4)
        if only_points:
            return np.array([hp1, hp2, hp3, hp4]) + center_pos, np.array([hpr1, hpr2, hpr3, hpr4]) + center_pos

        return cv2.findHomography(np.array([hp1, hp2, hp3, hp4]) + center_pos,
                                  np.array([hpr1, hpr2, hpr3, hpr4]) + center_pos)
        # return cv2.findHomography(np.array([hpr1,hpr2,hpr3,hpr4]),np.array([hp1,hp2,hp3,hp4])+400)
        # return cv2.getPerspectiveTransform(np.array([hp1,hp2,hp3,hp4]),np.array([hpr1,hpr2,hpr3,hpr4]))

    if True:
        hor_point1 = np.copy(hor_point1)
        hor_point2 = np.copy(hor_point2)
        hor_point1 -= center_pos
        hor_point2 -= center_pos
        # print("hor points: ", hor_point1, hor_point2)
        p3 = calculate_p3(hor_point1, hor_point2)
        distance = find_distance_fp_to_plane_v2(hor_point1, hor_point2, p3, angle=angle_between_lines)
        angle = angle_of_camera_v1(distance, p3)
        assert (abs(angle) > 1e-5)
        # print(distance, angle)
        return homography_matrix_internal(hor_point1, angle, 0.4, p3, distance)
