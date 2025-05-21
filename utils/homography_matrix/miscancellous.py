import numpy as np
import math
def convert_by_homography_mat(point,mat):
    """
    converts image using homography matrix
    can be used instead any linear transformation (i.e. a_1 = X*a_0+b ) and more
    """
    new_point = np.matmul(mat,np.concatenate([point,np.array([1])]))
    new_point/=new_point[2]
    return new_point[:2]
    

def homography_move(move_x,move_y):
    """
    returns homography matrix that moves image by move
    """
    return np.array([[1,0,move_x],[0,1,move_y],[0,0,1]])

def homography_move_v(move):
    """
    returns homography matrix that moves image by move
    """
    return homography_move(move[0],move[1])

def homography_scale(scale_x,scale_y):
    """
    returns homography matrix that scales image by scale along respective axes
    """
    return np.array([[scale_x,0,0],[0,scale_y,0],[0,0,1]])

def homography_scale_v(scale):
    """
    scale by vector
    """
    return homography_scale(scale[0],scale[1])

def homography_scale_s(scale):
    """
    scale by scalar
    """
    return homography_scale(scale,scale)

def homography_rotate(angle):
    """
    rotates image by angle (radians)
    """
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])

def homography_shear(direction,force):
    """
    example:
    ```
    -.-.-.
    -.-.-.
    -.-.-.
    ```
    converts into
    ```
            -.-.-.
        -.-.-.
    -.-.-.
    ```
    for direction = (1,0) and force = 1

    :param direction: direction is vector along which no distortion is happens
    :param force: for a point on the distance of 1 from direction vector how much should be added of direction vector
    """
    ort_vec = np.array([-direction[1],direction[0]])
    target = np.stack([direction,ort_vec+force*direction])
    input_vec = np.stack([direction,ort_vec])
    answer = np.identity(3)
    answer[:2,:2] = np.linalg.solve(input_vec,target)
    return answer

def homography_shear_angle(direction,angle):

    """
    example:
    ```
            -.-.-.
        -.-.-.
    -.-.-.
    ```
    converts into
    ```
    -.-.-.
    -.-.-.
    -.-.-.
    ```
    for direction = (1,0) and angle = pi/4

    :param direction: direction is vector along which no distortion is happens
    :param angle: angle of "unrotation"
    """
    return homography_shear(direction,-math.atan(angle))
    
def homography_mirror_x():
    """
    returns homography matrix that mirrors image along x axis
    """
    return np.array([[-1,0,0],[0,1,0],[0,0,1]])
