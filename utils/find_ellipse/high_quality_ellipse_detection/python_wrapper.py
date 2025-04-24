import cv2
import os


def find_ellipses_fast(img,edge_detector="canny",gradient=0):
    """
    :param img: cv2 image where detect ellipses
    :param edge_detector: either "canny" or "sobel"
    :param gradient: one of [-1,0,1], shows direction of gradient on ellipse boundary relative to its center
    :return: list[list[5 float]] , [[posx,posy,axis1,axis2,rotation]] (cv2.ellipse draws this data correctly)
    """
    detector_type =0
    if edge_detector == "canny":
        detector_type = 1
    elif edge_detector == "sobel":
        detector_type = 2
    else:
        raise RuntimeError("wrong edge detector")
    current_working_directory = os.getcwd()
    current_file_path = os.path.abspath(__file__)
    current_directory = os.path.dirname(current_file_path)
    os.chdir(current_directory)
    cv2.imwrite(current_directory+"/tmp.png",img)
    os.system("./a.out tmp.png "+str(detector_type)+" "+str(gradient))
    os.chdir(current_working_directory)
    ellipses = []
    with open(current_directory+"/output_candidate_ellipses.txt", 'r') as f:
        text = f.read().split('\n')
        for i in text:
            if i == '':
                continue
            ellipses.append(list(map(float,i.split(' '))))
    return ellipses

# img = cv2.imread('tmp/hehe4.png')
# print(find_ellipses_fast(img))