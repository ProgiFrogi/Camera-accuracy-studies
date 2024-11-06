import cv2
def detect_marker(path):
    cap = cv2.VideoCapture(path)

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    while True:
        ret, frame = cap.read()

        markerCorners, markerIds, rejectedCandidates = detector.detectMarkers(frame)

        new_frame = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)

        cv2.imshow('Found markers2', new_frame)
        if cv2.waitKey(2) & 0xFF == ord('q'):
            break

def detect_markers_return_ids(img, aruco_type):
    dictionary = cv2.aruco.getPredefinedDictionary(aruco_type)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    _, markerIds, _ = detector.detectMarkers(img)
    return markerIds




if __name__ == '__main__':
    path = '../materials_part1/1.mkv'

    cap = cv2.VideoCapture(path)

    while True:
        ret, frame = cap.read()

        print(detect_markers_return_ids(frame))
        if cv2.waitKey(2) & 0xFF == ord('q'):
            break
