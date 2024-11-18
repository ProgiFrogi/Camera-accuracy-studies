import cv2
import numpy

import crop_image


def isOpen(name: str) -> bool:
    return cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) != 0.0


def nothing(x: int):
    pass


# def convert_to_points(table:numpy.ndarray)->numpy.ndarray:


if __name__ == '__main__':
    i = 500
    path = '../materials_part1/5.mkv'
    cam = cv2.VideoCapture(path)
    cv2.namedWindow('test')
    cv2.createTrackbar("lh", "test", 60, 255, nothing)
    cv2.createTrackbar("ls", "test", 28, 255, nothing)
    cv2.createTrackbar("lv", "test", 57, 255, nothing)
    cv2.createTrackbar("hh", "test", 168, 255, nothing)
    cv2.createTrackbar("hs", "test", 255, 255, nothing)
    cv2.createTrackbar("hv", "test", 255, 255, nothing)
    width = cam.get(cv2.CAP_PROP_FRAME_WIDTH)  # float `width`
    height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
    # previous = numpy.ndarray((int(height), int(width), 3), numpy.float64)
    previous = numpy.ndarray((359, 479, 3), numpy.float64)
    # print(width, height)
    previous_memorize_cntr = 100
    sum = 0
    pframe = 0
    isclosed = False
    # previous = numpy.concatenate(
    #     (cv2.imread("../tmp/v5_up_bacground.png"), cv2.imread("../tmp/v5_down2_bacground.png")), axis=0)
    previous = cv2.imread("../tmp/v5_bacground.png")
    # cv2.imwrite("../tmp/v5_bacground.png", previous)
    # cv2.imshow("test", previous)
    # while True:
    #     cv2.imshow("test", previous)
    #     cv2.waitKey(1)

    while not isclosed:

        isclosed |= not isOpen("test")
        # keyCode = cv2.waitKey(50)
        success, frame = cam.read()
        if success == False:
            cam.release()
            cam = cv2.VideoCapture(path)
            continue
        frame = crop_image.get_crop_frame_from_frame(frame, 1)
        # frame = frame[240:, :]
        pframe = frame
        # cv2.imwrite("../tmp/v5_down_bacground.png", frame)
        # break
        # if frame.shape != previous.shape:
        #     previous = numpy.ndarray(frame.shape, numpy.float64)
        # lh = cv2.getTrackbarPos("lh","test")
        # ls = cv2.getTrackbarPos("ls","test")
        # lv = cv2.getTrackbarPos("lv","test")
        # hh = cv2.getTrackbarPos("hh","test")
        # hs = cv2.getTrackbarPos("hs","test")
        # hv = cv2.getTrackbarPos("hv","test")
        # # print(frame.shape)
        # frame = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        # frame = cv2.inRange(frame,(lh,ls,lv),(hh,hs,hv))
        # previous_tmp = previous.astype(numpy.uint8)
        # previous_tmp = cv2.cvtColor(previous_tmp,cv2.COLOR_BGR2HSV)
        # previous_tmp = cv2.inRange(previous_tmp,(lh,ls,lv),(hh,hs,hv))
        # previous_tmp = previous_tmp.astype(numpy.float64)
        frame_modified = frame.astype(numpy.float64)
        # frame_modified = numpy.absolute(frame_modified - previous)
        # frame_modified = numpy.absolute(frame_modified - previous_tmp, 0)
        frame_modified = numpy.maximum(frame_modified - previous, 0)

        # if previous_memorize_cntr > 0:
        #     previous = frame * 0.1 + previous * 0.9
        #     previous_memorize_cntr -= 1
        # if True:
        #     previous = (frame + previous * sum) / (sum + 1)
        #     sum += 1
        # print(frame.dtype)
        frame = frame_modified.astype(numpy.uint8)
        lh = cv2.getTrackbarPos("lh", "test")
        ls = cv2.getTrackbarPos("ls", "test")
        lv = cv2.getTrackbarPos("lv", "test")
        hh = cv2.getTrackbarPos("hh", "test")
        hs = cv2.getTrackbarPos("hs", "test")
        hv = cv2.getTrackbarPos("hv", "test")
        frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(frame2, (lh, ls, lv), (hh, hh, hv))
        w = numpy.asarray([[1, 1, 1],
                           [1, 1, 1],
                           [1, 1, 1]], dtype=numpy.uint8)
        mask = cv2.filter2D(mask, -1, cv2.flip(w, -1), borderType=cv2.BORDER_CONSTANT)
        # mask = cv2.filter2D(mask, -1, cv2.flip(w, -1), borderType=cv2.BORDER_CONSTANT)

        # Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        # Set flags (Just to avoid line break in the code)
        flags = cv2.KMEANS_RANDOM_CENTERS
        # Apply KMeans
        # compactness, labels, centers = cv2.kmeans(mask, 2, None, criteria, 10, flags)

        # Apply the Component analysis function
        (totalLabels, label_ids, values, centroid) = cv2.connectedComponentsWithStats(mask,
                                                                                      4,
                                                                                      cv2.CV_32S)
        # print(values,totalLabels)
        mask = mask*0
        for i in range(totalLabels):
            # print(values[i])
            if(values[i][4]>250 and values[i][4]<500):
                mask = cv2.rectangle(mask, (values[i][0], values[i][1]), (values[i][0]+values[i][2],values[i][1]+values[i][3]), (155, 155, 155), 2)


        # break
        isclosed |= not isOpen("test")
        # cv2.imshow("test", frame)
        cv2.imshow("test", mask)
        cv2.waitKey(1)

    # cv2.imwrite("../tmp/v5_down2_bacground.png", pframe)
