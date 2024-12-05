
import cv2
from ball_detection import nothing,isOpen

def tune_test(path:str):
    cv2.namedWindow('test')
    cv2.createTrackbar("lh", "test", 60, 255, nothing)
    cv2.createTrackbar("ls", "test", 28, 255, nothing)
    cv2.createTrackbar("lv", "test", 57, 255, nothing)
    cv2.createTrackbar("hh", "test", 168, 255, nothing)
    cv2.createTrackbar("hs", "test", 255, 255, nothing)
    cv2.createTrackbar("hv", "test", 255, 255, nothing)
    isclosed = False
    lh = ls=lv=hh=hs=hv =0
    if path.endswith('.mkv'):
        cam = cv2.VideoCapture(path)
        while not isclosed:
            isclosed |= not isOpen("test")
            if isclosed:
                break
            success, frame = cam.read()
            lh = cv2.getTrackbarPos("lh", "test")
            ls = cv2.getTrackbarPos("ls", "test")
            lv = cv2.getTrackbarPos("lv", "test")
            hh = cv2.getTrackbarPos("hh", "test")
            hs = cv2.getTrackbarPos("hs", "test")
            hv = cv2.getTrackbarPos("hv", "test")
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            frame = cv2.inRange(frame, (lh, ls, lv), (hh, hs, hv))
            isclosed |= not isOpen("test")
            if isclosed:
                break
            # cv2.imshow("test", frame)
            cv2.imshow("test", frame)
            cv2.waitKey(1)
    else:
        frame = cv2.imread(path)
        while not isclosed:
            isclosed |= not isOpen("test")
            if isclosed:
                break
            lh = cv2.getTrackbarPos("lh", "test")
            ls = cv2.getTrackbarPos("ls", "test")
            lv = cv2.getTrackbarPos("lv", "test")
            hh = cv2.getTrackbarPos("hh", "test")
            hs = cv2.getTrackbarPos("hs", "test")
            hv = cv2.getTrackbarPos("hv", "test")
            frame_t = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            frame_t = cv2.inRange(frame_t, (lh, ls, lv), (hh, hs, hv))
            isclosed |= not isOpen("test")
            if isclosed:
                break
            # cv2.imshow("test", frame)
            cv2.imshow("test", frame_t)
            cv2.waitKey(1)
    print(lh,ls,lv,hh,hs,hv)

def tune_from_params(image,lh,ls,lv,hh,hs,hv):
    frame = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    frame = cv2.inRange(frame, (lh, ls, lv), (hh, hs, hv))
    return frame

def rgb_tune_from_params(image,lh,ls,lv,hh,hs,hv):
    frame = cv2.inRange(image, (lh, ls, lv), (hh, hs, hv))
    return frame


def rgb_tune_test(path:str):
    cv2.namedWindow('test')
    cv2.createTrackbar("hr", "test", 60, 255, nothing)
    cv2.createTrackbar("hg", "test", 28, 255, nothing)
    cv2.createTrackbar("hb", "test", 57, 255, nothing)
    cv2.createTrackbar("lr", "test", 168, 255, nothing)
    cv2.createTrackbar("lg", "test", 255, 255, nothing)
    cv2.createTrackbar("lb", "test", 255, 255, nothing)
    isclosed = False
    lh = ls=lv=hh=hs=hv =0
    if path.endswith('.mkv'):
        cam = cv2.VideoCapture(path)
        while not isclosed:
            isclosed |= not isOpen("test")
            if isclosed:
                break
            success, frame = cam.read()
            lh = cv2.getTrackbarPos("lr", "test")
            ls = cv2.getTrackbarPos("lg", "test")
            lv = cv2.getTrackbarPos("lb", "test")
            hh = cv2.getTrackbarPos("hr", "test")
            hs = cv2.getTrackbarPos("hg", "test")
            hv = cv2.getTrackbarPos("hb", "test")
            frame = cv2.inRange(frame, (lh, ls, lv), (hh, hs, hv))
            isclosed |= not isOpen("test")
            if isclosed:
                break
            # cv2.imshow("test", frame)
            cv2.imshow("test", frame)
            cv2.waitKey(1)
    else:
        frame = cv2.imread(path)
        while not isclosed:
            isclosed |= not isOpen("test")
            if isclosed:
                break
            lh = cv2.getTrackbarPos("lr", "test")
            ls = cv2.getTrackbarPos("lg", "test")
            lv = cv2.getTrackbarPos("lb", "test")
            hh = cv2.getTrackbarPos("hr", "test")
            hs = cv2.getTrackbarPos("hg", "test")
            hv = cv2.getTrackbarPos("hb", "test")
            frame_t = cv2.inRange(frame, (lh, ls, lv), (hh, hs, hv))
            isclosed |= not isOpen("test")
            if isclosed:
                break
            # cv2.imshow("test", frame)
            cv2.imshow("test", frame_t)
            cv2.waitKey(1)
    print(lh,ls,lv,hh,hs,hv)