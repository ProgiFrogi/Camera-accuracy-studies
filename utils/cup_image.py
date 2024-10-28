import cv2

def get_cut_frame_from_video(path: str, frame_number: int, part : int = 1):
    width_start, width_end, height_start, height_end = 0, 479, 0, 359
    if (part == 1):
        width_start, width_end, height_start, height_end = 0, 479, 0, 359
    elif part == 2:
        width_start, width_end, height_start, height_end = 806, 1280, 0, 359
    elif part == 3:
        width_start, width_end, height_start, height_end = 0, 477, 362, 720
    elif part == 4:
        width_start, width_end, height_start, height_end = 806, 1280, 364, 720

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("Ошибка: не удалось открыть видео.")
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    success, frame = cap.read()
    if not success:
        frame_number = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame = cap.read()
    frame = frame[height_start:height_end, width_start:width_end]

    # cap.release()
    return (frame, frame_number)

def get_cut_frame_from_frame(frame, part : int = 1):
    width_start, width_end, height_start, height_end = 0, 479, 0, 359
    if (part == 1):
        width_start, width_end, height_start, height_end = 0, 479, 0, 359
    elif part == 2:
        width_start, width_end, height_start, height_end = 806, 1280, 0, 359
    elif part == 3:
        width_start, width_end, height_start, height_end = 0, 477, 362, 720
    elif part == 4:
        width_start, width_end, height_start, height_end = 806, 1280, 364, 720

    frame = frame[height_start:height_end, width_start:width_end]

    return (frame)

def cut_image(path: str):
    cv2.namedWindow('setting')
    cap = cv2.VideoCapture(path)

    if not cap.isOpened():
        print("Error: Can't open video stream or file.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def nothing(x):
        pass

    # Trackbars
    cv2.createTrackbar('height_start', 'setting', 0, height, nothing)
    cv2.createTrackbar('width_start', 'setting', 0, width, nothing)
    cv2.createTrackbar('height_end', 'setting', height, height, nothing)
    cv2.createTrackbar('width_end', 'setting', width, width, nothing)

    while True:
        success, frame = cap.read()
        if not success:
            # restart video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Get info from trackbars
        height_start = cv2.getTrackbarPos('height_start', 'setting')
        height_end = cv2.getTrackbarPos('height_end', 'setting')
        width_start = cv2.getTrackbarPos('width_start', 'setting')
        width_end = cv2.getTrackbarPos('width_end', 'setting')

        cropped_frame = frame[height_start:height_end, width_start:width_end]

        cv2.imshow("frame", cropped_frame)

        key = cv2.waitKey(10) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    i = 0
    path = '../materials_part1/1.mkv'
    while True:
        frame, i = get_cut_frame_from_video(path, i, 1)
        cv2.imshow('frame', frame)
