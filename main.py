from functions import *
from drawing_functions import *

color = [(randint(10, 255), randint(10, 255), randint(10, 255))]*100

if __name__ == '__main__':
    input_video_path = 'video_cuted/3_right_up_cut.mp4'
    output_video_path = 'video/3_right_up_processed.mp4'

    cap = cv2.VideoCapture(input_video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Кодек для сохранения видео

    # Создаем объект для записи нового видео
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    while True:
        ret, image = cap.read()
        if not ret:
            break

        clustered_frame = convert_image(image)
        height, width, _ = clustered_frame.shape
        img_perimetr = 2 * (height + width)
        lines = find_lines(clustered_frame)

        if lines is not None:
            concatenate_lines = concatenate_line(clustered_frame,lines)
            Clusters = merge_in_clusters(clustered_frame, concatenate_lines)

            for cluster in Clusters:
                cluster.draw(clustered_frame)
            l = len(Clusters)
            for i in range(1, l):
                cluster1 = Clusters[i]
                l = np.shape(cluster1.get_lines_params())[0]
                S = cluster1.max_norm(clustered_frame)
                if l > 4 or l == 1 or S < 10:
                    continue

                for j in range(i):
                    cluster2 = Clusters[j]
                    l = np.shape(cluster2.get_lines_params())[0]
                    S = cluster2.max_norm(clustered_frame)
                    if l > 4 or l == 1 or S < 10:
                        continue
                    points = intersections_clusters(cluster1, cluster2)

                    in_img = []

                    for point in points:
                        if correct_point(clustered_frame, point):
                            in_img.append(point)
                        clustered_frame = draw_point(clustered_frame, point)

                    in_img = np.array(in_img)
                    if len(in_img) > 0:
                        center = np.mean(in_img, 0)
                        dxy = np.sum((in_img- center) ** 2, 1)
                        R = np.max(dxy) ** 0.5

                        cv2.circle(clustered_frame, (int(center[0]), int(center[1])), int(R) + 1, (0, 0, 255), 2)

        out.write(clustered_frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Видео сохранено по адресу: {output_video_path}")
