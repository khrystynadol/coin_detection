import cv2
import os
add = "_trial"


def hough_circle_detection(panorama_path, min_r, max_r):
    image = cv2.imread(panorama_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 5)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        1,  # inverse ratio of accumulator res. to image res.
        40,  # 40,  # minimum distance between the centers of circles
        param1=80,  # 50,  # gradient value passed to edge detection
        # param1=50,  # much better for panorama_20191106_164512
        param2=30,  # 30,  # accumulator threshold for the circle centers
        minRadius=min_r*2,  # min circle radius
        maxRadius=max_r*2,  # max circle radius
    )

    if circles is not None:
        for circle in circles[0]:
            center = (int(circle[0]), int(circle[1]))
            radius = int(circle[2])
            cv2.circle(image, center, radius, (0, 255, 0), 2)
        cv2.imwrite(f'detected/sample_{panorama_path[19:-4]}{add}.jpg', image)
        cv2.imshow('Detected Circles', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return circles


def edge_detection(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    edges = cv2.Canny(blurred_image, 50, 150)

    cv2.imwrite(f'edges/edges_{image_path[19:-4]}{add}.jpg', edges)
    return edges


def video_to_images(video_path, output_folder, target_fps=2):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_skip = int(round(fps / target_fps))
    frame_number = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_number % frame_skip == 0:
            image_filename = os.path.join(output_folder, f"frame_{frame_number:04d}.jpg")
            cv2.imwrite(image_filename, frame)
        frame_number += 1
    cap.release()


def get_image_paths(folder_path, valid_extensions=('.jpg', '.jpeg', '.png', '.gif', '.bmp')):
    path_v = []
    if not os.path.exists(folder_path):
        print(f"Error: Folder not found - {folder_path}")
        return path_v
    all_files = os.listdir(folder_path)
    path_v = [os.path.join(folder_path, file) for file in all_files if file.lower().endswith(valid_extensions)]
    return path_v


def dist(x, y):
    return abs(x - y)


def knn(classified, r, k):
    distance = []
    for group in classified:
        for feature in classified[group]:
            dist_i = dist(feature, r)
            distance.append((dist_i, group))

    k1 = min(len(distance), k)
    sorted_dist = sorted(distance)
    distance = sorted_dist[:k1]

    # for idx in range(k1 - 1, len(sorted_dist) - 1):
    #     if sorted_dist[idx][0] == sorted_dist[idx + 1][0]:
    #         distance.append(sorted_dist[idx + 1])
    #     else:
    #         break
    # print("distance:", distance, k1)

    cnt = [0 for _ in range(len(classified))]
    for d1 in distance:
        cnt[d1[1]] += 1
    return cnt.index(max(cnt))


if __name__ == '__main__':
    # Part 1
    # video_name = f"video_2023-12-06_11-35-12.mp4"
    video_name = f"Copy of video6.mp4"
    # video_name = f"20191106_164450.mp4"
    # video_to_images(f"videos/{video_name}", f"images/images_{video_name[:-4]}{add}", target_fps=5)

    # video_name = f"20191106_164512.mp4"
    # video_to_images(f"videos/{video_name}", f"images/images_{video_name[:-4]}{add}", target_fps=10)
    image_paths = get_image_paths(f"images/images_{video_name[:-4]}{add}")
    images = [cv2.imread(image_path) for image_path in image_paths]

    # print(image_paths)
    # images = [cv2.imread(image_paths[i]) for i in range(0, len(image_paths), 2)]
    # images = [cv2.imread(image_paths[i]) for i in range(0, 227, 49)]

    stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)
    # stitcher.setPanoConfidenceThresh(0.7)
    status, panorama = stitcher.stitch(images)
    print("status", status)

    if status == cv2.Stitcher_OK:
        # cv2.imshow('Panorama', panorama)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        cv2.imwrite(f'panoramas/panorama_{video_name[:-4]}{add}.jpg', panorama)
    else:
        print("Error during stitching:", status)

    # # edge_detection(f'panoramas/panorama_{video_name[:-4]}.jpg')

    # Part 2
    # coins = hough_circle_detection(f"panoramas/panorama_video_2023-12-06_11-35-12.jpg", 0, 60)
    # # coins = hough_circle_detection(f"panoramas/panorama_Copy of video6.jpg", 0, 60)
    # # coins = hough_circle_detection(f"panoramas/panorama_20191106_164512.jpg", 0, 60)
    # radius_v = [coin[2] for coin in coins[0]]
    # sorted_v = radius_v
    # sorted_v.sort()
    # coins_cnt = len(coins[0])
    # # if coins is not None:
    # #     print(coins_cnt)
    # print(radius_v)
    #
    # d = {
    #     1: 16,
    #     2: 17.3,
    #     5: 24,
    #     10: 16.3,
    #     25: 20.8,
    #     50: 23,
    #     100: 26
    # }
    # old_range = d[100] - d[1]
    # new_range = sorted_v[-1] - sorted_v[0]
    # n, k = len(sorted_v), 1
    # ni = {}
    # idx = 0
    # for key, val in d.items():
    #     scaled_value = float(val - d[1]) / float(old_range)
    #     # print(key, val, scaled_value, (radius_v[0] + (scaled_value * new_range)))
    #     if key not in ni:
    #         ni[idx] = []
    #     ni[idx].append(sorted_v[0] + (scaled_value * new_range))
    #     idx += 1
    #
    # for j in range(1, n - 1):
    #     r = radius_v[j]
    #     # print(ni, r, k)
    #     res = knn(ni, r, k)
    #     ni[res].append(r)
    #     # print(r, res + 1, end=' ')
    #
    # add = {
    #     0: 1,
    #     1: 2,
    #     2: 5,
    #     3: 10,
    #     4: 25,
    #     5: 50,
    #     6: 100
    # }
    #
    # for key, val in ni.items():
    #     print(add[key], " (", len(val), ")", ": ", val, sep='')
    #
    # sum_gen = 0
    # for key, val in ni.items():
    #     sum_gen += len(val) * add[key]
    # print("Coins detected:", coins_cnt, "\nSum of nominals:", sum_gen)
