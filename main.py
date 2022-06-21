import math

import numpy as np
import cv2 as cv
import sys


def create_trackbars():
    cv.namedWindow('slider', cv.WINDOW_NORMAL)
    cv.namedWindow('image', cv.WINDOW_NORMAL)
    cv.createTrackbar('Histogram equalisation', 'slider', 1, 1, nothing)
    cv.createTrackbar('Gaussian blur', 'slider', 1, 1, nothing)
    cv.createTrackbar('Kernel', 'slider', 18, int(min(img.shape[:2]) / 8), nothing)
    cv.createTrackbar('Sigma', 'slider', 20, 250, nothing)
    cv.createTrackbar('Canny', 'slider', 0, 1, nothing)
    cv.createTrackbar('Threshold', 'slider', 0, 255, nothing)
    cv.createTrackbar('Hough transform', 'slider', 1, 1, nothing)
    cv.createTrackbar('dp', 'slider', 2, 5, nothing)
    cv.createTrackbar('minDist', 'slider', 21, 255, nothing)
    cv.createTrackbar('param1', 'slider', 90, 255, nothing)
    cv.createTrackbar('param2', 'slider', 136, 300, nothing)
    cv.createTrackbar('minRadius', 'slider', 3, 30, nothing)
    cv.createTrackbar('maxRadius', 'slider', 142, 255, nothing)
    cv.createTrackbar('bitwiseNot', 'slider', 0, 1, nothing)
    cv.createTrackbar('scale', 'slider', 0, 40, nothing)
    cv.createTrackbar('segmentation', 'slider', 0, 1, nothing)


def nothing(x):
    pass


def find_corner(circle):
    return [int(circle[0]) - int(circle[2]), (int(circle[1]) + int(circle[2])) * (-1), int(circle[0]) + int(circle[2]),
            (int(circle[1]) - int(circle[2])) * (-1)]


def calculate_iou(right_circle, i):
    x_1, y_1, x_2, y_2 = find_corner(right_circle)
    x_3, y_3, x_4, y_4 = find_corner(i)
    area_inter = abs(min(x_2, x_4) - max(x_1, x_3)) * abs(min(y_2, y_4) - max(y_1, y_3))
    area_union = abs(x_2 - x_1) * abs(y_2 - y_1) + abs(x_4 - x_3) * abs(y_4 - y_3) - area_inter
    return area_inter / area_union


def evaluate_circles(eye_image, tp, fp):
    fn = 2 - tp
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    evaluation = "TP: {}, FP: {} FN: {}, precision: {}, recall: {}".format(tp, fp, fn, precision, recall)

    cv.putText(eye_image, evaluation, org=(0, 250), fontFace=cv.FONT_HERSHEY_DUPLEX,
               fontScale=0.3, color=(0, 0, 0), thickness=1)


def show_best_circle(eye_image, bests_circle, tp):
    if bests_circle is not None:
        tp += 1
        cv.circle(eye_image, (int(bests_circle[0]), int(bests_circle[1])), int(bests_circle[2]),
                  (0, 255, 0), 2)
    return tp


def make_segmentation(eye_image, best_pupilla_circle, best_iris_circle):
    seg = cv.getTrackbarPos('segmentation', 'slider')
    if not seg:
        return
    for y in range(0, img.shape[0]):
        for x in range(0, img.shape[1]):
            length_pupilla = math.hypot(x - int(best_pupilla_circle[0]), y - int(best_pupilla_circle[1]))
            length_iris = math.hypot(x - int(best_iris_circle[0]), y - int(best_iris_circle[1]))
            if length_pupilla < int(best_pupilla_circle[2]) and length_iris > int(best_iris_circle[2]):
                eye_image[y][x][0] = 255
                eye_image[y][x][1] = 255
                eye_image[y][x][2] = 255
            else:
                eye_image[y][x][0] = 0
                eye_image[y][x][1] = 0
                eye_image[y][x][2] = 0


def make_circles(eye_image, circles, iris, pupilla):
    best_pupilla_circle = None
    best_iris_circle = None
    iou_min = 0.75
    best_iris_iou = 0
    best_pupilla_iou = 0

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            iris_iou = calculate_iou(iris, i)
            if iou_min <= iris_iou > best_iris_iou:
                best_iris_iou = iris_iou
                best_iris_circle = i.copy()

            pupilla_iou = calculate_iou(pupilla, i)
            if iou_min <= pupilla_iou > best_pupilla_iou:
                best_pupilla_iou = pupilla_iou
                best_pupilla_circle = i.copy()

            cv.circle(eye_image, (i[0], i[1]), i[2], (204, 255, 255), 2)

        cv.circle(eye_image, (int(iris[0]), int(iris[1])), int(iris[2]), (214, 219, 67), 2)
        cv.circle(eye_image, (int(pupilla[0]), int(pupilla[1])), int(pupilla[2]), (214, 219, 67), 2)

        tp = show_best_circle(eye_image, best_pupilla_circle, 0)
        tp = show_best_circle(eye_image, best_iris_circle, tp)

        if tp == 2:
            make_segmentation(eye_image, best_pupilla_circle, best_iris_circle)

        evaluate_circles(eye_image, tp, len(circles[0, :]) - tp)


if __name__ == '__main__':
    path = "data/duhovky/001/L/S1001L07.jpg"
    # path = "data/duhovky/001/R/S1001R09.jpg"
    # path = "data/duhovky/001/R/S1001R01.jpg"
    # path = "data/duhovky/001/L/S1001L02.jpg"
    # path = "data/duhovky/007/R/S1007R10.jpg"
    # path = "data/duhovky/002/R/S1002R01.jpg"
    img = cv.imread(cv.samples.findFile(path))
    if img is None:
        sys.exit("Could not read the image.")

    create_trackbars()

    with open("data/duhovky/iris_annotation.csv", 'r') as file:
        substring = path[13:]
        for row in file:
            if substring in row:
                iris = row.split(',')[1:4]
                pupilla = row.split(',')[4:7]
    while True:
        eye_image = img.copy()
        eye_image = cv.cvtColor(eye_image, cv.COLOR_BGR2GRAY)
        k = cv.waitKey(1) & 0xFF
        if k == 27:
            break

        h_e = cv.getTrackbarPos('Histogram equalisation', 'slider')
        g_b = cv.getTrackbarPos('Gaussian blur', 'slider')
        s = cv.getTrackbarPos('Sigma', 'slider')
        k = cv.getTrackbarPos('Kernel', 'slider')
        c_a = cv.getTrackbarPos('Canny', 'slider')
        t = cv.getTrackbarPos('Threshold', 'slider')
        h_t = cv.getTrackbarPos('Hough transform', 'slider')
        dp = cv.getTrackbarPos('dp', 'slider')
        minDist = cv.getTrackbarPos('minDist', 'slider')
        param1 = cv.getTrackbarPos('param1', 'slider')
        param2 = cv.getTrackbarPos('param2', 'slider')
        minRadius = cv.getTrackbarPos('minRadius', 'slider')
        maxRadius = cv.getTrackbarPos('maxRadius', 'slider')
        bitwiseNot = cv.getTrackbarPos('bitwiseNot', 'slider')
        scale = cv.getTrackbarPos('scale', 'slider')

        if h_e:
            eye_image = cv.equalizeHist(eye_image)
        if g_b:
            eye_image = cv.GaussianBlur(eye_image, (k * 2 + 1, k * 2 + 1), s * 0.1)
        if t and c_a:
            eye_image = cv.Canny(eye_image, t / 2, t)
        if bitwiseNot:
            eye_image = cv.bitwise_not(eye_image)
        if scale:
            eye_image = cv.convertScaleAbs(eye_image, alpha=scale * 0.1)

        if dp and minDist and param1 and param2 and h_t:
            circles = cv.HoughCircles(eye_image, cv.HOUGH_GRADIENT, dp, minDist,
                                      param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
            eye_image = cv.cvtColor(eye_image, cv.CV_8UC1)

            make_circles(eye_image, circles, iris, pupilla)

        cv.putText(eye_image, path[13:], org=(0, 270), fontFace=cv.FONT_HERSHEY_DUPLEX,
                   fontScale=0.3, color=(0, 0, 0), thickness=1)
        cv.imshow('image', eye_image)

    cv.destroyAllWindows()
