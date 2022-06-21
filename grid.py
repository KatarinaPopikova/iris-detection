import numpy as np
import cv2 as cv
import csv
from collections import Counter

def find_corner(circle):
    return [int(circle[0]) - int(circle[2]), (int(circle[1]) + int(circle[2])) * (-1), int(circle[0]) + int(circle[2]),
            (int(circle[1]) - int(circle[2])) * (-1)]


def calculate_iou(right_circle, i):
    x_1, y_1, x_2, y_2 = find_corner(right_circle)
    x_3, y_3, x_4, y_4 = find_corner(i)
    area_inter = abs(min(x_2, x_4) - max(x_1, x_3)) * abs(min(y_2, y_4) - max(y_1, y_3))
    area_union = abs(x_2 - x_1) * abs(y_2 - y_1) + abs(x_4 - x_3) * abs(y_4 - y_3) - area_inter
    return area_inter / area_union


def show_best_circle(eye_image, bests_circle, tp):
    if bests_circle is not None:
        tp += 1
        cv.circle(eye_image, (int(bests_circle[0]), int(bests_circle[1])), int(bests_circle[2]),
                  (0, 255, 0), 2)
    return tp


def make_circles(circles, iris, pupilla):
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

        if (best_pupilla_circle is not None and best_iris_circle is not None):
            return True
    return False


def grid_search(eye_image, writer,iris, pupilla):
    h_e = 1
    for k in range(15, 19):
        for s in range(15, 36, 5):
            if s == 15:
                s = 1
            if k != 15 and s == 1:
                break
            for dp in range(2, 4):
                for minDist in range(15, 26, 5):
                    for param1 in range(70, 91, 5):
                        for param2 in range(120, 141, 5):
                            for minRadius in range(3, 10, 3):
                                if minRadius == 9:
                                    minRadius = 0
                                for maxRadius in range(140, 151, 2):
                                    if maxRadius == 180:
                                        maxRadius = 0

                                    if h_e:
                                        eye_image = cv.equalizeHist(eye_image)
                                    eye_image = cv.GaussianBlur(eye_image, (k * 2 + 1, k * 2 + 1), s * 0.1)

                                    if dp and minDist and param1 and param2:
                                        circles = cv.HoughCircles(eye_image, cv.HOUGH_GRADIENT, dp, minDist,
                                                                  param1=param1, param2=param2, minRadius=minRadius,
                                                                  maxRadius=maxRadius)

                                        if circles is not None and len(circles[0, :]) == 2:
                                            if make_circles(circles, iris, pupilla):
                                                data = [h_e, k, s, dp, minDist, param1, param2, minRadius,
                                                        maxRadius]
                                                print(data)
                                                writer.writerow(data)
                if s == 1:
                    break


def most_common():
    data = 'data/saveData'
    with open(data, 'r') as f:
        column = (row[0] for row in csv.reader(f))
        print("h_e: {0}".format(Counter(column).most_common()[0][0]))
    with open(data, 'r') as f:
        column = (row[1] for row in csv.reader(f))
        print("k: {0}".format(Counter(column).most_common()[0][0]))
    with open(data, 'r') as f:
        column = (row[2] for row in csv.reader(f))
        print("s: {0}".format(Counter(column).most_common()[0][0]))
    with open(data, 'r') as f:
        column = (row[3] for row in csv.reader(f))
        print("dp: {0}".format(Counter(column).most_common()[0][0]))
    with open(data, 'r') as f:
        column = (row[4] for row in csv.reader(f))
        print("minDist: {0}".format(Counter(column).most_common()[0][0]))
    with open(data, 'r') as f:
        column = (row[5] for row in csv.reader(f))
        print("param1: {0}".format(Counter(column).most_common()[0][0]))
    with open(data, 'r') as f:
        column = (row[6] for row in csv.reader(f))
        print("param2: {0}".format(Counter(column).most_common()[0][0]))
    with open(data, 'r') as f:
        column = (row[7] for row in csv.reader(f))
        print("minRadius: {0}".format(Counter(column).most_common()[0][0]))
    with open(data, 'r') as f:
        column = (row[8] for row in csv.reader(f))
        print("maxRadius: {0}".format(Counter(column).most_common()[0][0]))


if __name__ == '__main__':
    i = 0
    header = ['h_e', 'k', 's', 'dp', 'minDist', 'param1', 'param2', 'minRadius', 'maxRadius']

    with open('data/saveData2', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        with open("data/duhovky/iris_annotation.csv", 'r') as file:
            for row in file:
                if i < 2:
                    i += 1
                    continue
                path = 'data/duhovky/' + row[:18]
                print(path)
                iris = row.split(',')[1:4]
                pupilla = row.split(',')[4:7]

                img = cv.imread(cv.samples.findFile(path))
                eye_image = img.copy()
                eye_image = cv.cvtColor(eye_image, cv.COLOR_BGR2GRAY)

                grid_search(eye_image, writer, iris, pupilla)
    most_common()




