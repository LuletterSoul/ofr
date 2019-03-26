import cv2
import math
import numpy as np


def get_hight(points_list):
    tl = points_list[0]
    tr = points_list[1]
    br = points_list[2]
    bl = points_list[3]
    h1 = math.sqrt((tl[0] - bl[0]) ** 2 +
                   (tl[1] - bl[1]) ** 2)
    h2 = math.sqrt((tr[0] - br[0]) ** 2 +
                   (tr[1] - br[1]) ** 2)
    return int(max(h1, h2))


def get_width(points_list):
    tl = points_list[0]
    tr = points_list[1]
    br = points_list[2]
    bl = points_list[3]
    w1 = math.sqrt((tl[0] - tr[0]) ** 2 +
                   (tl[1] - tr[1]) ** 2)
    w2 = math.sqrt((br[0] - bl[0]) ** 2 +
                   (br[1] - bl[1]) ** 2)
    return int(max(w1, w2))


def perspectiveTrans(img, points_list):
    if len(points_list) != 4:
        print("Points list len must be 4.")
        return None
    for point in points_list:
        cv2.circle(img, tuple(point), 1, (255, 185, 15), 10)
    cv2.imwrite("points_type3.jpg", img)
    # cv2.imshow("test", img)
    # cv2.waitKey(0)
    print(get_width(points_list))
    print(get_hight(points_list))
    pts1 = np.float32(points_list)
    pts2 = np.float32([[0, 0], [get_width(points_list), 0], [get_width(points_list), get_hight(points_list)],
                       [0, get_hight(points_list)]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, M, (get_width(points_list), get_hight(points_list)))


if __name__ == '__main__':
    # image = cv2.imread("E://nju//projects//ODF//test1.jpg")
    # image = cv2.resize(image, (480, 360))
    # image = cv2.imread("test_2.jpg")
    # image = cv2.imread("E://nju//projects//ODF//type//type//type1.jpg")
    # image = cv2.imread("E://nju//projects//ODF//type//type//type2.jpg")
    # image = cv2.imread("E://nju//projects//ODF//type//type//type3.jpg")
    image = cv2.imread('image/IMG_9064.JPG')
    img = image
    print(image.shape)

    # test2
    # tl = [52, 55]
    # tr = [192, 53]
    # bl = [47, 230]
    # br = [195, 235]

    # type1
    # tl = [441, 134]
    # tr = [2201, 103]
    # bl = [628, 3991]
    # br = [2060, 4020]

    # type2
    # tl = [115, 525]
    # tr = [2475, 600]
    # bl = [155, 2660]
    # br = [2455, 2522]

    # type3
    tl = [449, 98]
    tr = [1998, 173]
    bl = [612, 3947]
    br = [1902, 3847]
    points_list = [tl, tr, br, bl]
    dst = perspectiveTrans(img, points_list)
    cv2.imwrite("perspective_type3.jpg", dst)
    # cv2.imshow("test", dst)
    # cv2.waitKey(0)
