import cv2
import os
import json
import numpy as np
import Perspective as ps

imgPath = "image"

index = 0
images = os.listdir(imgPath)
win_ps_x = -1580
win_ps_y = -600
pts = []


def transform(img1, point1, point2, point3, point4):
    width = 400
    high = 800
    src = np.array([point1, point2, point3, point4], np.float32)
    dst = np.array([[0, 0], [width, 0], [width, high], [0, high]], np.float32)
    p = cv2.getPerspectiveTransform(src, dst)
    image_change = cv2.warpPerspective(img1, p, (width, high))
    return image_change


def get_img(event, x, y, flags, param):
    global img, img1, correct_img
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        pts.append(np.array([x, y]))
    if (len(pts) == 4):
        cv2.circle(img, tuple(pts[3]), 5, (0, 0, 255), -1)
        correct_img = ps.perspectiveTrans(img1,pts)
        #correct_img = transform(img1, pts[0], pts[1], pts[2], pts[3])
        pts.clear()
        cv2.namedWindow("get_temp")
        cv2.moveWindow("get_temp", win_ps_x, win_ps_y)
        cv2.imshow("get_temp", correct_img)
        cv2.setMouseCallback("get_temp", cutTemplate)


def cutTemplate(event, x, y, flags, param):
    global roiX, roiY, info, correct_img, imgname, index
    if event == cv2.EVENT_LBUTTONDOWN:
        roiX, roiY = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        ROI = correct_img[roiY:y, roiX:x].copy()
        cv2.rectangle(correct_img, (roiX, roiY), (x, y), (0, 0, 255), 2)
        cv2.imshow("get_temp", correct_img)
        cv2.namedWindow("label")
        cv2.moveWindow("label", win_ps_x, win_ps_y)
        cv2.imshow("label", ROI)
        label = input("输入label：")
        info["label"] = label
        info["imgname"] = imgname
        cv2.imwrite("template/" + imgname + "_" + str(index) + ".jpg", ROI)
        fp = open("config/" + imgname + "_" + str(index) + ".json", "w", encoding='utf-8')
        json.dump(info, fp, indent=4)
        index += 1


shrink = 4

for im in images:
    global img1, img
    print(im, "开始标注")
    img = cv2.imread(os.path.join(imgPath, im))
    img = cv2.resize(img, (0, 0), fx=1 / shrink, fy=1 / shrink)
    img1 = img.copy()
    imgname = im.split(".")[0]
    cv2.namedWindow("origin")
    cv2.moveWindow("origin", win_ps_x, win_ps_y)
    info = json.load(open("config/ConfigTemplate.json"))
    cv2.setMouseCallback('origin', get_img)
    index = 0
    while 1:
        cv2.imshow("origin", img)
        key = cv2.waitKey(10)
        if key & 0xFF == 9:  # 下一张，tab键
            cv2.destroyWindow("get_temp")

            break
        if key & 0xFF == 27:  # 重新标注 ESC
            cv2.destroyWindow("get_temp")
            img = img1.copy()

    print(im, "已经标好")

    cv2.destroyWindow("origin")
    cv2.waitKey(100)
