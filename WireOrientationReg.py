import cv2
import numpy as np
import os
import CalibrationUtil as cu
import Perspective as pv


def regOrientationBatch(img_dir):
    imgs = os.listdir(img_dir)
    for src in imgs:
        per_dir = 'image/' + src
        print("Image Name:", per_dir)
        img = cv2.imread(per_dir)
        _, pts = cu.select(img)
        print(pts)
        kernel_2 = np.ones((2, 2), np.uint8)  # 2x2的卷积核
        kernel_3 = np.ones((3, 3), np.uint8)  # 3x3的卷积核
        kernel_4 = np.ones((4, 4), np.uint8)  # 4x4的卷积核

        # 查表得到的HSV模型:黄色范围值
        minValues = np.array([26, 43, 46])
        maxValues = np.array([34, 255, 255])
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(img_hsv, minValues, maxValues)
        # 筛选出黄色部分
        img_yellow_org = cv2.bitwise_and(img, img, mask=mask)

        erosion = cv2.erode(mask, kernel_2, iterations=1)
        erosion = cv2.erode(erosion, kernel_2, iterations=1)
        dilation = cv2.dilate(erosion, kernel_2, iterations=1)
        dilation = cv2.dilate(dilation, kernel_2, iterations=1)
        img_yellow = cv2.bitwise_and(img, img, mask=dilation)
        # 二值化
        gray = cv2.cvtColor(img_yellow, cv2.COLOR_RGB2GRAY)
        retval, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        binary, contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 如果轮廓数目较多,去除较大的轮廓(线性轮廓像素点一般比较少)
        if len(contours) > 100:
            contours = [c for c in contours if len(c) > 150]
        if len(contours) <= 0:
            print("Not supported image content.")
            continue
        rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        thetas = []
        hor_cnt = 0
        ver_cnt = 0
        for contour in contours:
            rect = cv2.minAreaRect(contour)
            pts = np.int0(cv2.boxPoints(rect))
            m = cv2.moments(contour, True)
            if m['m00'] <= 0:
                continue
            # 计算轮廓的重心
            center = (int(m['m10'] / m['m00']), int(m['m01'] / m['m00']))
            a = m['m20'] / m['m00'] - np.power(center[0], 2)
            b = m['m11'] / m['m00'] - center[0] * center[1]
            c = m['m02'] / m['m00'] - np.power(center[1], 2)
            # 计算轮廓的方向,fastAtan2()返回的角度指自然坐标系下x轴正半轴按顺时针到图像轴的角
            theta = cv2.fastAtan2(2 * b, (a - c)) / 2
            # 水平轮廓投票计数
            if (0 <= theta <= 45) or (135 <= theta <= 180) or (180 <= theta <= 225) or (315 <= theta <= 360):
                hor_cnt += 1
            # 垂直轮廓投票计数
            else:
                ver_cnt += 1
            thetas.append(theta)
            # DEBUG :标识最小外接矩形
            cv2.drawContours(rgb, [pts], -1, (0, 0, 255), 3)
            # DEBUG :标识质心位置
            cv2.circle(rgb, center, 5, color=(0, 255, 0), thickness=cv2.FILLED, lineType=cv2.LINE_AA)
        print("HOR - VER cnt:", [hor_cnt, ver_cnt])
        # print("Thetas: ", thetas)
        if ver_cnt > hor_cnt:
            print("Orientation : Vertical")
        else:
            print("Orientation : Horizontal")
        print("\n\n")


def correctBatch(img_dir):
    imgs = os.listdir(img_dir)
    for src in imgs:
        per_dir = img_dir + src
        correct(per_dir, src)


def correct(per_dir, src):
    img = cv2.imread(per_dir)
    img = cv2.resize(img, (0, 0), fx=0.2, fy=0.2)
    _, pts = cu.select(img)
    # print(pts)
    dst = pv.perspectiveTrans(img, pts)
    if not dst:
        return
    filename, extension = os.path.splitext(src)
    output_name = 'output/' + filename + '_cor' + extension
    cv2.imwrite(output_name, dst)


if __name__ == '__main__':
    # regOrientationBatch('image')
    img_dir = 'image/'
    correctBatch(img_dir)
