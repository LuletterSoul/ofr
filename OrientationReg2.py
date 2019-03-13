import cv2
import numpy as np
import util.PlotUtil as plot
import os

plot_index = 0;


def inc():
    global plot_index
    plot_index += 1
    return plot_index


def reset():
    global plot_index
    plot_index = 0


imgs = os.listdir('image')
for src in imgs:
    img_dir = 'image/' + src
    print("Image Name:", img_dir)
    img = cv2.imread(img_dir)
    # plot.subImage(src=cv2.cvtColor(img, cv2.COLOR_RGB2BGR), index=inc(), title='Source', cmap='gray')
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

    # plot.subImage(src=cv2.cvtColor(img_yellow_org,cv2.COLOR_RGB2BGR), index=inc(), title='Org Yellow')
    erosion = cv2.erode(mask, kernel_2, iterations=1)
    erosion = cv2.erode(erosion, kernel_2, iterations=1)
    dilation = cv2.dilate(erosion, kernel_2, iterations=1)
    dilation = cv2.dilate(dilation, kernel_2, iterations=1)
    img_yellow = cv2.bitwise_and(img, img, mask=dilation)
    # plot.subImage(src=cv2.cvtColor(img_yellow, cv2.COLOR_RGB2BGR), index=inc(), title='Extract Yellow Color')
    # 二值化
    gray = cv2.cvtColor(img_yellow, cv2.COLOR_RGB2GRAY)
    retval, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    # plot.subImage(src=binary, index=inc(), title='Binary', cmap='gray')
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
    # plot.subImage(src=cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), index=inc(), title=src + " Contours")
#plot.save()
