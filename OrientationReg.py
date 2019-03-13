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
for img in imgs:
    img_dir = 'image/' + img
    print("Image Name:", img_dir)
    img = cv2.imread(img_dir)
    plot.subImage(src=cv2.cvtColor(img, cv2.COLOR_RGB2BGR), index=inc(), title='Source', cmap='gray')
    kernel_2 = np.ones((2, 2), np.uint8)  # 2x2的卷积核
    kernel_3 = np.ones((3, 3), np.uint8)  # 3x3的卷积核
    kernel_4 = np.ones((4, 4), np.uint8)  # 4x4的卷积核

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
    retval, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    # plot.subImage(src=binary, index=inc(), title='Binary', cmap='gray')
    binary, contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 如果轮廓数目较多,去除较大的轮廓(线性轮廓像素点一般比较少)
    if len(contours) > 100:
        contours = [c for c in contours if len(c) > 150]
    if len(contours) <= 0:
        print("Not supported image content.")
        continue
    rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    # cv2.drawContours(rgb, contours, -1, (0, 0, 255), 2)
    # 求每个轮廓的最小边界矩形，得到矩形与x轴的偏移角度,范围为(-90,0]
    angles = [cv2.minAreaRect(c)[2] for c in contours]
    # 按区间[-90,-81],[-81,-72],[-72,-55]....[-9,0]区间统计角度分布情况
    his = np.histogram(angles, 10, [-90, 0])
    low45 = np.sum(his[0][0:5])  # 前半区间的数目和
    high45 = np.sum(his[0][6:])  # 后半区间的数目和
    print("Low - High his:", [low45, high45])
    print(his)
    avg = 0
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        avg += rect[2]
        pts = np.int0(cv2.boxPoints(rect))
        cv2.drawContours(rgb, [pts], -1, (0, 0, 255), 3)
    if len(contours) > 0:
        avg /= len(contours)
    print("Avg sum: ", avg)
    if low45 > high45:
        print("Orientation : Vertical")
    else:
        print("Orientation : Horizontal")
    print("\n\n")
# plot.subImage(src=cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), index=inc(), title="Contours")
# plot.show()
